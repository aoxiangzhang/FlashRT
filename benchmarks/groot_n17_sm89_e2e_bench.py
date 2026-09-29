"""Full synthetic-e2e latency bench for the GROOT N1.7 SM89 fusions.

Assembles every backbone stage of the SM89 path with the frontend's real
dimensions (no checkpoint needed — random weights, stub attention) and
times the WHOLE chain fused vs the legacy-chain shim. Backbone runs once
per inference; the DiT head runs num_inference_timesteps (4) times.

Real dims (from flash_rt/frontends/torch/groot_n17_rtx_sm89.py):
  ViT:    S=1024 (2 views x 512), D=1024, NH=16, HD=64, FF=4096, 24 layers
  DeepStack: 3 mergers, Nout=256, Din=1024 -> Dmid=? (frontend value),
          Dout=2048
  LLM:    Se=768 (prompt-dependent; mid estimate), D=2048, NHQ=16,
          NHKV=8, HD=128, FF=6144, 16 layers
  VL:     T=Se, D=2048, NH=32, HD=64, FF=8192, 4 layers
  DiT:    Sa=41, D=1536, FF=6144, 32 layers x 4 steps

Run:
    python benchmarks/groot_n17_sm89_e2e_bench.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flash_rt.flash_rt_kernels as fvk  # noqa: E402
from tests.test_groot_n17_sm89_ele_fusion import (  # noqa: E402
    _legacy_fvk_shim, _llm_fixture, _p, _StubAttn, _StubDiTAttn,
    _DEV, _FP16,
)
from flash_rt.models.groot_n17 import pipeline_rtx_sm89 as P  # noqa: E402

_BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
KEEP = []


def _rnd(*shape, dtype=_FP16):
    return torch.randn(*shape, device=_DEV, dtype=dtype)


def _fp8w(N, K, scale_w=0.02):
    w = (torch.randn(N, K, device=_DEV, dtype=torch.float32) * scale_w).to(FP8)
    ws = torch.tensor([max(scale_w * 3.0 / 448.0, 1e-8)],
                      device=_DEV, dtype=torch.float32)
    KEEP.append(w); KEEP.append(ws)
    return int(w.data_ptr()), int(ws.data_ptr())


def _act_scale():
    s = torch.full((1,), max(4.0 / 448.0, 1e-8), device=_DEV,
                   dtype=torch.float32)
    KEEP.append(s)
    return int(s.data_ptr())


class _MultiSiteAttn:
    """Site-aware stub: one (S, D) O-tensor per site group, deterministic
    fill. Q/K/V slot pointers alias a scratch the GEMMs may clobber."""

    def __init__(self, sites):
        self.sites = sites          # {"vit": (S, D), "llm": (S, D), ...}
        self.O = {k: torch.zeros(*shp, device=_DEV, dtype=_FP16)
                  for k, shp in sites.items()}
        self._state = {k: torch.randn(*shp, device=_DEV, dtype=_FP16)
                       for k, shp in sites.items()}

    def get_slot_ptrs(self, site, li):
        return {"Q": self.O[site].data_ptr(), "K": self.O[site].data_ptr(),
                "V": self.O[site].data_ptr(), "O": self.O[site].data_ptr()}

    def run(self, site, li, *, q_seq, kv_seq=None, stream=0):
        self.O[site].copy_(self._state[site])
        return self.O[site].data_ptr()


def build_backbone(Se=768):
    """All fixtures in one go; returns a run(fvk) closure + attn."""
    Sv, nv = 1024, 2
    Sper = Sv // nv

    # ── ViT fixture (24 layers, D=1024, FF=4096) ──
    torch.manual_seed(301)
    Dv, NHv, HDv, FFv, Lv = 1024, 16, 64, 4096, 24
    vw = {k: [] for k in
          ("norm1_w", "norm1_b", "norm2_w", "norm2_b",
           "q_w", "q_b", "k_w", "k_b", "v_w", "v_b", "o_w", "o_b",
           "fc1_w", "fc1_b", "fc2_w", "fc2_b",
           "q_ws", "k_ws", "v_ws", "o_ws", "fc1_ws", "fc2_ws",
           "cos", "sin")}
    for li in range(Lv):
        for k in ("norm1_w", "norm2_w"):
            t = _rnd(Dv); KEEP.append(t); vw[k].append(t.data_ptr())
        for k in ("norm1_b", "norm2_b"):
            t = _rnd(Dv); KEEP.append(t); vw[k].append(t.data_ptr())
        for k, N, K in (("q_w", Dv, Dv), ("k_w", Dv, Dv), ("v_w", Dv, Dv),
                        ("o_w", Dv, Dv), ("fc1_w", FFv, Dv),
                        ("fc2_w", Dv, FFv)):
            wp, wsp = _fp8w(N, K)
            vw[k].append(wp); vw[k.replace("_w", "_ws")].append(wsp)
        for k, N in (("q_b", Dv), ("k_b", Dv), ("v_b", Dv), ("o_b", Dv),
                     ("fc1_b", FFv), ("fc2_b", Dv)):
            t = _rnd(N); KEEP.append(t); vw[k].append(t.data_ptr())
    cos_v = _rnd(Sv, HDv); sin_v = _rnd(Sv, HDv); KEEP.extend([cos_v, sin_v])
    vw["cos"], vw["sin"] = cos_v.data_ptr(), sin_v.data_ptr()
    vit_scales = {k: [_act_scale() for _ in range(Lv)]
                  for k in ("act_qkv", "act_o", "act_fc1", "act_fc2")}
    vit_bufs = {"h": _rnd(Sv, Dv), "xn": _rnd(Sv, Dv),
                "xn_fp8": torch.empty(Sv, Dv, device=_DEV, dtype=FP8),
                "o_proj_out": _rnd(Sv, Dv), "fc1_out": _rnd(Sv, FFv),
                "fc1_fp8": torch.empty(Sv, FFv, device=_DEV, dtype=FP8),
                "bf16_tmp": _rnd(Sv, Dv, dtype=_BF16),
                "bf16_ff": _rnd(Sv, FFv, dtype=_BF16)}
    KEEP.append(vit_bufs)
    vit_bufs = {k: v.data_ptr() for k, v in vit_bufs.items()}
    vit_dims = {"S": Sv, "D": Dv, "NH": NHv, "HD": HDv, "ff_inner": FFv,
                "Sper_view": Sper}
    taps = [5, 11, 17]
    dcap = [torch.empty(Sv, Dv, device=_DEV, dtype=_FP16) for _ in taps]
    KEEP.extend(dcap)

    def mk_cb(i):
        def cb(h_ptr):
            fvk.gpu_copy(dcap[i].data_ptr(), int(h_ptr), Sv * Dv * 2, 0)
        return cb

    dcap_cbs = [mk_cb(i) for i in range(len(taps))]
    dcap_ptrs = [t.data_ptr() for t in dcap]

    # ── DeepStack (3 mergers, Nout=256, Din=1024, Dmid=4096, Dout=2048) ──
    No, Din, Dmid, Dout = Sv // 4, Dv, 4096, 2048
    dsw = {k: [] for k in ("norm_w", "norm_b", "fc1_w", "fc1_b",
                           "fc2_w", "fc2_b", "fc1_ws", "fc2_ws")}
    for j in range(3):
        for k in ("norm_w", "norm_b"):
            # LN runs over Dmid (4096) — the merger LN input row width
            t = _rnd(Dmid); KEEP.append(t); dsw[k].append(t.data_ptr())
        # fc1: M=Nout, N=Dmid, K=Dmid (tap rows viewed (Nout, Dmid))
        f1p, f1s = _fp8w(Dmid, Dmid); f2p, f2s = _fp8w(Dout, Dmid)
        dsw["fc1_w"].append(f1p); dsw["fc1_ws"].append(f1s)
        dsw["fc2_w"].append(f2p); dsw["fc2_ws"].append(f2s)
        for k, N in (("fc1_b", Dmid), ("fc2_b", Dout)):
            t = _rnd(N); KEEP.append(t); dsw[k].append(t.data_ptr())
    _ln_out = _rnd(No, Din)
    _fp8_scr = torch.empty(max(No * Din, No * Dmid), device=_DEV,
                           dtype=FP8)
    _fc1_out = _rnd(No, Dmid)
    _ds_out = [_rnd(No, Dout) for _ in range(3)]
    _bf_t = _rnd(No, Dout, dtype=_BF16)
    _bf_f = _rnd(No, Dmid, dtype=_BF16)
    KEEP.extend([_ln_out, _fp8_scr, _fc1_out, *_ds_out, _bf_t, _bf_f])
    ds_bufs = {"in": dcap_ptrs, "ln_out": _ln_out.data_ptr(),
               "fp8_scratch": _fp8_scr.data_ptr(),
               "fc1_out": _fc1_out.data_ptr(),
               "out": [t.data_ptr() for t in _ds_out],
               "bf16_tmp": _bf_t.data_ptr(), "bf16_ff": _bf_f.data_ptr()}
    ds_scales = {"act_fc1": [_act_scale() for _ in range(3)],
                 "act_fc2": [_act_scale() for _ in range(3)]}
    ds_dims = {"Nin": Sv, "Din": Din, "Nout": No, "Dmid": Dmid,
               "Dout": Dout}

    # ── LLM (reuse the test fixture shape logic, Se tokens) ──
    _, gemm, llm_dims, mk_w, mk_b, llm_scales, _, _keep = _llm_fixture(Se)
    lw, lb = mk_w(), mk_b()

    # ── vlln + VL self-attn (T=Se, D=2048, NH=32, HD=64, FF=8192, 4L) ──
    Dl, NHl, HDl, FFl, Ll = 2048, 32, 64, 8192, 4
    vsw = {k: [] for k in
           ("norm1_w", "norm1_b", "norm3_w", "norm3_b",
            "q_w", "q_b", "k_w", "k_b", "v_w", "v_b", "o_w", "o_b",
            "fc1_w", "fc1_b", "fc2_w", "fc2_b",
            "q_ws", "k_ws", "v_ws", "o_ws", "fc1_ws", "fc2_ws")}
    for li in range(Ll):
        for k in ("norm1_w", "norm3_w", "norm1_b", "norm3_b"):
            t = _rnd(Dl); KEEP.append(t); vsw[k].append(t.data_ptr())
        for k, N, K in (("q_w", Dl, Dl), ("k_w", Dl, Dl), ("v_w", Dl, Dl),
                        ("o_w", Dl, Dl), ("fc1_w", FFl, Dl),
                        ("fc2_w", Dl, FFl)):
            wp, wsp = _fp8w(N, K)
            vsw[k].append(wp); vsw[k.replace("_w", "_ws")].append(wsp)
        for k, N in (("q_b", Dl), ("k_b", Dl), ("v_b", Dl), ("o_b", Dl),
                     ("fc1_b", FFl), ("fc2_b", Dl)):
            t = _rnd(N); KEEP.append(t); vsw[k].append(t.data_ptr())
    vl_scales = {k: [_act_scale() for _ in range(Ll)]
                 for k in ("act_qkv", "act_o", "act_fc1", "act_fc2")}
    vl_bufs = {"h": _rnd(Se, Dl), "xn": _rnd(Se, Dl),
               "xn_fp8": torch.empty(Se, Dl, device=_DEV, dtype=FP8),
               "o_proj_out": _rnd(Se, Dl), "fc1_out": _rnd(Se, FFl),
               "fc1_fp8": torch.empty(Se, FFl, device=_DEV, dtype=FP8),
               "bf16_tmp": _rnd(Se, Dl, dtype=_BF16),
               "bf16_ff": _rnd(Se, FFl, dtype=_BF16)}
    KEEP.append(vl_bufs)
    vl_bufs = {k: v.data_ptr() for k, v in vl_bufs.items()}
    vl_dims = {"T": Se, "D": Dl, "NH": NHl, "HD": HDl, "ff_inner": FFl}
    vlln_w = _rnd(Dl); vlln_b = _rnd(Dl); KEEP.extend([vlln_w, vlln_b])
    llm_h = torch.empty(Se, Dl, device=_DEV, dtype=_FP16); KEEP.append(llm_h)

    attn = _MultiSiteAttn({"vit": (Sv, Dv), "llm": (Se, Dl),
                           "vl_self_attn": (Se, Dl)})

    # ── DiT fixture (32 layers, 4 steps) ──
    from tests.test_groot_n17_sm89_ele_fusion import _dit_fixture
    Pth, _, dit_dims, mk_dw, mk_db, dkeep, (Sa, Dd) = _dit_fixture()
    dw, db = mk_dw(), mk_db()
    dit_attn = _StubDiTAttn(Sa, Dd)

    def run_backbone(fvkm, mask=(1, 1, 1, 1, 1, 1)):
        m_vit, m_ds, m_llm, m_vlln, m_vl, m_dit = mask
        if m_vit:
            P.qwen3vl_vit_forward(
                gemm, fvkm, vit_bufs, vw, vit_dims, vit_scales,
                attn=attn, deepstack_taps=taps, deepstack_capture=dcap_cbs,
                stream=0)
        if m_ds:
            P.deepstack_merge_forward(
                gemm, fvkm, ds_bufs, dsw, ds_dims, ds_scales, stream=0)
        if m_llm:
            P.qwen3vl_llm_forward(
                gemm, fvkm, lb, lw, llm_dims, llm_scales, attn=attn,
                stream=0)
        if m_vlln:
            P.vlln_forward(
                gemm, fvkm,
                {"x": lb["h"], "out": llm_h.data_ptr()},
                {"vlln_w": vlln_w.data_ptr(), "vlln_b": vlln_b.data_ptr()},
                {"S": Se, "D": Dl})
        if m_vl:
            P.vl_self_attn_forward(
                gemm, fvkm, vl_bufs, vsw, vl_dims, vl_scales, attn=attn,
                stream=0)
        if m_dit:
            for _ in range(4):
                Pth.dit_forward(
                    gemm, fvkm, db, dw, dit_dims, attn=dit_attn, stream=0)

    return run_backbone, fvk, _legacy_fvk_shim()


def bench_e2e(Se=768, iters=100):
    run_backbone, fvkm_fused, fvkm_legacy = build_backbone(Se)
    shim = fvkm_legacy
    # legacy leg = every fused entry point replaced by its pre-PR chain

    def lbr(res, x, bias, M, N, stream=0):
        fvkm_fused.add_bias_bf16(x, bias, M, N)
        fvkm_fused.residual_add(res, x, M * N)

    def lbg(x, bias, M, N, stream=0):
        fvkm_fused.add_bias_bf16(x, bias, M, N)
        fvkm_fused.gelu_inplace(x, M * N)
    shim.bias_residual_strict_bf16 = lbr
    shim.bias_gelu_bf16_strict = lbg

    def time_leg(fn):
        for _ in range(15):
            fn()
        torch.cuda.synchronize()
        st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            st[i].record(); fn(); en[i].record()
        torch.cuda.synchronize()
        return sorted(s.elapsed_time(e)
                      for s, e in zip(st, en))[iters // 2]

    t_legacy = time_leg(lambda: run_backbone(shim))
    t_fused = time_leg(lambda: run_backbone(fvkm_fused))
    print(f"[e2e synthetic] Se={Se} (ViT 24L + DS 3 + LLM 16L + VL 4L "
          f"+ DiT 32L x4 steps):")
    print(f"  legacy-chain: {t_legacy:.3f} ms")
    print(f"  fused:        {t_fused:.3f} ms")
    print(f"  delta:        {(t_legacy - t_fused) * 1e3:+.0f} us "
          f"({(t_legacy - t_fused) / t_legacy * 100:+.2f}%)")


if __name__ == "__main__":
    bench_e2e()
