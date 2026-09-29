"""GROOT N1.7 SM89 elementwise-fusion differential tests (random weights).

Kernel-level: each fused kernel is compared against the exact legacy
launch chain it replaces, on identical inputs. Pipeline-level A/B lives
in the same file (added when the pipeline is wired).

Run:
    python tests/test_groot_n17_sm89_ele_fusion.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flash_rt.flash_rt_kernels as fvk  # noqa: E402

torch.manual_seed(0)
_DEV = "cuda:0"
_FP16 = torch.float16


def _p(t: torch.Tensor) -> int:
    assert t.is_contiguous()
    return t.data_ptr()


def _legacy_qk_norm_rope(q, k, q_w, k_w, cos_t, sin_t, S, NHQ, NHKV, HD, eps):
    """The exact 4-launch chain pipeline_rtx_sm89.py runs today."""
    q = q.clone()
    k = k.clone()  # legacy chain is in-place on its inputs
    fvk.rms_norm_fp16(_p(q), _p(q_w), _p(q), S * NHQ, HD, eps)
    fvk.rms_norm_fp16(_p(k), _p(k_w), _p(k), S * NHKV, HD, eps)
    fvk.rope_rotate_half_fp16(_p(q), _p(cos_t), _p(sin_t), S, NHQ, HD)
    fvk.rope_rotate_half_fp16(_p(k), _p(cos_t), _p(sin_t), S, NHKV, HD)
    return q, k


def _legacy_norm(src, w, S, NH, HD, eps):
    """rms_norm_fp16 output (the legacy chain's norm intermediate)."""
    out = src.clone()
    fvk.rms_norm_fp16(_p(out), _p(w), _p(out), S * NH, HD, eps)
    return out


def _fused_norm(src, w, S, NH, HD, eps):
    """Fused kernel's norm intermediate, extracted via identity rope
    (cos=1, sin=0 leaves the normed value bit-identical)."""
    out = src.clone()
    ones = torch.ones(S, HD, device=_DEV, dtype=_FP16)
    zeros = torch.zeros(S, HD, device=_DEV, dtype=_FP16)
    fvk.qk_norm_rope_rotate_half_fp16(
        _p(out), _p(w), _p(ones), _p(zeros), S, NH, HD, eps)
    return out


def _run_fused(q, k, q_w, k_w, cos_t, sin_t, S, NHQ, NHKV, HD, eps):
    q, k = q.clone(), k.clone()  # kernel is in-place; protect the fixture
    fvk.qk_norm_rope_rotate_half_fp16(_p(q), _p(q_w), _p(cos_t), _p(sin_t),
                                       S, NHQ, HD, eps)
    fvk.qk_norm_rope_rotate_half_fp16(_p(k), _p(k_w), _p(cos_t), _p(sin_t),
                                       S, NHKV, HD, eps)
    return q, k


def test_qk_norm_rope_fused_matches_legacy_chain():
    # Shape sweep: GQA asymmetry, odd S, single token.
    for (S, NHQ, NHKV) in [(1024, 16, 8), (257, 16, 8), (129, 24, 4),
                           (512, 16, 16), (1, 16, 8)]:
        HD = 128
        q = torch.randn(S, NHQ * HD, device=_DEV, dtype=_FP16)
        k = torch.randn(S, NHKV * HD, device=_DEV, dtype=_FP16)
        q_w = torch.randn(HD, device=_DEV, dtype=_FP16)
        k_w = torch.randn(HD, device=_DEV, dtype=_FP16)
        # Untiled cos/sin (halves differ) pins the first-half-only read.
        cos_t = torch.randn(S, HD, device=_DEV, dtype=_FP16)
        sin_t = torch.randn(S, HD, device=_DEV, dtype=_FP16)
        eps = 1e-6

        q_ref, k_ref = _legacy_qk_norm_rope(q, k, q_w, k_w, cos_t, sin_t,
                                            S, NHQ, NHKV, HD, eps)
        q_new, k_new = _run_fused(q, k, q_w, k_w, cos_t, sin_t,
                                  S, NHQ, NHKV, HD, eps)

        for name, ref, new, src in [("Q", q_ref, q_new, q),
                                    ("K", k_ref, k_new, k)]:
            heads = NHQ if name == "Q" else NHKV
            bit_equal = torch.equal(ref, new)
            max_abs = (ref.float() - new.float()).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                ref.float().flatten(), new.float().flatten(), dim=0).item()
            # Gate: bit-equal, or (where the two reductions round differently)
            # every divergent element within 1 fp16 ulp of the fp64-exact
            # value. The fused kernel reduces warp-butterfly vs the legacy
            # block-tree — both are correct roundings of the same fp32 math.
            assert cos > 0.9999999, (name, S, NHQ, NHKV, cos)
            if not bit_equal:
                # The fused kernel reduces the RMSNorm sum warp-butterfly vs
                # the legacy block-tree; every observed full-chain divergence
                # traces to that norm-stage rounding (self element or its
                # rotate-half pair partner). Equivalence gate: each kernel's
                # output is within 1 fp16 ulp of the fp64 rope applied to ITS
                # OWN fp16 norm intermediate (verified: 46/46 diffs pass).
                w_t = q_w if name == "Q" else k_w
                src2 = q if name == "Q" else k
                mid_ref = _legacy_norm(src2, w_t, S, heads, HD, eps)
                mid_new = _fused_norm(src2, w_t, S, heads, HD, eps)
                diff = (ref != new).view(S, heads, HD)
                half = HD // 2
                cf = cos_t.double().view(S, 1, HD)[..., :half]
                sf = sin_t.double().view(S, 1, HD)[..., :half]
                for tag, out, mid in (("legacy", ref, mid_ref),
                                      ("fused", new, mid_new)):
                    m = mid.double().view(S, heads, HD)
                    lo, hi = m[..., :half], m[..., half:]
                    exact = torch.empty_like(m)
                    exact[..., :half] = lo * cf - hi * sf
                    exact[..., half:] = hi * cf + lo * sf
                    ulp = exact.abs() * 2.0 ** -11
                    dev = (out.double().view(S, heads, HD) - exact).abs()
                    bad = diff & (dev > ulp * 1.001)
                    assert not bad.any(), (tag, name, S, NHQ, NHKV,
                                           int(bad.sum()))
            print(f"[qk_norm_rope] S={S} NHQ={NHQ} NHKV={NHKV} {name}: "
                  f"bit_equal={bit_equal} max_abs={max_abs:.3e}")


def test_qk_norm_rope_fused_rejects_bad_hd():
    q = torch.zeros(4, 16 * 100, device=_DEV, dtype=_FP16)
    w = torch.ones(100, device=_DEV, dtype=_FP16)
    t = torch.zeros(4, 100, device=_DEV, dtype=_FP16)
    rc = fvk.qk_norm_rope_rotate_half_fp16(_p(q), _p(w), _p(t), _p(t),
                                           4, 16, 100, 1e-6)
    assert rc == -1, "HD=100 must be rejected by the HD==128 guard"


def test_bias_gelu_inplace_strict_fp16_matches_legacy_chain():
    for (M, N) in [(1024, 4304), (257, 6144), (41, 1536), (1, 128)]:
        x = torch.randn(M, N, device=_DEV, dtype=_FP16)
        bias = torch.randn(N, device=_DEV, dtype=_FP16)
        x_ref = x.clone()
        # legacy chain (bit-parity target)
        fvk.add_bias_fp16(_p(x_ref), _p(bias), M, N)
        fvk.gelu_inplace_fp16(_p(x_ref), M * N)
        # strict fused
        x_strict = x.clone()
        fvk.bias_gelu_inplace_strict_fp16(_p(x_strict), _p(bias), M, N)
        assert torch.equal(x_ref, x_strict), (M, N)

        # non-strict fused keeps x+bias in fp32 through the GELU; the
        # legacy chain rounds it to fp16 first. Bounded by ~2 fp16 ulp of
        # the INPUT magnitude through the gelu'<=1.13 slope (measured
        # max 1.77). The strict variant above is the bit-parity one.
        x_fast = x.clone()
        fvk.bias_gelu_inplace_fp16(_p(x_fast), _p(bias), M, N)
        dev = (x_ref.float() - x_fast.float()).abs()
        ulp_in = ((x.float() + bias.float()).abs() * 2.0 ** -11
                  + 2.0 ** -24)
        assert (dev <= 2.0 * ulp_in).all(), (M, N, dev.max().item())


def test_bias_residual_strict_fp16_matches_legacy_chain():
    for (M, N) in [(1024, 2048), (257, 1152), (41, 1536)]:
        res = torch.randn(M, N, device=_DEV, dtype=_FP16)
        x = torch.randn(M, N, device=_DEV, dtype=_FP16)
        bias = torch.randn(N, device=_DEV, dtype=_FP16)
        res_ref, x_ref = res.clone(), x.clone()
        # legacy chain (bit-parity target)
        fvk.add_bias_fp16(_p(x_ref), _p(bias), M, N)
        fvk.residual_add_fp16(_p(res_ref), _p(x_ref), M * N)
        # strict fused
        res_strict, x_strict = res.clone(), x.clone()
        fvk.bias_residual_strict_fp16(_p(res_strict), _p(x_strict), _p(bias), M, N)
        assert torch.equal(res_ref, res_strict), (M, N)


class _StubAttn:
    """Deterministic stand-in for the attention backend. Both A/B legs run
    the identical stub, so its math is irrelevant to the comparison — it
    only has to be deterministic and shape-correct (S x NHQ*HD == S x D)."""

    def __init__(self, S, D):
        self.O = torch.zeros(S, D, device=_DEV, dtype=_FP16)
        self._state = torch.randn(S, D, device=_DEV, dtype=_FP16)

    def get_slot_ptrs(self, site, li):
        return {"O": self.O.data_ptr()}

    def run(self, site, li, *, q_seq, kv_seq=None, stream=0):
        self.O.copy_(self._state)
        return self.O.data_ptr()


def _llm_fixture(S=1024):
    """Weights/bufs/dims/scales per qwen3vl_llm_forward's docstring."""
    from flash_rt.models.groot_n17 import pipeline_rtx_sm89 as P

    D, NHQ, NHKV, HD, FF, L = 2048, 16, 8, 128, 6144, 16
    torch.manual_seed(7)
    fp8 = torch.float8_e4m3fn

    def rnd(*shape, dtype=_FP16):
        return torch.randn(*shape, device=_DEV, dtype=dtype)

    def mk_wlists():
        return {"in_ln_w": [rnd(D) for _ in range(L)],
                "post_ln_w": [rnd(D) for _ in range(L)],
                "q_norm_w": [rnd(HD) for _ in range(L)],
                "k_norm_w": [rnd(HD) for _ in range(L)],
                "cos": rnd(S, HD), "sin": rnd(S, HD),
                "deepstack_inject": [0] * L,
                "act_scales": [torch.ones(1, device=_DEV,
                                          dtype=torch.float32)
                               for _ in range(L)]}

    def fp8w(N, K):
        # small weights + amax/448 weight scale = the calibrated-semantics
        # scale, keeping the descaled output in a sane fp16 range
        w = (torch.randn(N, K, device=_DEV,
                        dtype=torch.float32) * 0.02).to(fp8)
        ws = torch.tensor([max(0.02 * 3.0 / 448.0, 1e-8)],
                          device=_DEV, dtype=torch.float32)
        _keep.append(w); _keep.append(ws)
        return int(w.data_ptr()), int(ws.data_ptr())

    def mk_weights():
        torch.manual_seed(201)  # same values on every call (A/B legs)
        w = mk_wlists()
        # pointer-ify the plain-tensor lists (pipeline does int(w[li]))
        w["in_ln_w"] = [t.data_ptr() for t in w["in_ln_w"]]
        w["post_ln_w"] = [t.data_ptr() for t in w["post_ln_w"]]
        w["q_norm_w"] = [t.data_ptr() for t in w["q_norm_w"]]
        w["k_norm_w"] = [t.data_ptr() for t in w["k_norm_w"]]
        _keep.append(w["cos"]); _keep.append(w["sin"])
        w["cos"] = w["cos"].data_ptr(); w["sin"] = w["sin"].data_ptr()
        for name, N, K in [("q_w", NHQ * HD, D), ("k_w", NHKV * HD, D),
                           ("v_w", NHKV * HD, D), ("o_w", D, D),
                           ("gate_w", FF, D), ("up_w", FF, D),
                           ("down_w", D, FF)]:
            pairs = [fp8w(N, K) for _ in range(L)]
            w[name] = [p[0] for p in pairs]
            w[name.replace("_w", "_ws")] = [p[1] for p in pairs]
        return w

    _keep = []  # keep tensors alive behind the raw pointers

    def mk_bufs():
        torch.manual_seed(202)
        b = {"h": rnd(S, D), "xn": rnd(S, D),
             "xn_fp8": rnd(S, D).view(torch.uint8).view(fp8),
             "Q": rnd(S, NHQ * HD), "K": rnd(S, NHKV * HD),
             "V": rnd(S, NHKV * HD), "K_exp": rnd(S, NHQ * HD),
             "V_exp": rnd(S, NHQ * HD), "o_proj_out": rnd(S, D),
             "gate_out": rnd(S, FF), "up_out": rnd(S, FF),
             "gu_fp8": rnd(S, FF).view(torch.uint8).view(fp8),
             "bf16_tmp": rnd(S, D, dtype=torch.bfloat16),
             "bf16_ff": rnd(S, FF, dtype=torch.bfloat16)}
        _keep.append(b)
        return {k: v.data_ptr() for k, v in b.items()}

    dims = {"S": S, "D": D, "NHQ": NHQ, "NHKV": NHKV, "HD": HD, "FF": FF}
    # keep-alive list shared with the caller (tensor backing for raw ptrs)
    def adv(t):
        return t.data_ptr()

    def mk_scales():
        # amax/448 per amax_to_dev_scale: activations post-norm have
        # amax ~3-4, so s ~ 0.008
        for _ in range(4):  # 4 scale groups
            _keep.append(None)
        s = torch.full((1,), max(4.0 / 448.0, 1e-8), device=_DEV,
                       dtype=torch.float32)
        _keep[-4:] = [s] * 4
        return {k: [int(s.data_ptr()) for _ in range(L)]
                for k in ("act_qkv", "act_o", "act_gateup", "act_down")}

    scales = mk_scales()
    gemm = fvk.GemmRunner()
    return P, gemm, dims, mk_weights, mk_bufs, scales, (S, D), _keep


def _legacy_fvk_shim():
    """Legacy-chain shims for the fused entry points the pipeline uses."""
    import types
    shim = types.SimpleNamespace()
    for name in dir(fvk):
        if not name.startswith("_"):
            setattr(shim, name, getattr(fvk, name))

    def qk_norm_rope_rotate_half_fp16(x, w, cos_t, sin_t, S, NH, HD, eps,
                                      stream=0):
        fvk.rms_norm_fp16(x, w, x, S * NH, HD, eps)
        fvk.rope_rotate_half_fp16(x, cos_t, sin_t, S, NH, HD)

    def bias_gelu_inplace_strict_fp16(x, bias, M, N, stream=0):
        fvk.add_bias_fp16(x, bias, M, N)
        fvk.gelu_inplace_fp16(x, M * N)

    def bias_residual_strict_fp16(res, x, bias, M, N, stream=0):
        fvk.add_bias_fp16(x, bias, M, N)
        fvk.residual_add_fp16(res, x, M * N)

    shim.qk_norm_rope_rotate_half_fp16 = qk_norm_rope_rotate_half_fp16
    shim.bias_gelu_inplace_strict_fp16 = bias_gelu_inplace_strict_fp16
    shim.bias_residual_strict_fp16 = bias_residual_strict_fp16
    return shim


def test_qwen3vl_llm_forward_fused_matches_legacy():
    """Full LLM stage A/B: fused pipeline vs legacy-chain shim, random
    weights, identical seeds. The qk-norm fusion's reduce re-association
    can shift a few elements by 1 fp16 ulp per layer, so gate at
    cos >= 0.9999 + reported max_abs; the bias/residual and bias/gelu
    sites are strict (bit-identical chains)."""
    P, gemm, dims, mk_weights, mk_bufs, scales, (S, D), keep = _llm_fixture()
    attn = _StubAttn(S, D)

    # Leg A: fused pipeline (post-wiring code path)
    w1, b1 = mk_weights(), mk_bufs()
    h_seed = keep[-1]["h"].clone()
    P.qwen3vl_llm_forward(gemm, fvk, b1, w1, dims, scales, attn=attn,
                          stream=0)
    h_a = keep[-1]["h"].clone()

    # Leg B: legacy chains via the shim, identical weights/bufs
    w2, b2 = mk_weights(), mk_bufs()
    # overwrite leg B's h with the same seed values at the same address
    torch.Tensor._make_wrapper_subclass  # noqa: B018 (keep import surface)
    keep[-1]["h"].copy_(h_seed)
    attn2 = _StubAttn(S, D)
    attn2._state = attn._state.clone()
    P.qwen3vl_llm_forward(gemm, _legacy_fvk_shim(), b2, w2, dims, scales,
                          attn=attn2, stream=0)
    h_b = keep[-1]["h"].clone()

    a = h_a.float().flatten()
    b = h_b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    max_abs = (a - b).abs().max().item()
    assert cos >= 0.9999, cos
    print(f"[llm A/B] S={S} cos={cos:.7f} max_abs={max_abs:.3e}")


_BF16 = torch.bfloat16


def test_qkv_split_bias_bf16_matches_legacy_chain():
    # GROOT DiT self-attn: Sa tokens, D=1536, NH=32 heads x HD=48; the
    # kernel is generic — sweep asymmetric head splits too.
    for (S, Hq, Hk, Hv, HD) in [(41, 1536, 1536, 1536, 1),
                                (41, 32 * 48, 32 * 48, 32 * 48, 1),
                                (1024, 2048, 1024, 1024, 1)]:
        W = Hq + Hk + Hv
        packed = torch.randn(S, W, device=_DEV, dtype=_BF16)
        bias = torch.randn(W, device=_DEV, dtype=_BF16)
        q = torch.empty(S, Hq, device=_DEV, dtype=_BF16)
        k = torch.empty(S, Hk, device=_DEV, dtype=_BF16)
        v = torch.empty(S, Hv, device=_DEV, dtype=_BF16)
        # legacy chain: add_bias in place on packed, then three strided
        # copies
        ref = packed.clone()
        fvk.add_bias_bf16(_p(ref), _p(bias), S, W)
        q_ref = ref[:, :Hq].contiguous()
        k_ref = ref[:, Hq:Hq + Hk].contiguous()
        v_ref = ref[:, Hq + Hk:].contiguous()
        # fused
        fvk.qkv_split_bias_bf16(_p(packed), _p(bias), _p(q), _p(k), _p(v),
                                S, Hq, Hk, Hv)
        assert torch.equal(q, q_ref) and torch.equal(k, k_ref) \
            and torch.equal(v, v_ref), (S, Hq, Hk, Hv)


def test_residual_add_bias_bf16_matches_legacy_chain():
    for (M, N) in [(41, 1536), (1024, 2048)]:
        res = torch.randn(M, N, device=_DEV, dtype=_BF16)
        x = torch.randn(M, N, device=_DEV, dtype=_BF16)
        bias = torch.randn(N, device=_DEV, dtype=_BF16)
        res_ref, x_ref = res.clone(), x.clone()
        # legacy: add bias to x then residual add
        fvk.add_bias_bf16(_p(x_ref), _p(bias), M, N)
        fvk.residual_add(_p(res_ref), _p(x_ref), M * N)
        res_new = res.clone()
        fvk.residual_add_bias_bf16(_p(res_new), _p(x), _p(bias), M, N)
        # legacy rounds x+bias to bf16 before the residual add; the fused
        # kernel keeps the sum in fp32. Gate at 2 bf16 ulp of the
        # INPUT magnitudes (cancellation can zero the output, making
        # output-relative ulp meaningless; measured max 1 ulp).
        dev = (res_ref.float() - res_new.float()).abs()
        # double rounding (fp16 sum -> bf16, then +res -> bf16) can shift
        # the result by 1 ulp at the OUTPUT magnitude even when inputs
        # are smaller — bound on the largest of the three
        mag = torch.max(torch.maximum((x.float() + bias.float()).abs(),
                                     res.float().abs()),
                        res_ref.float().abs())
        ulp = mag * 2.0 ** -7 + 2.0 ** -126  # bf16: 7 mantissa bits
        assert (dev <= 1.001 * ulp).all(), (M, N, dev.max().item())


def test_dit_kernel_fusions_match_legacy_chains():
    """DiT bf16 fallback sites: strict twins must be bit-identical."""
    for (M, N) in [(41, 1536), (257, 6144), (1024, 2048)]:
        res = torch.randn(M, N, device=_DEV, dtype=_BF16)
        x = torch.randn(M, N, device=_DEV, dtype=_BF16)
        bias = torch.randn(N, device=_DEV, dtype=_BF16)
        res_ref, x_ref = res.clone(), x.clone()
        fvk.add_bias_bf16(_p(x_ref), _p(bias), M, N)
        fvk.residual_add(_p(res_ref), _p(x_ref), M * N)
        res_strict, x_strict = res.clone(), x.clone()
        fvk.bias_residual_strict_bf16(_p(res_strict), _p(x_strict), _p(bias),
                                      M, N)
        assert torch.equal(res_ref, res_strict), ("bias_res", M, N)

        xg = torch.randn(M, N, device=_DEV, dtype=_BF16)
        bg = torch.randn(N, device=_DEV, dtype=_BF16)
        xg_ref = xg.clone()
        fvk.add_bias_bf16(_p(xg_ref), _p(bg), M, N)
        fvk.gelu_inplace(_p(xg_ref), M * N)
        xg_strict = xg.clone()
        fvk.bias_gelu_bf16_strict(_p(xg_strict), _p(bg), M, N)
        assert torch.equal(xg_ref, xg_strict), ("bias_gelu", M, N)


def _dit_fixture(Sa=41):
    """Weights/bufs/dims per dit_forward's bf16 fallback branch (the one
    the RTX sm89 frontend takes — no fp8/fp4 weight keys)."""
    from flash_rt.models.groot_n17 import pipeline_thor as P
    D, FF, L = 1536, 6144, 32
    torch.manual_seed(11)
    keep = []

    def rnd(*shape):
        return torch.randn(*shape, device=_DEV, dtype=_BF16)

    def mk_weights():
        torch.manual_seed(101)  # same values on every call (A/B legs)
        w = {"scale_msa": [rnd(D) for _ in range(L)],
             "shift_msa": [rnd(D) for _ in range(L)],
             "q_w": [rnd(D, D) for _ in range(L)],
             "q_b": [rnd(D) for _ in range(L)],
             "k_w": [rnd(D, D) for _ in range(L)],
             "k_b": [rnd(D) for _ in range(L)],
             "v_w": [rnd(D, D) for _ in range(L)],
             "v_b": [rnd(D) for _ in range(L)],
             "o_w": [rnd(D, D) for _ in range(L)],
             "o_b": [rnd(D) for _ in range(L)],
             "ff_proj_w": [rnd(FF, D) for _ in range(L)],
             "ff_proj_b": [rnd(FF) for _ in range(L)],
             "ff_down_w": [rnd(D, FF) for _ in range(L)],
             "ff_down_b": [rnd(D) for _ in range(L)]}
        keep.append(w)
        return {k: ([t.data_ptr() for t in v] if isinstance(v, list) else v)
                for k, v in w.items()}

    def mk_bufs():
        torch.manual_seed(102)
        b = {"h": rnd(Sa, D), "xn": rnd(Sa, D), "o_proj_out": rnd(Sa, D),
             "ff_proj_out": rnd(Sa, FF)}
        keep.append(b)
        return {k: t.data_ptr() for k, t in b.items()}

    dims = {"Sa": Sa, "D": D, "FF": FF, "Skv_text": 64, "Skv_image": 64}
    gemm = fvk.GemmRunner()
    return P, gemm, dims, mk_weights, mk_bufs, keep, (Sa, D)


class _StubDiTAttn:
    """Deterministic DiT attention backend stand-in (32 self + 16 cross
    sites; O filled deterministically)."""

    def __init__(self, Sa, D, n_self=16, n_cross=16):
        self.O = torch.zeros(Sa, D, device=_DEV, dtype=_BF16)
        self._state = torch.randn(Sa, D, device=_DEV, dtype=_BF16)

    def get_slot_ptrs(self, site, j):
        return {"Q": self.O.data_ptr(), "K": self.O.data_ptr(),
                "V": self.O.data_ptr(), "O": self.O.data_ptr()}

    def run(self, site, j, *, q_seq, kv_seq=None, stream=0):
        self.O.copy_(self._state)
        return self.O.data_ptr()


def test_dit_forward_fused_matches_legacy():
    """Full 32-layer DiT A/B on the bf16 fallback branch: fused pipeline
    vs legacy-chain shim. All wired sites are strict (bit-identical), so
    the two legs must be torch.equal end to end."""
    P, gemm, dims, mk_weights, mk_bufs, keep, (Sa, D) = _dit_fixture()

    attn = _StubDiTAttn(Sa, D)
    w1, b1 = mk_weights(), mk_bufs()
    h_seed = keep[-1]["h"].clone()
    P.dit_forward(gemm, fvk, b1, w1, dims, attn=attn, stream=0)
    h_a = keep[-1]["h"].clone()

    shim = _legacy_fvk_shim()

    def legacy_bias_res(res, x, bias, M, N, stream=0):
        fvk.add_bias_bf16(x, bias, M, N)
        fvk.residual_add(res, x, M * N)

    def legacy_bias_gelu(x, bias, M, N, stream=0):
        fvk.add_bias_bf16(x, bias, M, N)
        fvk.gelu_inplace(x, M * N)
    shim.bias_residual_strict_bf16 = legacy_bias_res
    shim.bias_gelu_bf16_strict = legacy_bias_gelu

    attn2 = _StubDiTAttn(Sa, D)
    attn2._state = attn._state.clone()
    w2, b2 = mk_weights(), mk_bufs()
    keep[-1]["h"].copy_(h_seed)
    P.dit_forward(gemm, shim, b2, w2, dims, attn=attn2, stream=0)
    h_b = keep[-1]["h"].clone()

    assert torch.equal(h_a, h_b), \
        ((h_a.float() - h_b.float()).abs().max().item())
    print(f"[dit A/B] Sa={Sa}: bit-equal across 32 layers")


if __name__ == "__main__":
    test_qk_norm_rope_fused_matches_legacy_chain()
    test_qk_norm_rope_fused_rejects_bad_hd()
    test_bias_gelu_inplace_strict_fp16_matches_legacy_chain()
    test_bias_residual_strict_fp16_matches_legacy_chain()
    test_qwen3vl_llm_forward_fused_matches_legacy()
    test_qkv_split_bias_bf16_matches_legacy_chain()
    test_residual_add_bias_bf16_matches_legacy_chain()
    test_dit_kernel_fusions_match_legacy_chains()
    test_dit_forward_fused_matches_legacy()
    print("ALL PASS")
