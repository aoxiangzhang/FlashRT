"""Latency benchmark for the GROOT N1.7 SM89 elementwise fusions.

Two legs per measurement:
  1. kernel-level: qk_norm_rope_rotate_half_fp16 (Q+K) vs the legacy
     rms_norm_fp16 x2 + rope_rotate_half_fp16 x2 chain.
  2. stage-level: full qwen3vl_llm_forward, fused pipeline vs the
     legacy-chain shim (shared fixture with the differential test).

Run:
    python benchmarks/groot_n17_sm89_ele_fusion_bench.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flash_rt.flash_rt_kernels as fvk  # noqa: E402
from tests.test_groot_n17_sm89_ele_fusion import (  # noqa: E402
    _llm_fixture, _legacy_fvk_shim, _p, _StubAttn, _DEV, _FP16,
)

torch.manual_seed(0)


def _time(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]  # median ms


def bench_qk_norm_rope():
    print("== kernel-level: qk-norm + RoPE (Q+K pair) ==")
    for S in (257, 1024, 4096):
        NHQ, NHKV, HD, eps = 16, 8, 128, 1e-6
        q = torch.randn(S, NHQ * HD, device=_DEV, dtype=_FP16)
        k = torch.randn(S, NHKV * HD, device=_DEV, dtype=_FP16)
        q_w = torch.randn(HD, device=_DEV, dtype=_FP16)
        k_w = torch.randn(HD, device=_DEV, dtype=_FP16)
        cos_t = torch.randn(S, HD, device=_DEV, dtype=_FP16)
        sin_t = torch.randn(S, HD, device=_DEV, dtype=_FP16)

        def legacy():
            qq, kk = q.clone(), k.clone()
            fvk.rms_norm_fp16(_p(qq), _p(q_w), _p(qq), S * NHQ, HD, eps)
            fvk.rms_norm_fp16(_p(kk), _p(k_w), _p(kk), S * NHKV, HD, eps)
            fvk.rope_rotate_half_fp16(_p(qq), _p(cos_t), _p(sin_t), S, NHQ, HD)
            fvk.rope_rotate_half_fp16(_p(kk), _p(cos_t), _p(sin_t), S, NHKV, HD)

        def fused():
            qq, kk = q.clone(), k.clone()
            fvk.qk_norm_rope_rotate_half_fp16(
                _p(qq), _p(q_w), _p(cos_t), _p(sin_t), S, NHQ, HD, eps)
            fvk.qk_norm_rope_rotate_half_fp16(
                _p(kk), _p(k_w), _p(cos_t), _p(sin_t), S, NHKV, HD, eps)

        t_l, t_f = _time(legacy), _time(fused)
        print(f"S={S:5d}: legacy {t_l*1e3:8.2f} us  fused {t_f*1e3:8.2f} us"
              f"  speedup {t_l / t_f:.2f}x")


def bench_llm_stage():
    print("\n== stage-level: qwen3vl_llm_forward (16 layers, S=1024) ==")
    P, gemm, dims, mk_weights, mk_bufs, scales, (S, D), keep = _llm_fixture()
    attn = _StubAttn(S, D)
    # one weight/buf set, reused across iterations (h is re-randomized
    # per leg so the chain runs on fresh data; keep-alive holds tensors)
    w, b = mk_weights(), mk_bufs()
    keep_idx = len(keep) - 1

    def run_fused():
        keep[keep_idx]["h"].normal_()
        P.qwen3vl_llm_forward(gemm, fvk, b, w, dims, scales, attn=attn,
                              stream=0)

    shim = _legacy_fvk_shim()
    attn2 = _StubAttn(S, D)

    def run_legacy():
        keep[keep_idx]["h"].normal_()
        P.qwen3vl_llm_forward(gemm, shim, b, w, dims, scales, attn=attn2,
                              stream=0)

    # NOTE: stage time is dominated by the fp8 GEMMs; the elementwise delta
    # is the measurable difference between the two legs.
    t_l, t_f = _time(run_legacy), _time(run_fused)
    print(f"legacy-chain leg: {t_l:.3f} ms   fused leg: {t_f:.3f} ms"
          f"   delta {t_l - t_f:+.3f} ms ({(t_l - t_f) / t_l * 100:+.1f}%)")


if __name__ == "__main__":
    bench_qk_norm_rope()
    bench_llm_stage()
