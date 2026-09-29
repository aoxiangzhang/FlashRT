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


if __name__ == "__main__":
    test_qk_norm_rope_fused_matches_legacy_chain()
    test_qk_norm_rope_fused_rejects_bad_hd()
    test_bias_gelu_inplace_strict_fp16_matches_legacy_chain()
    test_bias_residual_strict_fp16_matches_legacy_chain()
    print("ALL PASS")
