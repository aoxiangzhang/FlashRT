# GROOT N1.7 RTX SM89 Elementwise Fusion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse the unfused elementwise chains of the GROOT N1.7 SM89 backbone pipeline (`flash_rt/models/groot_n17/pipeline_rtx_sm89.py`) — QKV qk-norm+RoPE fusion (new kernel), bias+residual and bias+GELU wiring (existing/template kernels) — verified bit-parity-or-ulp on a local RTX 4090.

**Architecture:** Three fusion families on the SM89 fp16 backbone path: F1 = one new kernel `qk_rmsnorm_rope_fused_fp16` replacing the 4-launch per-head q/k RMSNorm + rotate-half M-RoPE chain in the LLM stage; F2 = wiring `bias_residual_strict_fp16` (exists, strict = bit-identical to the current add_bias→residual_add two-round chain) at 4 sites; F3 = instantiating the templated `bias_gelu_strict_kernel<__half>` (bf16 twin exists) + binding, wired at 3 FFN sites. All changes are direct replacements (matching the upstream Thor fusion commit style, e.g. 9ad764e6), no opt-in flag; correctness is pinned by a new differential test that rebuilds the old kernel chain as the reference.

**Tech Stack:** CUDA C++ (pybind11 `flash_rt_kernels` module), PyTorch test harness, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-groot-n17-sm89-ele-fusion-design.md`

## Global Constraints

- Repo: `/home/zhangaoxiang/code/flashrt_github/FlashRT`, branch `feature/groot-n17-sm89-ele-fusion` (from origin/main 839b1597). All paths below are relative to repo root.
- Python: `/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python` (torch 2.13.0+cu130). GPU: RTX 4090 D (sm89).
- Build: `cmake -B build -S . -DGPU_ARCH=89 -DFA2_ARCH_NATIVE_ONLY=ON` then `cmake --build build --target flash_rt_kernels -j6` (multi-user box — max -j6).
- Bit-parity contract (upstream naming rule): the `strict` suffix means round-after-every-original-kernel. Sites replacing a two-launch chain MUST use the strict variant (`bias_residual_strict_fp16`, new `bias_gelu_inplace_strict_fp16`); their output must be `torch.equal` to the old chain. The F1 fused kernel re-associates the RMSNorm reduction (warp shuffle vs block tree) — its gate is max_abs ≤ 2 fp16 ulp (2^-10 relative at |x|≈1) AND cos = 1.0, plus pipeline-level cos ≥ 0.9999; report the measured number (it may be 0).
- The m.def binding must live in the main `flash_rt_kernels` module (`csrc/bindings.cpp`), same guard discipline as the TU it calls (all target TUs are unconditionally compiled — no `#ifdef` needed).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `git add -A`; the spec/plan files under `docs/superpowers/` stay out of the PR.
- Never stage `nsys_reports/`, `build/`, or anything under `docs/superpowers/`.

## Review Focus

1. **GQA head-count asymmetry** (Q 16 heads, K 8 heads): `seq_pos = row / heads` must use the *own-segment* head count, and the row→segment split at `rows_q` must not shift K rows. Pinned by Task 2's test shapes (NHQ=16/NHKV=8 and 48/16).
2. **Tail rows** (S·NH not divisible by ROWS_PER_BLOCK=8): the last block's out-of-range warps must return early without touching memory. Pinned by Task 2's odd-S test case.
3. **In-place aliasing** (q_out == q): the fused kernel must read both rotate-half halves before writing. Pinned by Task 2's in-place A/B and by the pipeline differential (the pipeline uses in-place).
4. **Strict two-round semantics**: `bias_residual_strict_fp16` and `bias_gelu_inplace_strict_fp16` must round x+bias to fp16 *before* the second op, matching the legacy chain. Pinned by `torch.equal` asserts in Task 3/Task 4 tests — and the non-strict variants must be shown to differ (guard against silently swapping semantics).
5. **cos/sin table layout drift**: the fused kernel reads `cos_t[s*HD + d]` for `d < HD/2` (the `rope_rotate_half_fp16` convention). A test with a table whose halves differ (not tiled) proves the kernel reads the first half only — catching any accidental full-width read.

---

### Task 1: Build baseline and green test suite

**Files:**
- Create: `build/` (build dir, gitignored)
- Test: existing `tests/test_groot_n17_*.py` (no edits)

**Interfaces:**
- Produces: a working `flash_rt.flash_rt_kernels` import from THIS repo (not the internal repo!) — all later tasks import `flash_rt.flash_rt_kernels as fvk` with repo root on `sys.path`.

- [ ] **Step 1: Configure and build the kernel module**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
cmake -B build -S . -DGPU_ARCH=89 -DFA2_ARCH_NATIVE_ONLY=ON
cmake --build build --target flash_rt_kernels -j6
```

Expected: `flash_rt/flash_rt_kernels.cpython-*.so` appears in the repo tree (editable-install contract). If cmake fails on a missing dependency, read `docs/INSTALL.md` for the supported toolchain and fix the environment, not the build files.

- [ ] **Step 2: Verify the module imports and the two kernels we will mirror exist**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python -c "
import flash_rt.flash_rt_kernels as fvk
for name in ('rms_norm_fp16','rope_rotate_half_fp16','add_bias_fp16',
             'residual_add_fp16','bias_residual_strict_fp16','gelu_inplace_fp16'):
    assert hasattr(fvk, name), name
print('baseline OK')"
```

Expected: `baseline OK`.

- [ ] **Step 3: Run the existing GROOT N1.7 test suite for a green baseline**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python -m pytest tests/ -k groot_n17 -x -q --no-header 2>&1 | tail -5
```

Expected: all pass or skip (fixture-gated tests skip without checkpoints — skips are fine, failures are not). Record the pass/skip counts in the task notes.

- [ ] **Step 4: No commit** (build artifacts only).

---

### Task 2: New kernel `qk_rmsnorm_rope_fused_fp16`

**Files:**
- Modify: `csrc/kernels/qk_norm_rope_fused.cu` (append RMSNorm variant after the existing Chameleon LayerNorm variant)
- Modify: `csrc/kernels/qk_norm_rope_fused.cuh` (declaration)
- Modify: `csrc/bindings.cpp` (one m.def, placed next to the existing `flash_rt_qk_norm_rope_fused_fp16` binding — find with `grep -n qk_norm_rope_fused csrc/bindings.cpp`)
- Test: `tests/test_groot_n17_sm89_ele_fusion.py` (new file, this task adds the kernel-level part)

**Interfaces:**
- Consumes: none (leaf kernel).
- Produces: Python entry `flash_rt_kernels.qk_rmsnorm_rope_fused_fp16(q, k, q_w, k_w, cos_t, sin_t, q_out, k_out, seq_len, heads_q, heads_k, dim, eps, stream=0)` — all pointer args as ints (uintptr_t), matching the repo's pointer-style binding convention. Q/K layouts `[S, heads*HD]` fp16 contiguous, weights `[HD]` fp16, cos/sin `[S, HD]` fp16 (first HD/2 entries used per row, `rope_rotate_half_fp16` convention). In-place when `{q_out,k_out} == {q,k}`. `dim` must be a power of two ≤ 256.

- [ ] **Step 1: Write the failing test (kernel-level differential)**

Create `tests/test_groot_n17_sm89_ele_fusion.py`:

```python
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
    q = q.clone(); k = k.clone()  # legacy chain is in-place on its inputs
    fvk.rms_norm_fp16(_p(q), _p(q_w), _p(q), S * NHQ, HD, eps)
    fvk.rms_norm_fp16(_p(k), _p(k_w), _p(k), S * NHKV, HD, eps)
    fvk.rope_rotate_half_fp16(_p(q), _p(cos_t), _p(sin_t), S, NHQ, HD)
    fvk.rope_rotate_half_fp16(_p(k), _p(cos_t), _p(sin_t), S, NHKV, HD)
    return q, k

def _run_fused(q, k, q_w, k_w, cos_t, sin_t, S, NHQ, NHKV, HD, eps):
    fvk.qk_rmsnorm_rope_fused_fp16(
        _p(q), _p(k), _p(q_w), _p(k_w), _p(cos_t), _p(sin_t),
        _p(q), _p(k), S, NHQ, NHKV, HD, eps)
    return q, k

def test_qk_rmsnorm_rope_fused_matches_legacy_chain():
    # Shape sweep: GQA asymmetry, odd S (tail warps), HD variants.
    for (S, NHQ, NHKV, HD) in [(1024, 16, 8, 128), (257, 16, 8, 128),
                               (129, 24, 4, 128), (512, 16, 16, 256),
                               (1, 16, 8, 128)]:
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

        for name, ref, new in [("Q", q_ref, q_new), ("K", k_ref, k_new)]:
            bit_equal = torch.equal(ref, new)
            max_abs = (ref.float() - new.float()).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                ref.float().flatten(), new.float().flatten(), dim=0).item()
            # Gate: bit-equal, or ulp-level with cos 1.0. Report which.
            assert cos == 1.0, (name, S, NHQ, NHKV, HD, cos)
            assert max_abs <= 2.0 * 2.0 ** -10, (name, S, NHQ, NHKV, HD, max_abs)
            print(f"[qk_rmsnorm_rope] S={S} NHQ={NHQ} NHKV={NHKV} HD={HD} "
                  f"{name}: bit_equal={bit_equal} max_abs={max_abs:.3e}")

def test_qk_rmsnorm_rope_fused_rejects_bad_dim():
    q = torch.zeros(4, 16 * 128, device=_DEV, dtype=_FP16)
    k = torch.zeros(4, 8 * 128, device=_DEV, dtype=_FP16)
    w = torch.ones(128, device=_DEV, dtype=_FP16)
    t = torch.zeros(4, 128, device=_DEV, dtype=_FP16)
    try:
        fvk.qk_rmsnorm_rope_fused_fp16(
            _p(q), _p(k), _p(w), _p(w), _p(t), _p(t), _p(q), _p(k),
            4, 16, 8, 100, 1e-6)  # dim not a power of two
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("dim=100 must be rejected")

if __name__ == "__main__":
    test_qk_rmsnorm_rope_fused_matches_legacy_chain()
    test_qk_rmsnorm_rope_fused_rejects_bad_dim()
    print("ALL PASS")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python tests/test_groot_n17_sm89_ele_fusion.py
```

Expected: FAIL with `AttributeError: module 'flash_rt.flash_rt_kernels' has no attribute 'qk_rmsnorm_rope_fused_fp16'` (the build is from Task 1; the binding does not exist yet).

- [ ] **Step 3: Implement the kernel**

Append to `csrc/kernels/qk_norm_rope_fused.cu` (inside namespace `flash_rt::kernels`, after the existing LayerNorm variant):

```cuda
// ================================================================
// Fused per-head QK RMSNorm (no bias) + Rotate-Half RoPE (FP16) —
// RMSNorm twin of qk_norm_rope_fused_fp16 above.
//
// Replaces the per-GROOT-N1.7-LLM-layer chain (4 launches -> 1):
//   rms_norm_fp16(Q, q_norm_w, Q, S*NHQ,  HD, eps)   // in-place
//   rms_norm_fp16(K, k_norm_w, K, S*NHKV, HD, eps)
//   rope_rotate_half_fp16(Q, cos, sin, S, NHQ,  HD)
//   rope_rotate_half_fp16(K, cos, sin, S, NHKV, HD)
//
// Layouts (same conventions as the kernels it replaces):
//   Q, K         : [S, heads*HD] fp16 contiguous, head-interleaved
//   q_w, k_w     : [HD] fp16, per-head RMSNorm weight (Q and K
//                  independent — GROOT N1.7 q_norm_w/k_norm_w)
//   cos_t, sin_t : [S, HD] fp16; only the first HD/2 entries of each
//                  row are read (rope_rotate_half_fp16 convention)
//   In-place when {q_out, k_out} == {q, k}.
//
// Math per (token, head) row (fp32 accumulate):
//   rms = rsqrtf(sum(x^2) / HD + eps)
//   n[d] = fp16(x * rms * w)          // fp16 round-trip matches the
//                                     // two-launch chain's intermediate
//   out[d]      = fp16(n_lo * cos - n_hi * sin)   // rotate_half pairs
//   out[d+HD/2] = fp16(n_hi * cos + n_lo * sin)
//
// Kernel layout: 32 lanes x ROWS_PER_BLOCK warps; each warp owns one
// (token, head) row independently — zero smem, zero __syncthreads.
// dim must be a power of two <= 256.
// ================================================================

constexpr int QK_RMS_ROPE_MAX_DIM = 256;

template<int ROWS_PER_BLOCK>
__global__ void qk_rmsnorm_rope_fused_fp16_kernel(
        const __half* __restrict__ q,  const __half* __restrict__ k,
        const __half* __restrict__ q_w, const __half* __restrict__ k_w,
        const __half* __restrict__ cos_t, const __half* __restrict__ sin_t,
        __half* __restrict__ q_out, __half* __restrict__ k_out,
        int rows_q,      // = S * heads_q
        int heads_q,
        int heads_k,
        int dim,
        float eps) {
    const int lane = threadIdx.x;
    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= rows_q + (rows_q / heads_q) * heads_k) return;  // tail guard
    // (gridDim covers S*(heads_q + heads_k) rows exactly; guard for safety)
    const bool is_k = row >= rows_q;
    const int r = is_k ? (row - rows_q) : row;
    const int heads = is_k ? heads_k : heads_q;
    const __half* x = is_k ? k : q;
    const __half* w = is_k ? k_w : q_w;
    __half* out = is_k ? k_out : q_out;
    const int seq_pos = r / heads;
    const int base = r * dim;
    const int half = dim >> 1;

    // 1) fp32 sum of squares over the row, warp butterfly reduce.
    float sq = 0.0f;
    for (int d = lane; d < dim; d += 32) {
        const float v = __half2float(x[base + d]);
        sq += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sq += __shfl_xor_sync(0xffffffffu, sq, off);
    const float rms = rsqrtf(sq / static_cast<float>(dim) + eps);

    // 2) normed = fp16(x * rms * w), cached in registers per lane.
    //    Each lane owns pairs (d, d+half), d = lane + i*32.
    constexpr int MAX_PAIRS_PER_LANE = QK_RMS_ROPE_MAX_DIM / 64;  // = 4
    const int pairs_per_lane = half >> 5;              // dim<=256 => 1..4
    __half n_lo[MAX_PAIRS_PER_LANE], n_hi[MAX_PAIRS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < MAX_PAIRS_PER_LANE; ++i) {
        if (i >= pairs_per_lane) break;
        const int d = lane + (i << 5);
        n_lo[i] = __float2half(
            __half2float(x[base + d]) * rms * __half2float(w[d]));
        n_hi[i] = __float2half(
            __half2float(x[base + d + half]) * rms * __half2float(w[d + half]));
    }

    // 3) rotate_half with the first-half cos/sin entries; both halves of
    //    the pair are already in registers, so in-place output is safe.
    const __half* crow = cos_t + seq_pos * dim;
    const __half* srow = sin_t + seq_pos * dim;
    #pragma unroll
    for (int i = 0; i < MAX_PAIRS_PER_LANE; ++i) {
        if (i >= pairs_per_lane) break;
        const int d = lane + (i << 5);
        const float c = __half2float(crow[d]);
        const float si = __half2float(srow[d]);
        const float lo = __half2float(n_lo[i]);
        const float hi = __half2float(n_hi[i]);
        out[base + d] = __float2half(lo * c - hi * si);
        out[base + d + half] = __float2half(hi * c + lo * si);
    }
}

void qk_rmsnorm_rope_fused_fp16(
        const __half* q, const __half* k,
        const __half* q_w, const __half* k_w,
        const __half* cos_t, const __half* sin_t,
        __half* q_out, __half* k_out,
        int seq_len, int heads_q, int heads_k, int dim, float eps,
        cudaStream_t stream) {
    if (dim <= 0 || dim > QK_RMS_ROPE_MAX_DIM || (dim & (dim - 1)) != 0)
        throw std::invalid_argument(
            "qk_rmsnorm_rope_fused_fp16 requires dim to be a power of two <= 256");
    constexpr int ROWS_PER_BLOCK = 8;
    const int rows_q = seq_len * heads_q;
    const int blocks = (rows_q + seq_len * heads_k + ROWS_PER_BLOCK - 1)
                       / ROWS_PER_BLOCK;
    const dim3 block(32, ROWS_PER_BLOCK);
    qk_rmsnorm_rope_fused_fp16_kernel<ROWS_PER_BLOCK>
        <<<blocks, block, 0, stream>>>(
            q, k, q_w, k_w, cos_t, sin_t, q_out, k_out,
            rows_q, heads_q, heads_k, dim, eps);
}
```

Notes for the implementer:
- The file already includes `<stdexcept>`-equivalent error paths? Check the top of the file; if `std::invalid_argument` is not available, add `#include <stdexcept>`.
- The tail-guard expression `rows_q + (rows_q / heads_q) * heads_k` recomputes `S * heads_k`; alternatively pass `total_rows`. Keep it simple: guard is `row >= rows_q + rows_k` with `rows_k` passed as a parameter — **change the kernel signature to take `rows_q, rows_k, heads_q, heads_k`** and compute `blocks` from both (this is the cleaner form; the signature in the Interfaces block above reflects the *binding*, not the kernel template).

Append to `csrc/kernels/qk_norm_rope_fused.cuh`:

```cuda
// RMSNorm (no-bias) + rotate-half RoPE fused twin (see .cu header comment).
void qk_rmsnorm_rope_fused_fp16(
        const __half* q, const __half* k,
        const __half* q_w, const __half* k_w,
        const __half* cos_t, const __half* sin_t,
        __half* q_out, __half* k_out,
        int seq_len, int heads_q, int heads_k, int dim, float eps,
        cudaStream_t stream);
```

Add the public C entry (mirror the existing `flash_rt_qk_norm_rope_fused_fp16` at the bottom of the .cu):

```cuda
extern "C" void flash_rt_qk_rmsnorm_rope_fused_fp16(
        const __half* q, const __half* k,
        const __half* q_w, const __half* k_w,
        const __half* cos_t, const __half* sin_t,
        __half* q_out, __half* k_out,
        int seq_len, int heads_q, int heads_k, int dim, float eps,
        cudaStream_t stream) {
    flash_rt::kernels::qk_rmsnorm_rope_fused_fp16(
        q, k, q_w, k_w, cos_t, sin_t, q_out, k_out,
        seq_len, heads_q, heads_k, dim, eps, stream);
}
```

Add the binding in `csrc/bindings.cpp` next to the existing one (find with `grep -n "qk_norm_rope_fused_fp16" csrc/bindings.cpp`; copy its include + m.def pattern):

```cpp
m.def("qk_rmsnorm_rope_fused_fp16",
      [](uintptr_t q, uintptr_t k, uintptr_t q_w, uintptr_t k_w,
         uintptr_t cos_t, uintptr_t sin_t, uintptr_t q_out, uintptr_t k_out,
         int seq_len, int heads_q, int heads_k, int dim, float eps,
         uintptr_t stream) {
          flash_rt_qk_rmsnorm_rope_fused_fp16(
              reinterpret_cast<const __half*>(q),
              reinterpret_cast<const __half*>(k),
              reinterpret_cast<const __half*>(q_w),
              reinterpret_cast<const __half*>(k_w),
              reinterpret_cast<const __half*>(cos_t),
              reinterpret_cast<const __half*>(sin_t),
              reinterpret_cast<__half*>(q_out),
              reinterpret_cast<__half*>(k_out),
              seq_len, heads_q, heads_k, dim, eps, to_stream(stream));
      }, py::arg("q"), py::arg("k"), py::arg("q_w"), py::arg("k_w"),
      py::arg("cos_t"), py::arg("sin_t"), py::arg("q_out"), py::arg("k_out"),
      py::arg("seq_len"), py::arg("heads_q"), py::arg("heads_k"),
      py::arg("dim"), py::arg("eps"), py::arg("stream") = 0);
```

- [ ] **Step 4: Rebuild and run the test**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
cmake --build build --target flash_rt_kernels -j6
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python tests/test_groot_n17_sm89_ele_fusion.py
```

Expected: `ALL PASS`. Record the printed per-shape `bit_equal` / `max_abs` values — they go into the commit message. If `bit_equal=False` on any shape, confirm max_abs ≤ 2 ulp and cos == 1.0 still hold (reduction re-association); if max_abs exceeds the gate, debug the round-trip placement (step 2 of the math must round n to fp16 before the rope multiply).

- [ ] **Step 5: Commit**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
git add csrc/kernels/qk_norm_rope_fused.cu csrc/kernels/qk_norm_rope_fused.cuh csrc/bindings.cpp tests/test_groot_n17_sm89_ele_fusion.py
git commit -m "feat(csrc): qk_rmsnorm_rope_fused_fp16 — per-head Q/K RMSNorm + rotate-half RoPE in one launch

RMSNorm (no-bias) twin of the Chameleon qk_norm_rope_fused_fp16
(LayerNorm+bias) kernel, for GQA models whose Q and K have different
head counts (Q/K independent (HD,) norm weights, e.g. GROOT N1.7
q_norm_w/k_norm_w). Replaces the 4-launch chain rms_norm_fp16 x2 +
rope_rotate_half_fp16 x2 with a single warp-per-row kernel: zero smem,
zero __syncthreads, per-lane register cache of the normed pair halves
(fp16 round-trip preserved between the norm and rope stages).

Correctness (RTX 4090, torch 2.13/cu130): differential vs the exact
legacy chain on (S,NHQ,NHKV,HD) in {(1024,16,8,128),(257,16,8,128),
(129,24,4,128),(512,16,16,256),(1,16,8,128)}: <fill in bit_equal /
max_abs / cos from the test run>.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: fp16 instances of the fused bias+GELU kernels (template instantiation + binding)

**Files:**
- Modify: `csrc/kernels/activation.cu` (instantiate `bias_gelu_kernel<__half>` / `bias_gelu_strict_kernel<__half>` host wrappers — templates already exist at "G7.11")
- Modify: `csrc/kernels/activation.cuh` (declarations)
- Modify: `csrc/bindings.cpp` (two m.def next to `bias_gelu_inplace_bf16` — find with `grep -n "bias_gelu_inplace_bf16" csrc/bindings.cpp`)
- Test: `tests/test_groot_n17_sm89_ele_fusion.py` (append)

**Interfaces:**
- Consumes: `bias_gelu_kernel<T>` / `bias_gelu_strict_kernel<T>` templates in `csrc/kernels/activation.cu` (already compiled — instantiation only).
- Produces: `flash_rt_kernels.bias_gelu_inplace_fp16(x, bias, M, N, stream=0)` and `flash_rt_kernels.bias_gelu_inplace_strict_fp16(x, bias, M, N, stream=0)`; in-place on x `[M, N]` fp16, bias `[N]` fp16 broadcast over rows. Strict = rounds (x+bias) to fp16 before GELU (bit-identical to `add_bias_fp16` → `gelu_inplace_fp16`).

- [ ] **Step 1: Write the failing test (append to `tests/test_groot_n17_sm89_ele_fusion.py`)**

```python
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

        # non-strict fused: allowed to differ by rounding, gate at ulp
        x_fast = x.clone()
        fvk.bias_gelu_inplace_fp16(_p(x_fast), _p(bias), M, N)
        max_abs = (x_ref.float() - x_fast.float()).abs().max().item()
        assert max_abs <= 2.0 * 2.0 ** -10, (M, N, max_abs)

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
```

Add both calls to the `__main__` block. Also register `test_qwen3vl_llm_forward_fused_matches_legacy` there once Task 4's Step 1 lands.

- [ ] **Step 2: Run to verify the strict-bias-gelu test fails**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python tests/test_groot_n17_sm89_ele_fusion.py
```

Expected: `AttributeError ... bias_gelu_inplace_strict_fp16` (`bias_residual_strict_fp16` already exists from Task 1's baseline — its test should pass already; that is fine, it pins the contract we wire in Task 4).

- [ ] **Step 3: Instantiate the host wrappers**

In `csrc/kernels/activation.cu`, after `bias_gelu_inplace_bf16` (the "G7.11" section):

```cuda
void bias_gelu_inplace_fp16(__half* x, const __half* bias,
                              int M, int N, cudaStream_t stream) {
    int total2 = (M * N) >> 1;
    bias_gelu_kernel<__half><<<(total2 + 255) / 256, 256, 0, stream>>>(
        x, bias, M, N);
}
void bias_gelu_inplace_strict_fp16(__half* x, const __half* bias,
                                     int M, int N, cudaStream_t stream) {
    int total2 = (M * N) >> 1;
    bias_gelu_strict_kernel<__half><<<(total2 + 255) / 256, 256, 0, stream>>>(
        x, bias, M, N);
}
```

Declarations in `csrc/kernels/activation.cuh` (next to the bf16 twin):

```cuda
void bias_gelu_inplace_fp16(__half* x, const __half* bias,
                              int M, int N, cudaStream_t stream);
void bias_gelu_inplace_strict_fp16(__half* x, const __half* bias,
                                     int M, int N, cudaStream_t stream);
```

Bindings in `csrc/bindings.cpp` next to `bias_gelu_inplace_bf16`:

```cpp
m.def("bias_gelu_inplace_fp16", [](uintptr_t x, uintptr_t bias,
                                     int m, int n, uintptr_t stream) {
    bias_gelu_inplace_fp16(reinterpret_cast<__half*>(x),
                            reinterpret_cast<const __half*>(bias),
                            m, n, to_stream(stream));
}, py::arg("x"), py::arg("bias"), py::arg("m"), py::arg("n"),
   py::arg("stream") = 0);
m.def("bias_gelu_inplace_strict_fp16", [](uintptr_t x, uintptr_t bias,
                                            int m, int n, uintptr_t stream) {
    bias_gelu_inplace_strict_fp16(reinterpret_cast<__half*>(x),
                                    reinterpret_cast<const __half*>(bias),
                                    m, n, to_stream(stream));
}, py::arg("x"), py::arg("bias"), py::arg("m"), py::arg("n"),
   py::arg("stream") = 0);
```

- [ ] **Step 4: Rebuild and run**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
cmake --build build --target flash_rt_kernels -j6
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python tests/test_groot_n17_sm89_ele_fusion.py
```

Expected: `ALL PASS` — strict variants `torch.equal` to the legacy chains.

- [ ] **Step 5: Commit**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
git add csrc/kernels/activation.cu csrc/kernels/activation.cuh csrc/bindings.cpp tests/test_groot_n17_sm89_ele_fusion.py
git commit -m "feat(csrc): fp16 instances of the fused bias+GELU kernels

Instantiates the existing bias_gelu_kernel<T> / bias_gelu_strict_kernel<T>
templates (G7.11) for __half, with bindings. The strict variant rounds
x+bias to fp16 before the tanh-GELU — bit-identical to the
add_bias_fp16 -> gelu_inplace_fp16 two-launch chain (torch.equal across
shape sweep (1024,4304),(257,6144),(41,1536),(1,128)).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Wire the fusions into `pipeline_rtx_sm89.py`

**Files:**
- Modify: `flash_rt/models/groot_n17/pipeline_rtx_sm89.py` (7 sites)
- Test: `tests/test_groot_n17_sm89_ele_fusion.py` (append pipeline-stage differential)

**Interfaces:**
- Consumes: `qk_rmsnorm_rope_fused_fp16`, `bias_gelu_inplace_strict_fp16`, `bias_residual_strict_fp16` (Tasks 2–3).
- Produces: the wired pipeline; sites below identified by the call they replace (line numbers from origin/main 839b1597 — re-locate by content if shifted).

- [ ] **Step 1: Write the failing pipeline-stage differential test (append)**

The LLM stage is the deepest wired consumer (F1 + F2 + F3 all present). The test rebuilds the stage's legacy elementwise chains and compares outputs of the full `qwen3vl_llm_forward` against a legacy-fixture pipeline: run `qwen3vl_llm_forward` twice — once with the new code, once against a snapshot of the old chains via a monkeypatched `fvk` shim. Concretely, append:

```python
def _legacy_fvk_shim(fvk):
    """Legacy-chain shims for the three fused entry points."""
    import types
    shim = types.SimpleNamespace()
    for name in dir(fvk):
        if not name.startswith("_"):
            setattr(shim, name, getattr(fvk, name))
    def qk_rmsnorm_rope_fused(q, k, q_w, k_w, cos_t, sin_t, q_out, k_out,
                               S, NHQ, NHKV, HD, eps, stream=0):
        fvk.rms_norm_fp16(q, q_w, q_out, S * NHQ, HD, eps)
        fvk.rms_norm_fp16(k, k_w, k_out, S * NHKV, HD, eps)
        fvk.rope_rotate_half_fp16(q_out, cos_t, sin_t, S, NHQ, HD)
        fvk.rope_rotate_half_fp16(k_out, cos_t, sin_t, S, NHKV, HD)
    def bias_gelu_inplace_strict_fp16(x, bias, M, N, stream=0):
        fvk.add_bias_fp16(x, bias, M, N)
        fvk.gelu_inplace_fp16(x, M * N)
    def bias_residual_strict_fp16(res, x, bias, M, N, stream=0):
        fvk.add_bias_fp16(x, bias, M, N)
        fvk.residual_add_fp16(res, x, M * N)
    shim.qk_rmsnorm_rope_fused_fp16 = qk_rmsnorm_rope_fused
    shim.bias_gelu_inplace_strict_fp16 = bias_gelu_inplace_strict_fp16
    shim.bias_residual_strict_fp16 = bias_residual_strict_fp16
    return shim

class _StubAttn:
    """Deterministic stand-in for the attention backend. The pipeline calls
    get_slot_ptrs("llm", li) for the O slot and run("llm", li, ...) which
    must fill it; both A/B legs run the identical stub, so its math is
    irrelevant to the comparison — it only has to be deterministic and
    shape-correct (S x NHQ*HD == S x D)."""

    def __init__(self, S, NHQ, HD, D):
        self.O = torch.zeros(S, D, device=_DEV, dtype=_FP16)
        self._state = torch.randn(S, D, device=_DEV, dtype=_FP16)

    def get_slot_ptrs(self, site, li):
        return {"O": self.O.data_ptr()}

    def run(self, site, li, *, q_seq, kv_seq=None, stream=0):
        self.O.copy_(self._state)   # deterministic fill
        return self.O.data_ptr()

def _llm_fixture(S=1024):
    """Weights/bufs/dims exactly per qwen3vl_llm_forward's docstring
    (pipeline_rtx_sm89.py:272-303). fp8 weights are random e4m3 viewed as
    uint8 with scale 1.0 — value legality does not matter for the A/B."""
    from flash_rt.models.groot_n17 import pipeline_rtx_sm89 as P
    import flash_rt.flash_rt_kernels as fvk_mod

    D, NHQ, NHKV, HD, FF, L = 2048, 16, 8, 128, 5632, 16
    torch.manual_seed(7)
    fp8 = torch.float8_e4m3fn

    def rnd(*shape, dtype=_FP16):
        return torch.randn(*shape, device=_DEV, dtype=dtype)

    def fp8w(N, K):
        w = torch.randn(N, K, device=_DEV, dtype=torch.float32).to(fp8)
        ws = torch.ones(1, device=_DEV, dtype=torch.float32)
        return int(w.data_ptr()), int(ws.data_ptr())

    def wlists():
        return {"in_ln_w": [rnd(D) for _ in range(L)],
                "post_ln_w": [rnd(D) for _ in range(L)],
                "q_norm_w": [rnd(HD) for _ in range(L)],
                "k_norm_w": [rnd(HD) for _ in range(L)],
                "cos": rnd(S, HD), "sin": rnd(S, HD),
                "deepstack_inject": [0] * L,
                "act_scales": [torch.ones(1, device=_DEV, dtype=torch.float32)
                               for _ in range(L)]}

    def mk_weights():
        w = wlists()
        for name, N, K in [("q_w", NHQ * HD, D), ("k_w", NHKV * HD, D),
                           ("v_w", NHKV * HD, D), ("o_w", D, D),
                           ("gate_w", FF, D), ("up_w", FF, D),
                           ("down_w", D, FF)]:
            w[name], w[name.replace("_w", "_ws")] = fp8w(N, K)
        return w

    def mk_bufs():
        return {"h": rnd(S, D), "xn": rnd(S, D), "xn_fp8": rnd(S, D, dtype=fp8),
                "Q": rnd(S, NHQ * HD), "K": rnd(S, NHKV * HD),
                "V": rnd(S, NHKV * HD), "K_exp": rnd(S, NHQ * HD),
                "V_exp": rnd(S, NHQ * HD), "o_proj_out": rnd(S, D),
                "gate_out": rnd(S, FF), "up_out": rnd(S, FF),
                "gu_fp8": rnd(S, FF, dtype=fp8),
                "bf16_tmp": rnd(S, D, dtype=torch.bfloat16),
                "bf16_ff": rnd(S, FF, dtype=torch.bfloat16)}

    dims = {"S": S, "D": D, "NHQ": NHQ, "NHKV": NHKV, "HD": HD, "FF": FF}
    scales = {"act_qkv": wlists()["act_scales"],
              "act_o": wlists()["act_scales"],
              "act_gateup": wlists()["act_scales"],
              "act_down": wlists()["act_scales"]}
    # NOTE: if the real scales_dev contract differs (e.g. per-layer
    # integer device pointers into one tensor), follow the pipeline
    # body's `int(scales_dev["act_qkv"][li])` usage — the fixture above
    # satisfies it with a list of 1-element float tensors.
    gemm = fvk_mod.GemmRunner()
    return P, gemm, dims, mk_weights, mk_bufs, scales, (S, NHQ, NHKV, HD, D)

def test_qwen3vl_llm_forward_fused_matches_legacy():
    P, gemm, dims, mk_weights, mk_bufs, scales, (S, NHQ, NHKV, HD, D) = _llm_fixture()
    attn = _StubAttn(S, NHQ, HD, D)

    # Leg A: fused pipeline (post-Task-4 code), fresh weights/bufs.
    w1, b1 = mk_weights(), mk_bufs()
    h_seed = b1["h"].clone()
    P.qwen3vl_llm_forward(gemm, fvk, b1, w1, dims, scales, attn=attn, stream=0)

    # Leg B: legacy chains via the shim, identical weights/bufs.
    w2, b2 = mk_weights(), mk_bufs()   # same seeds -> identical values
    b2["h"] = h_seed.clone()
    legacy_attn = _StubAttn(S, NHQ, HD, D)
    legacy_attn._state = attn._state
    P.qwen3vl_llm_forward(gemm, _legacy_fvk_shim(fvk), b2, w2, dims, scales,
                          attn=legacy_attn, stream=0)

    a, b = b1["h"].float().flatten(), b2["h"].float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    max_abs = (a - b).abs().max().item()
    assert cos >= 0.9999, cos
    print(f"[llm A/B] S={S} cos={cos:.6f} max_abs={max_abs:.3e}")

- [ ] **Step 2: Locate the 7 sites and apply the edits**

Re-locate by content (line numbers from 839b1597):

F1 — LLM qk-norm + M-RoPE (`qwen3vl_llm_forward`, :357-365). Replace the four calls

```python
        fvk.rms_norm_fp16(Q_ptr, int(weights["q_norm_w"][li]), Q_ptr,
                          S * NHQ, HD, 1e-6, int(stream))
        fvk.rms_norm_fp16(K_ptr, int(weights["k_norm_w"][li]), K_ptr,
                          S * NHKV, HD, 1e-6, int(stream))
        fvk.rope_rotate_half_fp16(Q_ptr, cos_ptr, sin_ptr, S, NHQ,  HD, int(stream))
        fvk.rope_rotate_half_fp16(K_ptr, cos_ptr, sin_ptr, S, NHKV, HD, int(stream))
```

with

```python
        # Fused per-head Q/K RMSNorm + M-RoPE (bit-parity differential-
        # tested; the fp32 reduction re-associates warp-shuffle vs block-
        # tree, so chain output differs at most at fp16 ulp level).
        fvk.qk_rmsnorm_rope_fused_fp16(
            Q_ptr, K_ptr, int(weights["q_norm_w"][li]), int(weights["k_norm_w"][li]),
            cos_ptr, sin_ptr, Q_ptr, K_ptr, S, NHQ, NHKV, HD, 1e-6, int(stream))
```

F2 — bias+residual, 4 sites. In `qwen3vl_vit_forward`:

:o_proj tail (:173-174):
```python
        fvk.add_bias_fp16(o_proj_out, int(weights["o_b"][li]), S, D, int(stream))
        fvk.residual_add_fp16(h_ptr, o_proj_out, S * D, int(stream))
```
→
```python
        # strict: rounds x+bias to fp16 before the residual add —
        # bit-identical to the add_bias -> residual_add chain.
        fvk.bias_residual_strict_fp16(
            h_ptr, o_proj_out, int(weights["o_b"][li]), S, D, int(stream))
```

:fc2 tail (:200-201): identical pattern with `fc2_b`.

In `vl_self_attn_forward`: o tail (:515-516) and fc2 tail (:540-541): identical pattern.

F3 — bias+GELU, 3 sites. In `qwen3vl_vit_forward` (:190-191):

```python
        fvk.add_bias_fp16(fc1_out_ptr, int(weights["fc1_b"][li]), S, FF, int(stream))
        fvk.gelu_inplace_fp16(fc1_out_ptr, S * FF, int(stream))
```
→
```python
        fvk.bias_gelu_inplace_strict_fp16(
            fc1_out_ptr, int(weights["fc1_b"][li]), S, FF, int(stream))
```

In `deepstack_merge_forward` (:253-254) and `vl_self_attn_forward` (:530-531): identical pattern.

Preserve the surrounding comments where they explain context; delete only comments that describe the replaced two-launch sequence. Match the file's existing comment style (short `# ── section ──` headers stay).

- [ ] **Step 3: Run the differential test**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python tests/test_groot_n17_sm89_ele_fusion.py
```

Expected: all pass; record the stage-level cos / max_abs printout.

- [ ] **Step 4: Run the existing N1.7 suite (no fixture → should still pass/skip identically to Task 1's baseline)**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python -m pytest tests/ -k groot_n17 -x -q --no-header 2>&1 | tail -5
```

Expected: same pass/skip counts as the Task 1 baseline.

- [ ] **Step 5: Commit**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
git add flash_rt/models/groot_n17/pipeline_rtx_sm89.py tests/test_groot_n17_sm89_ele_fusion.py
git commit -m "perf(groot_n17): fuse the SM89 backbone elementwise chains

Wires three fusion families into pipeline_rtx_sm89.py (all direct
replacements, matching the Thor fusion series 9ad764e6 style):

  - LLM qk-norm + M-RoPE: 4 launches -> 1 qk_rmsnorm_rope_fused_fp16
    per layer (per-head Q/K RMSNorm + rotate-half RoPE, warp-per-row,
    zero smem).
  - bias+residual (ViT o/fc2, VL self-attn o/fc2): add_bias_fp16 +
    residual_add_fp16 -> bias_residual_strict_fp16, bit-identical.
  - bias+GELU (ViT/deepstack/VL fc1): add_bias_fp16 + gelu_inplace_fp16
    -> bias_gelu_inplace_strict_fp16, bit-identical.

Per-layer launch delta (LLM, 16 layers): -3; (ViT 27 + VL 16 blocks):
-2 each on the six wired sites. Under CUDA graph the win is HBM traffic
(each fused site removes one full activation-row round trip), plus a
smaller captured graph.

Correctness (RTX 4090, random weights, seed-fixed): <fill from the test
run — stage-level cos and per-site torch.equal results>.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Latency benchmark, docs, final review

**Files:**
- Create: `benchmarks/groot_n17_sm89_ele_fusion_bench.py`
- Modify: `docs/kernel_catalog.md`, `docs/kernel_fusion.md` (counts + entries)

**Interfaces:**
- Consumes: the Task-4 fixture helper for stage weights.
- Produces: perf numbers for the PR body (upstream CONTRIBUTING requires GPU/CUDA/command/latency+precision in the PR description).

- [ ] **Step 1: Write the benchmark**

`benchmarks/groot_n17_sm89_ele_fusion_bench.py` — reuse the test's fixture helper; time `qwen3vl_llm_forward` (fused) vs the shimmed legacy chain over ≥200 iterations after 20 warmups, `torch.cuda.synchronize()` around timing, median of 5 runs; plus a kernel-level microbench of `qk_rmsnorm_rope_fused_fp16` vs the 4-launch chain at S ∈ {257, 1024, 4096}, NHQ/NHKV/HD = 16/8/128. Print a table. Structure it after `benchmarks/bench_lm_forward.py` (read it first; follow its arg/iteration conventions).

- [ ] **Step 2: Run and record**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python benchmarks/groot_n17_sm89_ele_fusion_bench.py
```

Record the numbers. Expectation from the roofline: the 4-launch chain reads+writes Q/K four times (~3× the fused traffic at S=1024); a 1.5–2.5× kernel-level speedup and a measurable stage-level delta are in range — report whatever is measured, do not tune to a target.

- [ ] **Step 3: Update docs**

- `docs/kernel_catalog.md`: add `qk_rmsnorm_rope_fused_fp16`, `bias_gelu_inplace_fp16`, `bias_gelu_inplace_strict_fp16` entries (match the file's existing entry format; update the §3 count).
- `docs/kernel_fusion.md`: add the three GROOT N1.7 SM89 wiring sites to the fusion tables if the file catalogs per-model coverage (read the file's structure first and follow it; if it only catalogs kernels, add the kernel rows).

- [ ] **Step 4: Final full-suite run**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python -m pytest tests/ -k groot_n17 -q --no-header 2>&1 | tail -3
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python tests/test_groot_n17_sm89_ele_fusion.py
```

Expected: baseline pass/skip counts; `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
git add benchmarks/groot_n17_sm89_ele_fusion_bench.py docs/kernel_catalog.md docs/kernel_fusion.md
git commit -m "bench(groot_n17): SM89 elementwise-fusion latency benchmark + docs

<fill: one-paragraph summary of the measured numbers, GPU, CUDA,
command line per CONTRIBUTING>

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Branch summary for the PR body** — assemble the verification block (correctness: per-site torch.equal list + stage cos; latency table; test commands; GPU/CUDA versions) and hand it to the author for the PR description. Do NOT push.

---

### Task 6: DiT-head elementwise fusion in the shared `dit_forward` (bf16 fallback branch)

**Files:**
- Create: `csrc/kernels/qkv_split_bias_gqa.cu` + `csrc/kernels/qkv_split_bias_gqa.cuh` (new TU; CMake source-list edit needed — find `qk_norm_rope_fused` in `CMakeLists.txt` and add the new TU next to it)
- Modify: `flash_rt/models/groot_n17/pipeline_thor.py` (3 sites in `dit_forward`, bf16 fallback branch only)
- Modify: `csrc/bindings.cpp` (two m.def)
- Test: `tests/test_groot_n17_sm89_ele_fusion.py` (append)

**Interfaces:**
- Consumes: existing `bias_residual` (bf16, bindings.cpp:1120), `bias_gelu_inplace_bf16` (bindings.cpp:1035 — verify name; the bf16 fused bias+GELU exists at "G7.11").
- Produces: `flash_rt_kernels.qkv_split_bias_gqa_bf16(packed_qkv, qkv_bias, q_out, k_out, v_out, k_exp_out, v_exp_out, seq_len, dim, heads_q, heads_k, stream=0)` — packed `[S, 3D]` bf16 (row = [Q | K | V] with Q `heads_q*HD`, K/V `heads_k*HD`), bias `[3D]` bf16, outputs Q `[S, heads_q*HD]`, K `[S, heads_k*HD]`, V `[S, heads_k*HD]`, and (optional, 0 disables) GQA-expanded K/V `[S, heads_q*HD]` written by head-broadcast from K/V (source head `h // (heads_q/heads_k)`). And: `bias_residual_strict_bf16` — verify whether it already exists (`grep -n bias_residual_strict csrc/bindings.cpp`); the bf16 strict twin of the fp16 one; add only if missing.

**Scope guard:** the fused-`bf16_nn_bias` epilogue failure the comments describe (CUBLAS_STATUS_NOT_SUPPORTED at M=Sa=41) is why the fallback chain exists. Our fusions never touch GEMMs — only the elementwise post-GEMM chains — so the M=41 GEMM constraint is untouched.

**Sharing note:** `dit_forward` is shared by Thor / sm89 / RTX-fp16 frontends. All edits are dtype-math-identical replacements of elementwise chains, exercised on sm89 locally; the PR notes Thor/sm120 equivalence is by kernel-identity argument plus maintainer re-run.

- [ ] **Step 1: Write the failing tests (append)**

```python
def test_qkv_split_bias_gqa_bf16_matches_legacy_chain():
    # GROOT DiT self-attn shapes: Sa=41, D=1536, NH=32, HD=48 → but kernel
    # is generic; sweep asymmetric GQA too.
    for (S, NHQ, NHKV, HD) in [(41, 16, 4, 128), (1024, 16, 8, 128),
                               (257, 24, 4, 128)]:
        D = NHQ * HD
        packed = torch.randn(S, 3 * D, device=_DEV, dtype=_BF16)
        bias = torch.randn(3 * D, device=_DEV, dtype=_BF16)
        q, k, v = (torch.randn(S, h * HD, device=_DEV, dtype=_BF16)
                   for h in (NHQ, NHKV, NHKV))
        # legacy chain (bit-parity target): add_bias_bf16 in place on packed
        # then 3 strided copies
        ref_packed = packed.clone()
        fvk.add_bias_bf16(_p(ref_packed), _p(bias), S, 3 * D)
        q_ref = ref_packed[:, :NHQ * HD].contiguous()
        k_ref = ref_packed[:, NHQ * HD:NHQ * HD + NHKV * HD].contiguous()
        v_ref = ref_packed[:, NHQ * HD + NHKV * HD:].contiguous()
        # fused
        fvk.qkv_split_bias_gqa_bf16(
            _p(packed), _p(bias), _p(q), _p(k), _p(v), 0, 0,
            S, D, NHQ, NHKV)
        assert torch.equal(q, q_ref) and torch.equal(k, k_ref) \
            and torch.equal(v, v_ref), (S, NHQ, NHKV, HD)

def test_bias_residual_strict_bf16_matches_legacy_chain():
    for (M, N) in [(41, 1536), (1024, 2048)]:
        res = torch.randn(M, N, device=_DEV, dtype=_BF16)
        x = torch.randn(M, N, device=_DEV, dtype=_BF16)
        bias = torch.randn(N, device=_DEV, dtype=_BF16)
        res_ref, x_ref = res.clone(), x.clone()
        fvk.add_bias_bf16(_p(x_ref), _p(bias), M, N)
        fvk.residual_add(_p(res_ref), _p(x_ref), M * N)
        res_strict, x_strict = res.clone(), x.clone()
        fvk.bias_residual_strict_bf16(_p(res_strict), _p(x_strict), _p(bias), M, N)
        assert torch.equal(res_ref, res_strict), (M, N)
```

Add both calls to the `__main__` block. (`_BF16 = torch.bfloat16` constant — add near `_FP16`.)

- [ ] **Step 2: Run to verify failure**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python tests/test_groot_n17_sm89_ele_fusion.py
```

Expected: `AttributeError ... qkv_split_bias_gqa_bf16` (and possibly `bias_residual_strict_bf16` — check the grep in this task's Interfaces first; if the bf16 strict twin already exists, that test passes immediately and pins the contract).

- [ ] **Step 3: Implement the kernels**

`csrc/kernels/qkv_split_bias_gqa.cu`:

```cuda
// ================================================================
// FlashRT — Fused QKV split + bias + optional GQA head-broadcast (BF16)
//
// Replaces the GROOT N1.7 DiT self-attn post-GEMM chain (4 launches
// -> 1):
//   add_bias_bf16(packed, qkv_bias, S, 3D)
//   gpu_strided_copy_fp16(packed, Q, S, D,     3D, 0)
//   gpu_strided_copy_fp16(packed, K, S, HKV*HD, 3D, D)
//   gpu_strided_copy_fp16(packed, V, S, HKV*HD, 3D, D + HKV*HD)
// (plus, when the attention backend needs expanded K/V, the two
// gpu_repeat_interleave_heads calls — folded here as k_exp/v_exp.)
//
// Layout: packed rows are [Q (HQ*HD) | K (HKV*HD) | V (HKV*HD)] bf16.
// fp32 bias add, bf16 round-trip after the add (matches add_bias_bf16's
// two-round strict semantics? NO — add_bias_bf16 writes bf16(x+bias)
// as the final value of the copy; the fused split is one round).
// Bit-parity: torch.equal against the legacy chain pinned by the test.
// ================================================================
```

(then the kernel: 1 thread per output element; grid over `S * 3D` total elements in Q/K/V segment order; each thread computes its segment by comparing its flat index against `S*HQ*HD` and `S*(HQ+HKV)*HD` boundaries; Q threads write `q_out[flat] = bf16(fp32(packed[flat]) + fp32(bias[flat % D_row... ]))` — compute the per-segment column as `flat_in_segment % seg_width` and read `bias[seg_offset + col]`; K/V threads additionally, if `k_exp_out != 0`, write `k_exp_out[..., dst_head*HD + d]` with `dst_head = seg_row_head_index` and source head `dst_head // gqa_ratio` — i.e. loop over `gqa_ratio` destination heads per source K element. `gqa_ratio = heads_q / heads_k` — assert it divides evenly in the host wrapper.)

Host wrapper + `extern "C" flash_rt_qkv_split_bias_gqa_bf16` + `qkv_split_bias_gqa.cuh` declaration, following the `qk_norm_rope_fused.cu/.cuh` pattern. For `bias_residual_strict_bf16` (if missing): the fp16 twin `bias_res_strict_fp16_kernel` in `csrc/kernels/elementwise.cu` is the template — instantiate `__nv_bfloat16` with the same two-round math and add the binding next to `bias_residual`.

- [ ] **Step 4: Rebuild, run tests, run existing suite**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
cmake --build build --target flash_rt_kernels -j6
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python tests/test_groot_n17_sm89_ele_fusion.py
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python -m pytest tests/ -k groot_n17 -q --no-header 2>&1 | tail -3
```

Expected: `ALL PASS`; suite counts match Task 1's baseline.

- [ ] **Step 5: Wire `dit_forward` (3 sites, bf16 fallback branch only)**

In `flash_rt/models/groot_n17/pipeline_thor.py` `dit_forward`, locate the final `else:` branch (the one starting `gemm.bf16_nn(xn_ptr, int(weights["q_w"][li]), Q_ptr, ...)`). Within it:

**Site D1 — self-attn QKV** (the 3 GEMM + 3 add_bias sequence). The split kernel consumes a *fused* QKV GEMM output, but this branch runs 3 separate GEMMs. Two options, decide by reading the buffer situation: (a) keep 3 GEMMs writing into `qkv_buf` slices (needs `bufs["qkv_buf"]` present — the fp8 branch already uses it), then one `qkv_split_bias_gqa_bf16` call for bias+split; (b) leave the 3 GEMMs direct to slots (they already write Q_ptr/K_ptr/V_ptr) and drop the add_bias from the GEMM path — then the fusion collapses to just folding bias into the strided copy, which the direct-GEMM layout already achieves. **Read the actual buffer availability in `_run_dit` (frontend `bufs_ptrs` has only h/xn/o_proj_out/ff_proj_out — NO `qkv_buf`!).** Therefore option (b) variant: run the 3 bf16 GEMMs into `o_proj_out`-style staging is NOT available either. The minimal correct move here is: **keep the 3 GEMMs as-is (they write the attention slots directly) and only fuse the bias into them via the existing `bias_residual`-family — i.e. D1 becomes: move `add_bias_bf16` calls into the *next* consumer or skip D1 if the layout doesn't allow a fused QKV GEMM without a new buffer.** Resolve this in implementation: if `_run_dit`'s buffer dict lacks a (Sa, 3D) staging buffer, ADD one (`bufs_ptrs["qkv_buf"] = torch.empty(Sa, 3*D)`) in the RTX frontend's `_run_dit` and switch the branch to a single fused `gemm.bf16_nn` of shape (Sa, 3D) — requires the Q/K/V weights to be concatenated at set_prompt time in the frontend. **This is the one structural change of Task 6** — it mirrors the internal-repo qwen35vla "fused QKV GEMM" step (b63d2bf). Update `_run_dit` in `flash_rt/frontends/torch/groot_n17_rtx.py` accordingly (weight concat once at load: `torch.cat([q_w, k_w, v_w], dim=0)` per layer — note K/V are GQA-smaller, concat along output dim; bias likewise) and pass `qkv_w`/`qkv_b` keys instead of separate q/k/v.
  - GQA note: the attention backend slots expect `K` at `[S, heads_q*HD]` expanded or `[S, heads_k*HD]` packed depending on backend — read `RtxFlashAttnBackendGrootN17`'s dit_self slot shapes (`flash_rt/hardware/rtx/attn_backend_groot_n17.py`) and match: pass the expanded outputs (`k_exp_out/v_exp_out`) when the slots are GQA-expanded.

**Site D2 — o_proj tail** (`gemm.bf16_nn(O_ptr, o_w, o_out_ptr)` + `add_bias_bf16` + `residual_add`): replace the last two calls with `fvk.bias_residual_strict_bf16(h_ptr, o_out_ptr, o_b[li], Sa, D)`.

**Site D3 — FFN tail** (`gemm.bf16_nn(ff_out, ff_down_w, o_out_ptr)` + `add_bias_bf16` + `residual_add`): same pattern, `ff_down_b`.

**Site D4 — FFN up bias+GELU** (`add_bias_bf16(ff_out_ptr, ff_proj_b)` + `gelu_inplace`): replace with `fvk.bias_gelu_inplace_bf16(ff_out_ptr, ff_proj_b[li], Sa, FF)` (kernel exists — "G7.11"; binding `bias_gelu_inplace_bbf16`? verify exact name via `grep -n "bias_gelu" csrc/bindings.cpp`, it is `bias_gelu_inplace_bf16` at line 1035).

**CRITICAL**: these edits are in the **shared** `dit_forward` — the fp8/NVFP4 branches of Thor frontends must be untouched (edits confined to the `else:` bf16 fallback branch; the fp8 branches already have their own fusions). Verify by grep that no `qkv_w_fp8` key is consumed by our edited code path.

- [ ] **Step 5 (cont.): D1 structural change detail**

The weight concat happens **once at load** in the frontend, not per-forward: in `GrootN17TorchFrontendRtx._run_dit` (or its weight-loading `set_prompt` equivalent — find where `_dit_q_w` etc. are assigned), build fused tensors:

```python
# once, at weight-load time (frontend)
self._dit_qkv_w = [torch.cat([q, k, v], dim=0).contiguous()   # [3D, D]
                   for q, k, v in zip(self._dit_q_w, self._dit_k_w, self._dit_v_w)]
self._dit_qkv_b = [torch.cat([bq, bk, bv], dim=0).contiguous()  # [3D]
                   for bq, bk, bv in zip(self._dit_q_b, self._dit_k_b, self._dit_v_b)]
```

then in `_run_dit`'s weights dict pass `qkv_w`/`qkv_b` (and drop `q_w/k_w/v_w/q_b/k_b/v_b` only if no other consumer reads them — grep first; the cross-attn Q GEMM reads `q_w[li]` too, keep it). Pass a new staging buffer `bufs_ptrs["qkv_buf"] = torch.empty(Sa, 3 * 1536, ...)`: allocate in `_build_dit_attn` or lazily per unique Sa (the frontend caches by Sa already — `if not hasattr(self, "_dit_attn")`). Then in `dit_forward` bf16 self-attn branch:

```python
        gemm.bf16_nn(xn_ptr, int(weights["qkv_w"][li]),
                      qkv_buf_ptr, Sa, 3 * D, D, int(stream))
        fvk.qkv_split_bias_gqa_bf16(
            qkv_buf_ptr, int(weights["qkv_b"][li]),
            Q_ptr, K_ptr, V_ptr, K_exp_ptr, V_exp_ptr,
            Sa, D, NHQ, NHKV, int(stream))
```

(NHQ/NHKV enter via `dims`; add `NH`/`HD` keys to `_run_dit`'s dims dict — DiT self-attn is NH=32 heads, HD=48. K/V slot layout per the attention backend decides whether K_exp/V_exp are used or 0.)

- [ ] **Step 6: Differential test for the DiT stage (append)**

Same pattern as Task 4's LLM A/B: `_StubAttn` variant for the DiT (sites `dit_self`/`dit_cross`, slot dicts with Q/K/V/O), legacy-chain shim for `qkv_split_bias_gqa_bf16` / `bias_residual_strict_bf16` / `bias_gelu_inplace_bf16` (legacy = `add_bias_bf16` + strided copies / `add_bias_bf16`+`residual_add` / `add_bias_bf16`+`gelu_inplace`), random bf16 weights at Sa=41, and an end-to-end `dit_forward` comparison fused vs shimmed-legacy. Gate: `torch.equal` on all D2/D3/D4-touched outputs is impossible layerwise (AdaLN/attention are common); gate at cos ≥ 0.9999 + reported max_abs across the full `h` after 32 layers, plus `torch.equal` for D1 on the first self-attn layer's Q/K/V slot contents (kernel-level test already pins bit-parity; the stage test gates accumulation).

- [ ] **Step 7: Rebuild, full test run, existing suite**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
cmake --build build --target flash_rt_kernels -j6
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python tests/test_groot_n17_sm89_ele_fusion.py
/home/zhangaoxiang/miniconda3/envs/flashrt_pi05/bin/python -m pytest tests/ -k groot_n17 -q --no-header 2>&1 | tail -3
```

Expected: `ALL PASS`; suite baseline counts unchanged.

- [ ] **Step 8: Commit**

```bash
cd /home/zhangaoxiang/code/flashrt_github/FlashRT
git add csrc/kernels/qkv_split_bias_gqa.cu csrc/kernels/qkv_split_bias_gqa.cuh CMakeLists.txt csrc/bindings.cpp flash_rt/models/groot_n17/pipeline_thor.py flash_rt/frontends/torch/groot_n17_rtx.py tests/test_groot_n17_sm89_ele_fusion.py
git commit -m "perf(groot_n17): fuse the DiT-head elementwise chains (bf16 fallback branch)

Shared dit_forward's bf16 fallback — used by the RTX sm89 frontend —
ran per self-attn block: 3 Q/K/V GEMMs + 3 add_bias + (per backend)
GQA expand copies; o/FFN-down tails as add_bias+residual_add; FFN up
as add_bias+gelu. This PR:

  - self-attn QKV: fused (Sa,3D) bf16 GEMM (weights concatenated at
    load) + qkv_split_bias_gqa_bf16 (split+bias+optional GQA expand,
    4+ launches -> 2), mirrors the internal qwen35vla fused-QKV step.
  - o/FFN tails: bias_residual_strict_bf16 (bit-identical two-round).
  - FFN up: bias_gelu_inplace_bf16 (G7.11, existing kernel wired).

fp8/NVFP4 branches of the shared dit_forward are untouched (they
carry their own epilogue fusions). Thor/sm120 equivalence: kernel-
identical replacements; maintainer re-run requested in the PR.

Correctness (RTX 4090, random weights): <fill — torch.equal per-site
and stage cos / max_abs>. Latency: <fill>.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 9: bench + docs for the DiT part** — extend `benchmarks/groot_n17_sm89_ele_fusion_bench.py` with a `dit_forward` leg (fused vs shimmed legacy, ≥200 iters, 4 denoise steps simulated or single dit_forward call, median of 5); update `docs/kernel_catalog.md` (+`qkv_split_bias_gqa_bf16`, +strict-bf16 entries) and `docs/kernel_fusion.md`; commit as part of Task 5's bench commit or its own if Task 5 already landed.
