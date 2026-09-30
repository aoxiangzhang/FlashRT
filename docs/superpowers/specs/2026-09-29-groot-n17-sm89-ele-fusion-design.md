# GROOT N1.7 RTX SM89 Elementwise Fusion — Design Spec

## Background

The FlashRT internal fork (qwen3.5-VLA / mr0 work by this author) accumulated a
set of validated elementwise-fusion patterns: QKV post-processing fusion
(split + per-head qk-norm + RoPE in one kernel), bias+residual / bias+gelu
wiring onto existing upstream kernels, and A/B differential testing
(fused path vs unfused path, bit-identical or cos gate) with random weights.

The open-source GROOT N1.7 RTX pipelines (`flash_rt/models/groot_n17/
pipeline_rtx_sm89.py`, 541 lines) never received the elementwise-fusion
treatment: git history shows the RTX side was stood up as a functional
baseline (2026-06-01), given FP8 NT fused epilogues + CUDA graph coverage
(#160, 2026-08-02), and then development moved to Thor tiers and NPU. The
same fusions were done on the Thor path (Fuse LLM RMSNorm+residual+FP8
quantize, Fuse DiT AdaLN+FP8 quantize, Fuse DiT QKV, 2026-06-02 series) —
this work ports that proven playbook to the RTX SM89 path.

**Why it still matters under CUDA graph**: graph replay removes launch
overhead but not bandwidth. The unfused chains re-read/re-write Q/K/V and
activation rows multiple times; the fusion win is HBM traffic, plus a
smaller graph.

## Goal

Fuse the elementwise chains of the GROOT N1.7 SM89 path
(`pipeline_rtx_sm89.py`) using (where possible) already-existing upstream
kernels, plus one new kernel variant. All changes default-off or
numerically transparent; verified by A/B differential tests on random
weights on a local RTX 4090.

In-scope (added after design review): the shared `dit_forward`'s **bf16
fallback branch** (the one the RTX sm89 frontend takes — its `_run_dit`
passes bf16 weight keys, hitting the `else: gemm.bf16_nn` + `add_bias_bf16`
+ `gelu_inplace` + `residual_add` chain): fused QKV GEMM (weights
concatenated at load) + new `qkv_split_bias_gqa_bf16` kernel, wired
`bias_residual_strict_bf16` at o/FFN tails, wired `bias_gelu_inplace_bf16`
at FFN up. The fp8/NVFP4 branches of the shared function stay untouched.

Non-goals: SM120 (5090) wiring (follow-up once hardware is available; the
same edits apply to `pipeline_rtx_fp8.py` / `pipeline_rtx_fp16.py` by
analogy), Thor's own fp8/NVFP4 tiers, NPU/AMD, any new FP8
coverage on ViT/SigLIP.

## Fusion inventory (per site)

### F1 — Backbone QKV post-processing (7 launches → 2)

Current (`pipeline_rtx_sm89.py`, backbone per layer): after the fused QKV
GEMM, `gpu_strided_copy_fp16` ×3 (Q/K/V split) + `rms_norm_fp16` ×2
(per-head q/k norm on (Se*NHQ, 128) views) + `rope_rotate_half_fp16` ×2.

Plan: keep `qkv_split`-style split (1 launch, `qkv_split_fp16` if the
layout matches, else keep 3 strided copies), then one new kernel
**`qk_rmsnorm_rope_fused_fp16`** applying per-head RMSNorm (independent
Q/K weights, no bias, eps=1e-6) + full-dim rotate-half RoPE in place on
Q and K — 2 launches total.

New kernel lives next to the existing `qk_norm_rope_fused_fp16`
(`csrc/kernels/qk_norm_rope_fused.cu`, the Chameleon-7B LayerNorm+bias
variant): same file, RMSNorm no-bias variant, no model suffix
(cross-model shared convention). fp32 accumulation, fp16 IO.

### F2 — DiT head bias+residual / bias+gelu wiring (existing kernels)

Per DiT block per denoise step (~13 launches):
- `add_bias_fp16` + `residual_add_fp16` (o_proj tail, FFN-down tail) →
  `bias_residual_fp16` (exists upstream)
- `add_bias_fp16` + `gelu_inplace_fp16` (FFN-up) → `bias_gelu_fp16_strict`
  (exists upstream; verify strict semantics match — GROOT uses exact
  GELU, pipeline currently calls `gelu_inplace_fp16`)
- `add_bias_fp16` + split copies (DiT QKV) → `qkv_split_bias_fp16`
  variant if worth it, else fold bias into the split copies

### F4 — DiT-head bf16 fallback branch (shared dit_forward)

Per self-attn block: 3 Q/K/V GEMMs + 3 add_bias (+ GQA expand copies);
o/FFN-down tails add_bias+residual_add; FFN up add_bias+gelu. Plan: fused
(Sa,3D) QKV GEMM via load-time weight concat + new
`qkv_split_bias_gqa_bf16` (split+bias+optional GQA expand); strict
bias+residual at tails; existing `bias_gelu_inplace_bf16` at FFN up.
Sa=41 → launch-bound; elementwise fusion matters despite small rows.

### F3 — FFN bias+GELU (corrected scope)

The originally-sketched "merged gate/up silu_mul" F3 is void: the SM89 LLM
FFN already runs `silu_mul_split_fp8_fp16` as a single fused launch
(pipeline_rtx_sm89.py:408). The real remaining FFN-site gap is
`add_bias_fp16` + `gelu_inplace_fp16` at the ViT / DeepStack / VL FFN
first-projection sites (:190-191, :253-254, :530-531) — fused via the
templated `bias_gelu_strict_kernel<__half>` (bf16 twin exists at "G7.11"),
instantiated + bound + wired.

Priority: F1 > F2 > F3 (bandwidth rank). All shipped in one PR if test
results are clean, else split.

## Kernel changes

1. **New**: `qk_rmsnorm_rope_fused_fp16` in
   `csrc/kernels/qk_norm_rope_fused.cu` (+ .cuh decl, bindings.cpp m.def,
   CMake already compiles the TU — verify). Grid/block follow the
   Chameleon sibling (32×8 threads, per-lane register cache, dim ≤ 256).
2. **Possibly new**: fp16 merged `silu_mul` (F3) — only if the existing
   `silu_mul_split_fp8_fp16` path can't be reshaped (decision during
   implementation, prefer reshaping).
3. Wiring-only for F2 kernels.

## Verification (on RTX 4090, sm89)

1. **Kernel-level**: three-way alignment test for the new kernel
   (torch reference vs kernel, cos=1.0 / max_abs ≤ 1 fp16 ulp on
   normalized magnitudes), template following
   `tests/test_groot_n17_*` + `tests/test_generic_kernel_bindings.py`.
2. **Pipeline A/B differential**: random-weights GROOT N1.7 SM89
   pipeline, fused vs unfused path (env/flag toggle), same input+seed:
   logits/action outputs cos ≥ 0.9999, and for F2/F3 bit-identical
   (elementwise-exact math). This mirrors the internal repo's
   three-way-alignment methodology and needs no checkpoint.
3. **Existing test suite**: `pytest tests/ -k groot_n17` (fixture-gated
   tests skip cleanly without checkpoints).
4. **Latency**: e2e random-weights latency (backbone + DiT 4 steps),
   median of ≥20, fused vs unfused; per-site microbench for F1.
5. **Build**: `cmake --build --target flash_rt_kernels` clean on sm89;
   CONTRIBUTING §binding/CMake guard alignment check.

## PR conventions

- Commit message: design + correctness + perf numbers with GPU/CUDA/
  command line (CONTRIBUTING.md requirement).
- Opt-in or numerically transparent (F2/F3 must be bit-identical; F1's
  norm+rope reassociation is sub-ulp, documented with the A/B numbers).
- Update `docs/kernel_catalog.md` + `docs/kernel_fusion.md` counts.
- Work stays on local branch `feature/groot-n17-sm89-ele-fusion`; the
  author pushes and opens the PR.

---

# Addendum (2026-09-29): PR-2 — DiT QKV GEMM structural merge

Approved in-session: develop + verify on 4090 now, re-verify on 5090 later.

## Scope
Shared `dit_forward` bf16 fallback branch (self-attn layers): 3 Q/K/V
GEMMs + 3 add_bias (6 launches/layer) -> 1 packed (Sa, 3D) bf16 GEMM +
qkv_split_bias_bf16 split (2 launches/layer), matching the Thor fp8
path's existing load-time concat precedent (groot_n17_thor.py:678,
torch.cat([q_w,k_w,v_w], dim=1)).

## Contract
Opt-in via weights keys: when `qkv_w`/`qkv_b` (list[16], packed
(D, 3D) row-major == (3D, D) GEMM input) are present AND
`bufs["qkv_buf"]` (Sa, 3D bf16) is supplied, the merged path runs;
otherwise the 3-GEMM path is untouched. Cross-attn layers keep the
single Q GEMM (q_w key unchanged). Thor fp8/NVFP4 branches untouched.

## Consumers
- groot_n17_rtx.py `_run_dit`: load-time concat for the 16 self-attn
  layers + staging buffer + new keys. (sm89/sm120 shared.)
- Thor frontend NOT wired (has its own fp8 concat path).

## Verification (4090 now)
- DiT stage A/B merged vs 3-GEMM chain (same GEMM runner): bit-equal.
- Golden frozen-chain test extended to the merged path.
- e2e bench delta. Go/no-go: commit only if DiT stage improves >=5%;
  5090 re-run later confirms the sm120 cuBLASLt scheduling delta.

## Dependency
Consumes PR-1's main-module binding of qkv_split_bias_bf16
(commit 08c6bc2c lineage). PR-2 branch starts from PR-1's branch tip
(rebased onto origin/main) so both remain reviewable independently.

---

# Addendum (2026-09-30): PR-3 — pipeline_rtx_fp8.py elementwise fusion

## Correction to the earlier feasibility claim
`pipeline_rtx_fp8.py`'s GEMM primitive `fp8_descale_fp16` (FP16-out
cuBLASLt FP8) is NOT_SUPPORTED on sm89 — Ada cuBLASLt FP8 requires
BF16 output (the sm89 pipeline's fp8_nt_dev + cast workaround exists
precisely for this). So the fp8 pipeline cannot run end-to-end on the
4090 as-is. Revised verification split:

- 4090 (dev machine): wiring-level stage A/B with a GEMM shim
  (fp8_nt_dev bf16-out + cast_bf16_to_fp16 standing in for
  fp8_descale_fp16). Both A/B legs share the shim, so the fused-vs-
  legacy elementwise differential is still isolated and valid for
  proving the WIRING. Kernel math itself is already pinned by the
  PR-1 kernel-level tests (identical kernels, identical shapes).
- 5090: same one-shot script — the real fp8_descale_fp16 runs there;
  expected delta on the production backbone path (README 16.6ms tier).

## Fusion sites (mirror of the sm89 wiring)
- LLM qk-norm + M-RoPE: qk_norm_rope_rotate_half_fp16 (4->2 launches)
- bias+residual (o/fc2 tails, ViT/DS/VL): bias_residual_strict_fp16
- bias+GELU (fc1): bias_gelu_inplace_strict_fp16
All strict -> bit-identical chains; dit_forward untouched (its fp8
branch already fuses).
