# PR draft — GROOT N1.7 elementwise + QKV fusion (feature/groot-n17-ele-fusion)

## Summary

The GROOT N1.7 RTX (SM89) paths never received the elementwise-fusion
treatment the Thor tier got in June (cf. "Fuse the GROOT N1.7 Thor LLM
RMSNorm, residual add and FP8 quantize", 9ad764e6): the fp16 backbone
ran unfused bias/residual/GELU chains and the per-head qk-norm + M-RoPE
as four launches per layer; the shared DiT head's bf16 tier ran 3
separate launch-bound Q/K/V GEMMs. This PR ports the Thor fusion
playbook to BOTH SM89 backbones + the shared DiT bf16 tier + the FP8
(SM120 production) backbone, reusing existing kernels wherever one
existed and adding only dtype variants where none matched.
**Every wired fusion is bit-identical to the chain it replaces**
(strict two-round rounding preserved; torch.equal on hardware), except
the qk-norm+RoPE kernel whose reduce re-association is verified at
1 fp16 ulp against each leg's own fp64 reference.

## Changes (9 commits, reviewed as two logical halves)

**Elementwise fusions (strict, bit-identical):**
1. `qk_norm_rope_rotate_half_fp16` — the fused per-head Q/K RMSNorm +
   rotate-half RoPE kernel (N1.6 Thor tier) templated to fp16 and moved
   out of the Thor-only build guard. LLM backbone: 4 launches -> 2/layer.
2. fp16 instances of `bias_gelu_inplace[_strict]_fp16` (G7.11 templates)
   and a bf16 twin `bias_residual_strict_bf16`.
3. Backbone wiring (`pipeline_rtx_sm89.py`): qk-norm+RoPE (LLM),
   bias+residual x4, bias+GELU x3 (ViT / DeepStack / VL self-attn).
4. DiT wiring (shared `dit_forward`): o-tail bias+residual (all
   branches) and FFN-up bias+GELU (bf16 tier); the common FFN tail
   residual is byte-identical to origin/main (a dropped-residual bug
   was caught in review and fixed — see Testing).
5. `qkv_split_bias_bf16` / `residual_add_bias_bf16` exposed in the main
   module (previously qwen3_vl-gated), with a proper `.cuh`.

**Packed-QKV structural merge (opt-in):**
6. DiT self-attn bf16 tier: 3 Q/K/V GEMMs + 3 add_bias (6 launches) ->
   one (Sa, 3D) bf16 GEMM over load-time-concatenated weights +
   `qkv_split_bias_bf16` scatter (2 launches), mirroring the concat
   convention the Thor fp8 tier already uses. Activates only when
   `weights['qkv_w'/'qkv_b']` + `bufs['qkv_buf']` are supplied; every
   other consumer falls back to the 3-GEMM path untouched.
7. RTX frontend (sm89/sm120 shared) load-time concat + staging buffer.
8. **FP8 pipeline (SM120 production tier)** — `pipeline_rtx_fp8.py`
   (the README 5090 FP8 path's backbone): the identical 7 strict fusion
   sites wired, same kernels and shapes as the SM89 pipeline.
9. `benchmarks/groot_n17_sm89_{ele_fusion,e2e}_bench.py` +
   `scripts/verify_groot_n17_fusion_rtx.sh` (the one-shot verification
   used during development, packaged for a 5090 re-run).
10. kernel_catalog.md / kernel_fusion.md updated (counts reconciled).

## Verification (dual-GPU: RTX 4090 D + RTX 5090)

Run everything on either GPU: `PY=<python> ./scripts/verify_groot_n17_fusion_rtx.sh`

Primary numbers below from RTX 4090 D (dsl-dev torch 2.13.0+cu130,
nvcc 13.1); the 5090 re-run (torch 2.11.0+cu128, CUDA 12.8) reproduced
every correctness gate and improved the latency deltas — see the 5090
column.

Correctness — `tests/test_groot_n17_sm89_ele_fusion.py` (12 tests):
- kernel-level: strict variants `torch.equal` to the legacy chains
  across shape sweeps; qkv_split_bias `torch.equal`; the qk-norm fusion's
  divergent elements (46/46 at S=1024) within 1 fp16 ulp of each leg's
  own fp64 rope reference.
- stage-level: full 16-layer LLM fp8 backbone A/B — **torch.equal
  (bit-equal)**. Full 32-layer DiT A/B — **bit-equal**. Merged-QKV layer
  vs the 3-GEMM chain — **bit-equal**, with a path-selection guard
  (asserts the merged path actually ran).
- golden: one DiT layer vs a frozen copy of the origin/main chain —
  torch.equal; mutation-tested (deleting the FFN residual fails at
  9236 abs). This closes the A/B blind spot where both legs share the
  pipeline body — it caught exactly that bug during review.
- repo suite: groot_n17 108 passed / 10 failed — the 10 are pre-existing
  Thor-sm110-only kernel failures on non-Thor builds (e.g.
  `rope_rotate_half_fp16_vec`), unchanged from baseline; frontend graph
  contract 11 passed; `test_generic_kernel_bindings.py` 6 passed.
- **RTX 5090 reproduction**: the full differential suite (ALL PASS) and
  every bit-equal stage gate reproduced — torch.equal on both GPUs,
  architecture-independent as claimed.
- **FP8 pipeline wiring**: the fp8-pipeline stage A/B runs with a GEMM
  shim (both legs share it, isolating the elementwise differential):
  full 16-layer stage — **torch.equal on both the 4090 and the 5090**.
  Two GEMM-primitive findings, both upstream and out of this PR's
  scope, recorded here for the maintainers:
  (a) on Ada, cuBLASLt FP8 requires bf16 output — `fp8_descale_fp16`
  (fp16-out) returns NOT_SUPPORTED; the SM89 pipeline's
  fp8_nt_dev + cast workaround exists for this;
  (b) on the 5090 with CUDA 12.8 (torch 2.11, cuBLASLt 91900),
  `fp8_descale_fp16`'s col-major NN layout combination is also
  NOT_SUPPORTED for every shape (while `torch._scaled_mm`'s TN
  combination works on the same machine) — the fused-epilogue paths
  the "SM120-safe" descale pattern was written to avoid may need a
  revisit, or the upstream 5090 environment runs CUDA 13.

Latency (median of 200; the fused-vs-legacy delta is the portable number):

| bench | RTX 4090 D | RTX 5090 |
|---|---|---|
| qk-norm + M-RoPE pair (S=257 / 1024 / 4096) | 1.46x / 2.20x / 3.49x | 1.32x / 1.50x / **4.24x** |
| bias_residual / bias_gelu site (S=1024) | 1.16x / 1.40x | — (same kernels) |
| LLM stage (16L, S=1024) | -3.3% | **-6.7%** |
| FP8-pipeline LLM stage (GEMM-shimmed, 4090) | -3.1% | real-GEMM run via the script |
| DiT stage (32L, Sa=41, incl. packed-QKV merge) | -7.3% | — |
| **synthetic e2e (Se=768)** | **-7.8%** (27.66 -> 25.51 ms) | **-10.3%** (22.34 -> 20.03 ms) |

e2e = ViT 24L + DS 3 + LLM 16L + VL 4L + DiT 32L x4 steps, real
frontend dims, stub attn, random weights. On the 4090 the delta is
stable -4.4% (ele fusions alone) to -7.8% (both) across Se=384..1152.
The 5090's larger share tracks its faster GEMMs raising the elementwise
fraction; the packed-QKV merged (41,4608,1536) GEMM shows NO sm120
scheduling penalty — the per-arch gating note below is moot.

## Notes for maintainers

- `qk_norm_rope_rotate_half_bf16.cu` and `bias_epilogue_bf16.cu` moved
  to unconditional compilation — plain elementwise, no arch-specific
  code; the Thor path consuming the bf16 entry keeps its semantics.
- The DiT QKV merge is opt-in via weight keys. The 5090 re-run
  confirmed no sm120 scheduling penalty for the merged
  (41,4608,1536) GEMM (e2e improved to -10.3%), so no arch gating is
  needed — the opt-in keys remain purely a consumer contract.
- Deferred (out of scope): fused-epilogue sites inside the fp8/NVFP4
  DiT tiers (they carry their own); SigLIP ViT QKV post-chain (different
  kernel contract); the Thor frontend's own wiring.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
