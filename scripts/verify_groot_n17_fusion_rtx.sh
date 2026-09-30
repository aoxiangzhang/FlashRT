#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# verify_groot_n17_fusion_rtx.sh — one-shot verification for the GROOT
# N1.7 elementwise-fusion PRs (PR-1: strict ele fusions; PR-2: packed-QKV
# merge) on any RTX-class machine. Developed + verified on RTX 4090 D
# (sm89); intended for a 5090 (sm120) re-run before merging.
#
# What it does:
#   1. Builds flash_rt_kernels for the local GPU arch (with FA2).
#   2. Runs the differential test suite — every fusion is compared
#      bit-for-bit (torch.equal) against the legacy kernel chain it
#      replaces, plus golden frozen-chain tests. Arch-independent gates.
#   3. Runs the existing groot_n17 test files (fixture tests skip
#      without a checkpoint; Thor-sm110-only kernel tests fail on any
#      non-Thor build — pre-existing, expected).
#   4. Runs both latency benches (kernel/stage + synthetic e2e). The
#      fused-vs-legacy DELTA is the portable number; absolute ms will
#      differ per GPU.
#
# Prereqs:
#   - CUDA toolkit >= what this GPU needs (sm120 -> CUDA 12.8+)
#   - cmake >= 3.24 on PATH (pip install cmake works)
#   - CUTLASS headers: git clone --depth 1 --branch v4.4.2 \
#         https://github.com/NVIDIA/cutlass.git third_party/cutlass
#   - a python with torch (cuda build) + pytest, passed via $PY (default:
#     python). NOTE: the module .so must match this python's minor
#     version (the build writes flash_rt_kernels.cpython-3XX.so).
#
# Usage:
#   PY=/path/to/python ./scripts/verify_groot_n17_fusion_rtx.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

PY="${PY:-python}"
cd "$(dirname "$0")/.."
REPO=$(pwd)

echo "══ [1/4] GPU / toolchain ══════════════════════════════════════════"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
CC_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "compute cap: ${CC_CAP}"
cmake --version | head -1
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "══ [2/4] Build flash_rt_kernels (arch ${CC_CAP}) + fa2 ════════════"
if [ ! -d third_party/cutlass/include ]; then
    echo "FATAL: third_party/cutlass missing (see header comment)" >&2
    exit 1
fi
cmake -B build -S . -DGPU_ARCH="${CC_CAP}" -DFA2_ARCH_NATIVE_ONLY=ON \
      -DCMAKE_CUDA_COMPILER="${CUDACXX:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}" \
      -DPython3_EXECUTABLE="$(command -v $PY)"
cmake --build build --target flash_rt_kernels -j"$(nproc)"

echo "══ [3/4] Differential correctness ═════════════════════════════════"
# The single source of truth for PR-1/PR-2 correctness: every test in
# this file gates at torch.equal (bit-identical) or an explicit ulp
# bound vs a fp64 reference. Prints per-test PASS lines.
$PY tests/test_groot_n17_sm89_ele_fusion.py

# Existing repo tests: groot_n17 non-fixture unit/contract tests.
# Expected on a non-Thor build: a stable number of FAILs from
# test_groot_n17_thor_fp4_kernels.py (Thor sm110-only kernels not
# compiled) — they gate the CI Thor build, not this PR.
$PY -m pytest tests/ -k groot_n17 -q --no-header \
    --ignore=tests/test_groot_n17_thor_fp4_kernels.py \
    --ignore=tests/test_fp4_integration_meta.py \
    --ignore=tests/test_pi05_cuda_graph_modes.py \
    --ignore=tests/test_pi05_decoder_fp4_kernels.py \
    --ignore=tests/test_pi05_fp4_fusion_kernels.py \
    || echo "(note failures above — compare against the pre-PR baseline)"

echo "══ [4/4] Latency (fused vs legacy, same machine) ═════════════════"
$PY benchmarks/groot_n17_sm89_ele_fusion_bench.py
$PY benchmarks/groot_n17_sm89_e2e_bench.py

echo "══ Summary ════════════════════════════════════════════════════════"
echo "Reference (RTX 4090 D, torch 2.13.0+cu130):"
echo "  ele_fusion bench: qk-norm+RoPE 1.46x/2.20x/3.49x (S=257/1024/4096);"
echo "    bias_residual 1.16x, bias_gelu 1.40x; LLM stage -3.3%"
echo "  e2e bench (Se=768): PR-1 -4.5%; PR-1+PR-2 cumulative -7.8%"
echo "  (27.66 -> 25.51 ms on the 4090)"
echo "On this GPU, compare the DELTAS above, not absolute ms. The one"
echo "arch-dependent variable is the cuBLASLt schedule of the merged"
echo "(41,4608,1536) GEMM vs 3x (41,1536,1536) — if the e2e 'dit' share"
echo "regresses on sm120, PR-2's opt-in keys can be gated per-arch."
