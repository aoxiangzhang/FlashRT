// Declarations for the fused bf16 bias epilogues (bias_epilogue_bf16.cu).
// The TU is shared: it compiles into the main flash_rt_kernels module and
// into the FLASHRT_BUILD_QWEN3_VL module (which historically declared
// these inline).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

void residual_add_bias_bf16(
    __nv_bfloat16* residual, const __nv_bfloat16* x,
    const __nv_bfloat16* bias, int rows, int dim, cudaStream_t stream);

void qkv_split_bias_bf16(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias, __nv_bfloat16* q,
    __nv_bfloat16* k, __nv_bfloat16* v, int rows, int hq, int hk, int hv,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
