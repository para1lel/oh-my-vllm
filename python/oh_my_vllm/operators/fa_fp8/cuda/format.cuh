#pragma once

#include "../../primitives/cuda/common.cuh"

namespace fa_fp8 {
constexpr int PageRows = 128;
constexpr int Heads = 4;
constexpr int Width = 256;

__host__ __device__ inline int64_t scale_index(int page, int plane, int token, int head) {
  return ((int64_t(page) * 2 + plane) * PageRows + token) * Heads + head;
}

__host__ __device__ inline int64_t code_index(int page, int plane, int token, int head, int dim) {
  return scale_index(page, plane, token, head) * Width + dim;
}

__device__ inline float bf16(float value) { return __bfloat162float(__float2bfloat16_rn(value)); }

__device__ inline void encode_vector(float (&values)[8], uint8_t *codes, float *scale) {
  float maximum = 0;
  bool finite = true;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    finite &= isfinite(values[j]);
    maximum = fmaxf(maximum, fabsf(values[j]));
  }
  if (__any_sync(0xffffffff, !finite)) {
    asm volatile("trap;");
    return;
  }
  maximum = warp_max(maximum);
  float divisor = maximum == 0.f ? 1.f : fmaxf(__fdiv_rn(maximum, 448.f), __uint_as_float(1));
  if ((threadIdx.x & 31) == 0)
    *scale = divisor;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    codes[(threadIdx.x & 31) + j * 32] =
        __nv_cvt_float_to_fp8(__fdiv_rn(values[j], divisor), __NV_SATFINITE, __NV_E4M3);
}

__device__ inline __nv_bfloat16 decode(uint8_t code, float scale) {
  if ((code & 127) == 127 || !(scale > 0.f) || !isfinite(scale)) {
    asm volatile("trap;");
    return __float2bfloat16_rn(0.f);
  }
  __nv_fp8_e4m3 value;
  value.__x = code;
  float product = __fmul_rn(float(value), scale);
  __nv_bfloat16 result = __float2bfloat16_rn(product);
  if (!isfinite(product) || !isfinite(__bfloat162float(result)))
    asm volatile("trap;");
  return result;
}
} // namespace fa_fp8
