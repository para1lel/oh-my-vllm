#pragma once

#include "../../runtime/cuda/contracts.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

inline void residual_contract(TensorView tensor, TensorView reference, int rows) {
  bool fp32 = tensor.dtype().code == kDLFloat;
  tensor_contract(tensor, reference, fp32 ? kDLFloat : kDLBfloat, fp32 ? 32 : 16, {rows, 5120});
}

template <class Function> inline void dispatch_residual(TensorView tensor, Function function) {
  if (tensor.dtype().code == kDLFloat)
    function(static_cast<const float *>(tensor.data_ptr()));
  else
    function(static_cast<const __nv_bfloat16 *>(tensor.data_ptr()));
}

__device__ inline void load_float_eight(const __nv_bfloat16 *source, float (&values)[8]) {
  uint4 packed = *reinterpret_cast<const uint4 *>(source);
  uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
  for (int i = 0; i < 8; ++i)
    values[i] = __uint_as_float(((words[i / 2] >> ((i & 1) * 16)) & 0xffffu) << 16);
}

__device__ inline void load_float_eight(const float *source, float (&values)[8]) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    float4 packed = reinterpret_cast<const float4 *>(source)[i];
    values[i * 4] = packed.x;
    values[i * 4 + 1] = packed.y;
    values[i * 4 + 2] = packed.z;
    values[i * 4 + 3] = packed.w;
  }
}

__device__ inline float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_xor_sync(0xffffffff, value, offset);
  return value;
}

__device__ inline float warp_max(float value) {
  return __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(value)));
}

__device__ inline void quantize_four(float4 value, __nv_fp8_e4m3 *output, float *scale,
                                     float minimum = 1e-10f) {
  float maximum =
      warp_max(fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)), fmaxf(fabsf(value.z), fabsf(value.w))));
  float divisor = fmaxf(maximum, minimum) / 448.f;
  if ((threadIdx.x & 31) == 0)
    *scale = divisor;
  *reinterpret_cast<uint32_t *>(output) =
      __nv_fp8x4_e4m3(
          make_float4(value.x / divisor, value.y / divisor, value.z / divisor, value.w / divisor))
          .__x;
}
