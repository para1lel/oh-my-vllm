#pragma once
#include "normalization.cuh"

namespace ffn_mx_native {
template <int Threads>
__global__ void
norm_mx_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ residual,
               const float *__restrict__ weight, __nv_bfloat16 *__restrict__ summed,
               __nv_bfloat16 *__restrict__ normalized, __nv_fp8_e4m3 *__restrict__ prepared,
               unsigned char *__restrict__ prepared_scales, int rows) {
  constexpr int Chunks = 5120 / (Threads * 8);
  int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  if (row >= rows) {
    for (int chunk = 0; chunk < Chunks; ++chunk) {
      int column = chunk * Threads * 8 + threadIdx.x * 8;
      *reinterpret_cast<uint2 *>(prepared + int64_t(row) * 5120 + column) = make_uint2(0, 0);
      if (lane % 16 == 0) {
        int64_t tile = int64_t(row / 128) * 40 + column / 128;
        int64_t offset = tile * 512 + (row % 32) * 16 + (row % 128 / 32) * 4;
        *reinterpret_cast<uint32_t *>(prepared_scales + offset) = 0;
      }
    }
    return;
  }
  float values[Chunks][8], total = 0;
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    int column = chunk * Threads * 8 + threadIdx.x * 8;
    uint4 input = *reinterpret_cast<const uint4 *>(x + row * 5120 + column);
    uint4 skip = *reinterpret_cast<const uint4 *>(residual + row * 5120 + column);
    uint32_t input_words[4] = {input.x, input.y, input.z, input.w};
    uint32_t skip_words[4] = {skip.x, skip.y, skip.z, skip.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int shift = (j & 1) * 16;
      float value = __uint_as_float(((input_words[j / 2] >> shift) & 0xffffu) << 16) +
                    __uint_as_float(((skip_words[j / 2] >> shift) & 0xffffu) << 16);
      values[chunk][j] = value;
      total += values[chunk][j] * values[chunk][j];
    }
    store_bf16_eight(summed + row * 5120 + column, values[chunk]);
  }
  __shared__ float partial[Threads / 32];
  total = warp_sum(total);
  if (lane == 0)
    partial[warp] = total;
  __syncthreads();
  total = warp_sum(lane < Threads / 32 ? partial[lane] : 0);
  float inverse = rsqrtf(total / 5120.f + 1e-6f);
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    int column = chunk * Threads * 8 + threadIdx.x * 8;
    float4 weight_low = *reinterpret_cast<const float4 *>(weight + column);
    float4 weight_high = *reinterpret_cast<const float4 *>(weight + column + 4);
    float weights[8] = {weight_low.x,  weight_low.y,  weight_low.z,  weight_low.w,
                        weight_high.x, weight_high.y, weight_high.z, weight_high.w};
    float maximum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      values[chunk][j] =
          __bfloat162float(__float2bfloat16_rn(values[chunk][j] * inverse * (weights[j] + 1.f)));
      maximum = fmaxf(maximum, fabsf(values[chunk][j]));
    }
    if (normalized)
      store_bf16_eight(normalized + row * 5120 + column, values[chunk]);
    unsigned mask = lane < 16 ? 0xffffu : 0xffff0000u;
    maximum = __uint_as_float(__reduce_max_sync(mask, __float_as_uint(maximum)));
    float divisor = fmaxf(maximum, 1e-10f) / 448.f;
    if (!(divisor >= 0.f && isfinite(divisor)))
      asm volatile("trap;");
    unsigned bits = __float_as_uint(divisor);
    unsigned exponent = (bits >> 23) & 255u;
    exponent += (bits & 0x7fffffu) != 0u && exponent < 254u;
    float inverse_scale = 1.f / __uint_as_float(exponent << 23);
    if (lane % 16 == 0) {
      int64_t tile = int64_t(row / 128) * 40 + column / 128;
      int64_t offset = tile * 512 + (row % 32) * 16 + (row % 128 / 32) * 4;
      *reinterpret_cast<uint32_t *>(prepared_scales + offset) = exponent * 0x01010101u;
    }
#pragma unroll
    for (int j = 0; j < 8; j += 4)
      *reinterpret_cast<uint32_t *>(prepared + int64_t(row) * 5120 + column + j) =
          __nv_fp8x4_e4m3(make_float4(values[chunk][j] * inverse_scale,
                                      values[chunk][j + 1] * inverse_scale,
                                      values[chunk][j + 2] * inverse_scale,
                                      values[chunk][j + 3] * inverse_scale))
              .__x;
  }
}

} // namespace ffn_mx_native
