#pragma once

#include "common.cuh"
#include "launch.cuh"

template <int Threads>
__global__ void
residual_rms_quant_kernel(const __nv_bfloat16 *__restrict__ x,
                          const __nv_bfloat16 *__restrict__ residual,
                          const float *__restrict__ weight, __nv_bfloat16 *__restrict__ summed,
                          __nv_fp8_e4m3 *__restrict__ data, float *__restrict__ scales,
                          __nv_bfloat16 *__restrict__ normalized = nullptr) {
  constexpr int Chunks = 5120 / (Threads * 8);
  int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
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
      summed[row * 5120 + column + j] = __float2bfloat16_rn(value);
      values[chunk][j] = value;
      total += values[chunk][j] * values[chunk][j];
    }
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
      if (normalized)
        normalized[row * 5120 + column + j] = __float2bfloat16_rn(values[chunk][j]);
      maximum = fmaxf(maximum, fabsf(values[chunk][j]));
    }
    unsigned mask = lane < 16 ? 0xffffu : 0xffff0000u;
    maximum = __uint_as_float(__reduce_max_sync(mask, __float_as_uint(maximum)));
    float divisor = fmaxf(maximum, 1e-10f) / 448.f;
    if (lane % 16 == 0)
      scales[row * 40 + column / 128] = divisor;
#pragma unroll
    for (int j = 0; j < 8; j += 4)
      *reinterpret_cast<uint32_t *>(data + row * 5120 + column + j) =
          __nv_fp8x4_e4m3(make_float4(values[chunk][j] / divisor, values[chunk][j + 1] / divisor,
                                      values[chunk][j + 2] / divisor,
                                      values[chunk][j + 3] / divisor))
              .__x;
  }
}

inline void residual_rms_quant_launch(const __nv_bfloat16 *x, const __nv_bfloat16 *residual,
                                      const float *norm, __nv_bfloat16 *summed,
                                      __nv_fp8_e4m3 *quantized, float *scales, int rows,
                                      cudaStream_t stream, __nv_bfloat16 *normalized = nullptr) {
  if (rows <= 32)
    residual_rms_quant_kernel<640>
        <<<rows, 640, 0, stream>>>(x, residual, norm, summed, quantized, scales, normalized);
  else
    residual_rms_quant_kernel<320>
        <<<rows, 320, 0, stream>>>(x, residual, norm, summed, quantized, scales, normalized);
  check_cuda(cudaGetLastError());
}
