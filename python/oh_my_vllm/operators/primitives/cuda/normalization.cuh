#pragma once

#include "common.cuh"
#include "../../runtime/cuda/dependent.cuh"

__device__ inline void store_normalization_eight(__nv_bfloat16 *destination,
                                                 const float (&values)[8]) {
  uint4 packed;
  auto *words = reinterpret_cast<uint32_t *>(&packed);
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    __nv_bfloat162 raw = __floats2bfloat162_rn(values[pair * 2], values[pair * 2 + 1]);
    words[pair] = *reinterpret_cast<const uint32_t *>(&raw);
  }
  *reinterpret_cast<uint4 *>(destination) = packed;
}

__device__ inline void store_normalization_eight(float *destination, const float (&values)[8]) {
#pragma unroll
  for (int i = 0; i < 2; ++i)
    reinterpret_cast<float4 *>(destination)[i] =
        make_float4(values[i * 4], values[i * 4 + 1], values[i * 4 + 2], values[i * 4 + 3]);
}

template <int Threads, bool Programmatic, class Residual, class Sum>
__global__ void residual_rms_quant_kernel(
    const __nv_bfloat16 *__restrict__ x, const Residual *__restrict__ residual,
    const float *__restrict__ weight, Sum *__restrict__ summed, __nv_fp8_e4m3 *__restrict__ data,
    float *__restrict__ scales, __nv_bfloat16 *__restrict__ normalized = nullptr) {
  if constexpr (Programmatic)
    launch_consumer();
  constexpr int Chunks = 5120 / (Threads * 8);
  int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  float values[Chunks][8], total = 0;
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    int column = chunk * Threads * 8 + threadIdx.x * 8;
    float input[8], skip[8];
    load_float_eight(x + row * 5120 + column, input);
    load_float_eight(residual + row * 5120 + column, skip);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float value = input[j] + skip[j];
      values[chunk][j] = value;
      total += values[chunk][j] * values[chunk][j];
    }
    store_normalization_eight(summed + row * 5120 + column, values[chunk]);
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
      store_normalization_eight(normalized + row * 5120 + column, values[chunk]);
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

template <bool Programmatic = false, class Residual, class Sum>
inline void residual_rms_quant_launch(const __nv_bfloat16 *x, const Residual *residual,
                                      const float *norm, Sum *summed, __nv_fp8_e4m3 *quantized,
                                      float *scales, int rows, cudaStream_t stream,
                                      __nv_bfloat16 *normalized = nullptr) {
  if (rows <= 32)
    residual_rms_quant_kernel<640, Programmatic>
        <<<rows, 640, 0, stream>>>(x, residual, norm, summed, quantized, scales, normalized);
  else
    residual_rms_quant_kernel<320, false>
        <<<rows, 320, 0, stream>>>(x, residual, norm, summed, quantized, scales, normalized);
  check_cuda(cudaGetLastError());
}
