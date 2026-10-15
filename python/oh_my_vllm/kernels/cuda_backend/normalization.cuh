#pragma once

#include "common.cuh"

template <int Threads>
__global__ void residual_rms_quant_kernel(const __nv_bfloat16 *x, const __nv_bfloat16 *residual,
                                          const float *weight, __nv_bfloat16 *summed,
                                          __nv_fp8_e4m3 *data, float *scales,
                                          __nv_bfloat16 *normalized = nullptr) {
  constexpr int Chunks = 5120 / (Threads * 8);
  int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  float values[Chunks][8], total = 0;
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    int column = chunk * Threads * 8 + threadIdx.x * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float value = __bfloat162float(x[row * 5120 + column + j]) +
                    __bfloat162float(residual[row * 5120 + column + j]);
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
    float maximum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      values[chunk][j] = __bfloat162float(
          __float2bfloat16_rn(values[chunk][j] * inverse * (weight[column + j] + 1.f)));
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
