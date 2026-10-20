#pragma once

__global__ void vision_residual_norm_kernel(const float *projected,
                                            const __nv_bfloat16 *residual_bias,
                                            const __nv_bfloat16 *value, const __nv_bfloat16 *weight,
                                            const __nv_bfloat16 *bias, __nv_bfloat16 *output,
                                            __nv_bfloat16 *normalized_output) {
  int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  __shared__ float partial[8];
  float cached[5];
  float sum = 0;
#pragma unroll
  for (int slot = 0; slot < 5; ++slot) {
    int column = threadIdx.x + 256 * slot;
    if (column < 1152) {
      int64_t index = int64_t(row) * 1152 + column;
      __nv_bfloat16 stored = __float2bfloat16_rn(
          vision_biased(projected, residual_bias, index, column) + __bfloat162float(value[index]));
      output[index] = stored;
      cached[slot] = __bfloat162float(stored);
      sum += cached[slot];
    }
  }
  sum = warp_sum(sum);
  if (lane == 0)
    partial[warp] = sum;
  __syncthreads();
  float mean = warp_sum(lane < 8 ? partial[lane] : 0) / 1152.f;
  float variance = 0;
#pragma unroll
  for (int slot = 0; slot < 5; ++slot) {
    int column = threadIdx.x + 256 * slot;
    if (column < 1152) {
      float centered = cached[slot] - mean;
      variance += centered * centered;
    }
  }
  variance = warp_sum(variance);
  __syncthreads();
  if (lane == 0)
    partial[warp] = variance;
  __syncthreads();
  float inverse = rsqrtf(warp_sum(lane < 8 ? partial[lane] : 0) / 1152.f + 1e-6f);
#pragma unroll
  for (int slot = 0; slot < 5; ++slot) {
    int column = threadIdx.x + 256 * slot;
    if (column < 1152) {
      float normalized = (cached[slot] - mean) * inverse;
      normalized_output[int64_t(row) * 1152 + column] = __float2bfloat16_rn(
          normalized * __bfloat162float(weight[column]) + __bfloat162float(bias[column]));
    }
  }
}
