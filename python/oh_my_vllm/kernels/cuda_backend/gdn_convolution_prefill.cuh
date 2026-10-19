#pragma once
#include "gdn_kernels.cuh"

template <int Rows>
__global__ void gdn_convolution_tiled_kernel(
    const __nv_bfloat16 *__restrict__ projection, const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ pool, const int *__restrict__ starts, const int *__restrict__ reads,
    const int *__restrict__ writes, __nv_bfloat16 *__restrict__ output, int rows, int batch,
    int slots, bool verify) {
  int sequence = blockIdx.z, lane = threadIdx.x % 32;
  int group = blockIdx.y * 4 + threadIdx.x / 32;
  int first = starts[sequence], last = starts[sequence + 1];
  int begin = first + blockIdx.x * Rows, source = reads[sequence];
  gdn_require(starts[0] == 0 && starts[batch] == rows);
  gdn_require(first >= 0 && first <= last && last <= rows);
  gdn_require(!verify || last - first <= 5);
  if (begin >= last)
    return;
  gdn_require(source >= 0 && source < slots);
  int channel = group * 128 + lane * 4;
  uint4 packed_weights[2] = {*reinterpret_cast<const uint4 *>(weight + channel * 4),
                             *reinterpret_cast<const uint4 *>(weight + channel * 4 + 8)};
  float weights[4][4];
#pragma unroll
  for (int column = 0; column < 4; ++column)
#pragma unroll
    for (int tap = 0; tap < 4; ++tap)
      weights[column][tap] = __bfloat162float(
          reinterpret_cast<const __nv_bfloat16 *>(packed_weights)[column * 4 + tap]);
  float history[3][4];
#pragma unroll
  for (int tap = 0; tap < 3; ++tap) {
    int position = begin + tap - 3;
    if (position >= first) {
      uint2 packed =
          *reinterpret_cast<const uint2 *>(projection + int64_t(position) * 16384 + channel);
#pragma unroll
      for (int column = 0; column < 4; ++column)
        history[tap][column] =
            __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(&packed)[column]);
    } else {
#pragma unroll
      for (int column = 0; column < 4; ++column)
        history[tap][column] = __bfloat162float(
            pool[(int64_t(source) * 10240 + channel + column) * 3 + position - first + 3]);
    }
  }
#pragma unroll 1
  for (int local = 0; local < Rows && begin + local < last; ++local) {
    int row = begin + local, target = writes[row];
    gdn_require(target >= -1 && target < slots);
    if (target >= 0)
      for (int request = 0; request < batch; ++request)
        gdn_require(target != reads[request]);
    uint2 packed = *reinterpret_cast<const uint2 *>(projection + int64_t(row) * 16384 + channel);
    float current[4], values[4], square = 0.f;
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      current[column] = __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(&packed)[column]);
      float sum = 0.f;
#pragma unroll
      for (int tap = 0; tap < 3; ++tap)
        sum += history[tap][column] * weights[column][tap];
      sum += current[column] * weights[column][3];
      values[column] = __bfloat162float(__float2bfloat16_rn(sum * gdn_sigmoid(sum)));
      square += values[column] * values[column];
    }
    float inverse = !verify && group < 32 ? rsqrtf(warp_sum(square) + 1e-6f) : 1.f;
    __nv_bfloat16 result[4];
#pragma unroll
    for (int column = 0; column < 4; ++column)
      result[column] = __float2bfloat16_rn(values[column] * inverse);
    *reinterpret_cast<uint2 *>(output + int64_t(row) * 10240 + channel) =
        *reinterpret_cast<const uint2 *>(result);
    if (target >= 0) {
      __nv_bfloat16 checkpoint[12];
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        checkpoint[column * 3] = __float2bfloat16_rn(history[1][column]);
        checkpoint[column * 3 + 1] = __float2bfloat16_rn(history[2][column]);
        checkpoint[column * 3 + 2] = __float2bfloat16_rn(current[column]);
      }
#pragma unroll
      for (int vector = 0; vector < 3; ++vector)
        *reinterpret_cast<uint2 *>(pool + (int64_t(target) * 10240 + channel) * 3 + vector * 4) =
            reinterpret_cast<const uint2 *>(checkpoint)[vector];
    }
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      history[0][column] = history[1][column];
      history[1][column] = history[2][column];
      history[2][column] = current[column];
    }
  }
}
