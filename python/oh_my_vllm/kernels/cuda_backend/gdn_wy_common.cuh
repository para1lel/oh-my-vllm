#pragma once

#include "gdn_kernels.cuh"

namespace gdn_wy {

constexpr int Block = 64, Width = 128, Threads = 256, Parts = 1;
constexpr int TokenStride = 72, ProductStride = 65;

__device__ inline int64_t transform_offset(int chunk, int head, int part, int kind) {
  return (((int64_t(chunk) * 48 + head) * Parts + part) * 2 + kind) * Block * Width;
}

__global__ void metadata_kernel(const int *starts, const int *reads, const int *writes,
                                int *metadata, int capacity, int rows, int slots) {
  int sequence = blockIdx.x, lane = threadIdx.x;
  int first = starts[sequence], last = starts[sequence + 1], source = reads[sequence];
  gdn_require(first >= 0 && first <= last && last <= rows && source >= 0 && source < slots);
  int base = sequence * capacity;
  for (int index = lane; index < capacity * 4; index += 32)
    metadata[base * 4 + index] = 0;
  __syncwarp();
  int count = 0;
  for (int begin = first; begin < last;) {
    gdn_require(count < capacity);
    int target0 = begin + lane < last ? writes[begin + lane] : -1;
    int target1 = begin + lane + 32 < last ? writes[begin + lane + 32] : -1;
    gdn_require(target0 >= -1 && target0 < slots && target0 != source);
    gdn_require(target1 >= -1 && target1 < slots && target1 != source);
    unsigned mask0 = __ballot_sync(0xffffffffu, target0 >= 0);
    unsigned mask1 = __ballot_sync(0xffffffffu, target1 >= 0);
    int length = min(Block, last - begin);
    if (mask0)
      length = __ffs(mask0);
    else if (mask1)
      length = 32 + __ffs(mask1);
    int target = __shfl_sync(0xffffffffu, length <= 32 ? target0 : target1, (length - 1) % 32);
    if (lane == 0) {
      metadata[(base + count) * 4] = begin;
      metadata[(base + count) * 4 + 1] = begin + length;
      metadata[(base + count) * 4 + 2] = target;
    }
    begin += length;
    ++count;
  }
  if (lane == 0)
    metadata[base * 4 + 3] = count;
}

template <class Storage> __device__ inline void inverse_blocks(Storage &s) {
  for (int index = threadIdx.x; index < Block * Block; index += Threads) {
    int row = index / Block, column = index % Block;
    s.inverse[row * TokenStride + column] = row == column ? 1.f : 0.f;
  }
  __syncthreads();
  // Invert the four diagonal blocks, then combine the two block levels.
  if (threadIdx.x < 64) {
    int base = threadIdx.x / 16 * 16, column = threadIdx.x % 16;
    float solved[16];
#pragma unroll
    for (int row = 0; row < 16; ++row)
      solved[row] = row == column ? 1.f : 0.f;
#pragma unroll
    for (int row = 1; row < 16; ++row) {
      float value = 0.f;
      if (column < row) {
#pragma unroll
        for (int previous = 0; previous < row; ++previous)
          value = fmaf(-s.products[(base + row) * ProductStride + base + previous],
                       solved[previous], value);
        solved[row] = value;
      }
    }
#pragma unroll
    for (int row = 0; row < 16; ++row)
      s.inverse[(base + row) * TokenStride + base + column] = solved[row];
  }
  __syncthreads();
  for (int index = threadIdx.x; index < 2 * 16 * 16; index += Threads) {
    int block = index / 256, row = index / 16 % 16, column = index % 16;
    int row_base = block * 32 + 16, column_base = block * 32;
    float value = 0.f;
#pragma unroll
    for (int inner_base = 0; inner_base < 16; inner_base += 8)
      if (row >= inner_base) {
#pragma unroll
        for (int inner = inner_base; inner < inner_base + 8; ++inner)
          value =
              fmaf(-s.inverse[(row_base + row) * TokenStride + row_base + inner],
                   s.products[(row_base + inner) * ProductStride + column_base + column], value);
      }
    s.temporary[index] = value;
  }
  __syncthreads();
  for (int index = threadIdx.x; index < 2 * 16 * 16; index += Threads) {
    int block = index / 256, row = index / 16 % 16, column = index % 16;
    int row_base = block * 32 + 16, column_base = block * 32;
    float value = 0.f;
#pragma unroll
    for (int inner = 0; inner < 16; ++inner)
      value = fmaf(s.temporary[block * 256 + row * 16 + inner],
                   s.inverse[(column_base + inner) * TokenStride + column_base + column], value);
    s.inverse[(row_base + row) * TokenStride + column_base + column] = value;
  }
  __syncthreads();
  int partial_row = threadIdx.x / 32, partial_column = threadIdx.x % 32;
  float partials[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int inner_base = 0; inner_base < 32; inner_base += 8) {
#pragma unroll
    for (int inner = inner_base; inner < inner_base + 8; ++inner) {
      float right = s.products[(32 + inner) * ProductStride + partial_column];
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        int row = partial_row + part * 8;
        if (row >= inner_base)
          partials[part] =
              fmaf(-s.inverse[(32 + row) * TokenStride + 32 + inner], right, partials[part]);
      }
    }
  }
#pragma unroll
  for (int part = 0; part < 4; ++part)
    s.temporary[(partial_row + part * 8) * 32 + partial_column] = partials[part];
  __syncthreads();
  int first_row = threadIdx.x / 32, column = threadIdx.x % 32;
  float values[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int inner = 0; inner < 32; ++inner) {
    float right = s.inverse[inner * TokenStride + column];
#pragma unroll
    for (int part = 0; part < 4; ++part)
      values[part] = fmaf(s.temporary[(first_row + part * 8) * 32 + inner], right, values[part]);
  }
#pragma unroll
  for (int part = 0; part < 4; ++part)
    s.inverse[(32 + first_row + part * 8) * TokenStride + column] = values[part];
}

} // namespace gdn_wy
