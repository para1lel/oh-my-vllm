#pragma once

#include "../recurrent/gdn_kernels.cuh"

namespace gdn_wy {

constexpr int Block = 64, Width = 128, Threads = 256, Parts = 1;
constexpr int TokenStride = 72, ProductStride = 65;

__host__ __device__ inline int chunk_capacity(int queries) {
  return queries ? (queries + 63) / 64 + (queries + 783) / 784 + 1 : 0;
}

__device__ inline int sequence_chunk_base(const int *starts, int sequence, int batch, int rows,
                                          int chunks, int max_capacity) {
  gdn_require(starts[0] == 0 && starts[batch] == rows);
  int base = 0, end = 0;
  for (int index = 0; index <= sequence; ++index) {
    int first = starts[index], last = starts[index + 1];
    gdn_require(first >= 0 && first <= last && last <= rows);
    int capacity = chunk_capacity(last - first);
    gdn_require(capacity <= max_capacity);
    base = end;
    end += capacity;
    gdn_require(end <= chunks);
  }
  gdn_require(sequence + 1 < batch || end == chunks);
  return base;
}

__device__ inline int64_t transform_offset(int chunk, int head, int part, int kind) {
  return (((int64_t(chunk) * 48 + head) * Parts + part) * 2 + kind) * Block * Width;
}

__global__ void metadata_kernel(const int *starts, const int *reads, const int *writes,
                                int *metadata, int *offsets, int chunks, int max_capacity, int rows,
                                int slots) {
  int sequence = blockIdx.x, lane = threadIdx.x;
  int first = starts[sequence], last = starts[sequence + 1], source = reads[sequence];
  gdn_require(first >= 0 && first <= last && last <= rows && source >= 0 && source < slots);
  int capacity = chunk_capacity(last - first);
  int base = sequence_chunk_base(starts, sequence, gridDim.x, rows, chunks, max_capacity);
  if (lane == 0) {
    offsets[sequence] = base;
    if (sequence + 1 == gridDim.x)
      offsets[sequence + 1] = base + capacity;
  }
  if (!capacity)
    return;
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

} // namespace gdn_wy
