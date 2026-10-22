#pragma once
#include "gdn_wy_common.cuh"

namespace gdn_metadata_parallel {
__global__ void initialize(const int *starts, const int *reads, const int *writes, int *metadata,
                           int *offsets, int *phase, int total_chunks, int max_capacity, int rows,
                           int slots) {
  int sequence = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  int first = starts[sequence], last = starts[sequence + 1], source = reads[sequence];
  gdn_require(first >= 0 && first <= last && last <= rows && source >= 0 && source < slots);
  int capacity = gdn_wy::chunk_capacity(last - first);
  int base =
      gdn_wy::sequence_chunk_base(starts, sequence, gridDim.x, rows, total_chunks, max_capacity);
  if (tid == 0) {
    offsets[sequence] = base;
    if (sequence + 1 == gridDim.x)
      offsets[sequence + 1] = base + capacity;
  }
  if (!capacity)
    return;
  for (int i = tid; i < capacity * 4; i += 256)
    metadata[base * 4 + i] = 0;
  int earliest = last, limit = min(first + 784, last);
  for (int i = first + tid; i < limit; i += 256) {
    int target = writes[i];
    gdn_require(target >= -1 && target < slots && target != source);
    if (target >= 0)
      earliest = min(earliest, i);
  }
  __shared__ int minima[8];
  int minimum = __reduce_min_sync(0xffffffffu, earliest);
  if (lane == 0)
    minima[warp] = minimum;
  __syncthreads();
  minimum = lane < 8 ? minima[lane] : last;
  earliest = __reduce_min_sync(0xffffffffu, minimum);
  int initial = 0, chunks = 0;
  if (last > first) {
    gdn_require(earliest < limit);
    initial = earliest - first + 1;
    int remaining = last - first - initial;
    chunks = (initial + 63) / 64 + (remaining / 784) * 13 + (remaining % 784 + 63) / 64;
    gdn_require(chunks <= capacity);
  }
  if (tid == 0) {
    phase[sequence] = initial;
    metadata[base * 4 + 3] = chunks;
  }
}

__global__ void build(const int *starts, const int *reads, const int *writes, int *metadata,
                      const int *offsets, const int *phase, int rows, int slots) {
  int sequence = blockIdx.x, chunk = blockIdx.y * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  int base = offsets[sequence], capacity = offsets[sequence + 1] - base;
  if (chunk >= capacity)
    return;
  int count = metadata[base * 4 + 3];
  if (chunk >= count)
    return;
  int first = starts[sequence], last = starts[sequence + 1], source = reads[sequence],
      initial = phase[sequence];
  int initial_chunks = (initial + 63) / 64, begin, end;
  if (chunk < initial_chunks) {
    begin = first + chunk * 64;
    end = min(begin + 64, first + initial);
  } else {
    int after = chunk - initial_chunks, interval = after / 13, within = after % 13;
    begin = first + initial + interval * 784 + within * 64;
    end = min(min(begin + 64, first + initial + (interval + 1) * 784), last);
  }
  gdn_require(first >= 0 && first <= begin && begin < end && end <= last && last <= rows &&
              source >= 0 && source < slots);
  bool exported = end == last || end == first + initial || (end - first - initial) % 784 == 0;
  for (int i = begin + lane; i < end; i += 32) {
    int target = writes[i];
    gdn_require(target >= -1 && target < slots && target != source);
    gdn_require(i == end - 1 && exported ? target >= 0 : target == -1);
  }
  if (lane == 0) {
    metadata[(base + chunk) * 4] = begin;
    metadata[(base + chunk) * 4 + 1] = end;
    metadata[(base + chunk) * 4 + 2] = writes[end - 1];
  }
}

inline void launch(const int *starts, const int *reads, const int *writes, int *metadata,
                   int *offsets, __nv_bfloat16 *snapshots, int capacity, int chunks, int rows,
                   int slots, int batch, cudaStream_t stream) {
  // Carry replaces this scratch before output reads the snapshots.
  auto *phase = reinterpret_cast<int *>(snapshots);
  initialize<<<batch, 256, 0, stream>>>(starts, reads, writes, metadata, offsets, phase, chunks,
                                        capacity, rows, slots);
  check_cuda(cudaGetLastError());
  build<<<dim3(batch, (capacity + 3) / 4), 128, 0, stream>>>(starts, reads, writes, metadata,
                                                             offsets, phase, rows, slots);
  check_cuda(cudaGetLastError());
}
} // namespace gdn_metadata_parallel
