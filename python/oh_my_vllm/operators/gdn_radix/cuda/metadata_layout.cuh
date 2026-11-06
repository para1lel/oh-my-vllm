#pragma once

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

namespace gdn_radix_metadata {

__host__ __device__ inline int chunk_count(const int *writes, int first, int last) {
  int chunks = 0;
  for (int begin = first; begin < last; ++chunks) {
    int end = begin + 64 < last ? begin + 64 : last;
    for (int row = begin; row < end; ++row) {
      if (writes[row] >= 0) {
        end = row + 1;
        break;
      }
    }
    begin = end;
  }
  return chunks;
}

__host__ __device__ inline int layout_chunks(const int *writes, const int *starts, int rows,
                                             int batch) {
  if (starts[0] != 0 || starts[batch] != rows)
    return -1;
  int chunks = 0;
  for (int sequence = 0; sequence < batch; ++sequence) {
    int first = starts[sequence], last = starts[sequence + 1];
    if (first < 0 || first > last || last > rows)
      return -1;
    chunks += chunk_count(writes, first, last);
  }
  return chunks;
}

__host__ __device__ inline bool build_sequence(const int *writes, int first, int last,
                                               int *metadata, int base, int capacity) {
  int count = chunk_count(writes, first, last);
  if (base < 0 || count > capacity - base)
    return false;
  int chunk = base;
  for (int begin = first; begin < last; ++chunk) {
    int end = begin + 64 < last ? begin + 64 : last;
    for (int row = begin; row < end; ++row) {
      if (writes[row] >= 0) {
        end = row + 1;
        break;
      }
    }
    metadata[chunk * 4] = begin;
    metadata[chunk * 4 + 1] = end;
    metadata[chunk * 4 + 2] = writes[end - 1];
    metadata[chunk * 4 + 3] = chunk == base ? count : 0;
    begin = end;
  }
  return true;
}

} // namespace gdn_radix_metadata

#ifndef __CUDACC__
#undef __host__
#undef __device__
#endif
