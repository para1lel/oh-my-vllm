#pragma once

#include "metadata_layout.cuh"
#include "../../gdn/cuda/recurrent/gdn_kernels.cuh"

namespace gdn_radix_metadata {

__global__ void initialize_owners(int *owners, int slots) {
  for (int slot = blockIdx.x * blockDim.x + threadIdx.x; slot < slots;
       slot += blockDim.x * gridDim.x)
    owners[slot] = -1;
}

__global__ void validate_sequences(const int *starts, const int *reads, int *owners, int rows,
                                   int batch, int slots, int max_queries, bool verify) {
  int sequence = blockIdx.x * blockDim.x + threadIdx.x;
  if (sequence >= batch)
    return;
  gdn_require(starts[0] == 0 && starts[batch] == rows);
  int first = starts[sequence], last = starts[sequence + 1], source = reads[sequence];
  gdn_require(first >= 0 && first <= last && last <= rows && last - first <= max_queries);
  gdn_require(!verify || last - first <= 5);
  gdn_require(source >= 0 && source < slots);
  atomicExch(owners + source, -2);
}

__global__ void validate_exports(const int *writes, int *owners, int rows, int slots, bool verify) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  int target = writes[row];
  gdn_require(target >= -1 && target < slots && (!verify || target >= 0));
  if (target >= 0)
    gdn_require(atomicCAS(owners + target, -1, row) == -1);
}

__global__ void validate_capacity(const int *starts, const int *writes, int rows, int batch,
                                  int capacity) {
  int count = layout_chunks(writes, starts, rows, batch);
  gdn_require(count >= 0 && count <= capacity);
}

inline void validate(const int *starts, const int *reads, const int *writes, int *owners, int rows,
                     int batch, int slots, int max_queries, int chunks, bool verify,
                     cudaStream_t stream) {
  initialize_owners<<<(slots + 255) / 256, 256, 0, stream>>>(owners, slots);
  check_cuda(cudaGetLastError());
  validate_sequences<<<(batch + 255) / 256, 256, 0, stream>>>(starts, reads, owners, rows, batch,
                                                              slots, max_queries, verify);
  check_cuda(cudaGetLastError());
  if (rows) {
    validate_exports<<<(rows + 255) / 256, 256, 0, stream>>>(writes, owners, rows, slots, verify);
    check_cuda(cudaGetLastError());
  }
  if (chunks >= 0) {
    validate_capacity<<<1, 1, 0, stream>>>(starts, writes, rows, batch, chunks);
    check_cuda(cudaGetLastError());
  }
}

__global__ void build(const int *starts, const int *writes, int *metadata, int *offsets, int rows,
                      int capacity) {
  int sequence = blockIdx.x;
  if (threadIdx.x)
    return;
  int base = 0;
  for (int prior = 0; prior < sequence; ++prior)
    base += chunk_count(writes, starts[prior], starts[prior + 1]);
  int first = starts[sequence], last = starts[sequence + 1];
  gdn_require(first >= 0 && first <= last && last <= rows);
  gdn_require(build_sequence(writes, first, last, metadata, base, capacity));
  offsets[sequence] = base;
  if (sequence + 1 == gridDim.x)
    offsets[sequence + 1] = base + chunk_count(writes, first, last);
}

inline void launch(const int *starts, const int *writes, int *metadata, int *offsets, int rows,
                   int chunks, int batch, cudaStream_t stream) {
  check_cuda(cudaMemsetAsync(metadata, 0, int64_t(chunks) * 4 * sizeof(int), stream));
  build<<<batch, 32, 0, stream>>>(starts, writes, metadata, offsets, rows, chunks);
  check_cuda(cudaGetLastError());
}

} // namespace gdn_radix_metadata
