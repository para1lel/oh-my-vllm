#pragma once

#include "format.cuh"

namespace fa_fp8 {
__global__ void validate_rows(const int32_t *slots, const int8_t *mask, bool use_mask, int start,
                              int end, int rows, int64_t cache_tokens, bool write) {
  int row = start + blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= end)
    return;
  int value = use_mask ? mask[row] : int(write);
  if ((value != 0 && value != 1) || value != int(write) ||
      (write && (slots[row] < PageRows || slots[row] >= cache_tokens)))
    asm volatile("trap;");
}

__global__ void validate_sequences(const int32_t *starts, const int32_t *lengths,
                                   const int32_t *tables, int rows, int batch, int width,
                                   int max_queries, int max_context, int first, int pages) {
  int seq = blockIdx.x, begin = starts[seq], end = starts[seq + 1];
  if (starts[0] != 0 || starts[batch] != rows || begin < 0 || end <= begin || end > rows ||
      end - begin > max_queries) {
    asm volatile("trap;");
    return;
  }
  for (int row = begin + threadIdx.x; row < end; row += blockDim.x) {
    int length = lengths[row];
    if (length < first || length > max_context || length > width * PageRows ||
        (row > begin && lengths[row - 1] > length))
      asm volatile("trap;");
  }
  int final = lengths[end - 1];
  if (final < first || final > max_context || final > width * PageRows)
    return;
  for (int logical = first / PageRows + threadIdx.x; logical < (final + PageRows - 1) / PageRows;
       logical += blockDim.x) {
    int page = tables[int64_t(seq) * width + logical];
    if (page <= 0 || page >= pages)
      asm volatile("trap;");
  }
}
} // namespace fa_fp8
