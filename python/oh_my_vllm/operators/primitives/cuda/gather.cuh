#pragma once

#include <algorithm>

#include "../../runtime/cuda/contracts.cuh"

__global__ void gather_rows_kernel(uint8_t *destination, const uint8_t *source, const int *indices,
                                   int rows, int source_rows, int64_t row_bytes) {
  int row = blockIdx.x;
  int index = indices[row];
  if (index < 0 || index >= source_rows) {
    asm volatile("trap;");
    return;
  }
  // Use 16-byte vectorized loads when the row is 16-byte aligned and a
  // multiple of 16 bytes.  All model tensor rows satisfy this (BF16 widths
  // 5120, 6144, 10240 are all divisible by 8 elements = 16 bytes).
  if ((row_bytes & 15) == 0 && (reinterpret_cast<uintptr_t>(destination) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(source) & 15) == 0) {
    int64_t vec_count = row_bytes / 16;
    const uint4 *src4 = reinterpret_cast<const uint4 *>(source) + int64_t(index) * vec_count;
    uint4 *dst4 = reinterpret_cast<uint4 *>(destination) + int64_t(row) * vec_count;
    for (int64_t i = int64_t(blockIdx.y) * blockDim.x + threadIdx.x; i < vec_count;
         i += int64_t(gridDim.y) * blockDim.x)
      dst4[i] = src4[i];
  } else {
    for (int64_t byte = int64_t(blockIdx.y) * blockDim.x + threadIdx.x; byte < row_bytes;
         byte += int64_t(gridDim.y) * blockDim.x)
      destination[int64_t(row) * row_bytes + byte] = source[int64_t(index) * row_bytes + byte];
  }
}

void gather_rows(TensorView destination, TensorView source, TensorView indices) {
  TVM_FFI_ICHECK(source.ndim() > 0 && destination.ndim() == source.ndim());
  TVM_FFI_ICHECK(source.IsContiguous() && destination.IsContiguous() && indices.IsContiguous());
  TVM_FFI_ICHECK(source.device().device_type == kDLCUDA &&
                 destination.device().device_type == kDLCUDA &&
                 indices.device().device_type == kDLCUDA);
  TVM_FFI_ICHECK(source.device().device_id == destination.device().device_id &&
                 source.device().device_id == indices.device().device_id);
  TVM_FFI_ICHECK(source.dtype() == destination.dtype());
  TVM_FFI_ICHECK(indices.dtype().code == kDLInt && indices.dtype().bits == 32 &&
                 indices.dtype().lanes == 1 && indices.ndim() == 1);
  TVM_FFI_ICHECK(source.size(0) > 0 && source.size(0) <= INT32_MAX &&
                 destination.size(0) <= INT32_MAX && indices.size(0) == destination.size(0));
  for (int i = 1; i < source.ndim(); ++i)
    TVM_FFI_ICHECK(source.size(i) == destination.size(i));
  disjoint(source, destination);
  disjoint(indices, destination);
  if (destination.size(0) == 0)
    return;
  tvm::ffi::CUDADeviceGuard guard(source.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, source.device().device_id));
  int64_t row_bytes = tvm::ffi::GetDataSize(source) / source.size(0);
  int column_blocks = std::clamp<int64_t>(row_bytes / 4096 + (row_bytes % 4096 != 0), 1,
                                          std::max<int64_t>(1, 2048 / destination.size(0)));
  gather_rows_kernel<<<dim3(destination.size(0), column_blocks), 256, 0, stream>>>(
      static_cast<uint8_t *>(destination.data_ptr()),
      static_cast<const uint8_t *>(source.data_ptr()), static_cast<const int *>(indices.data_ptr()),
      destination.size(0), source.size(0), row_bytes);
  check_cuda(cudaGetLastError());
}
