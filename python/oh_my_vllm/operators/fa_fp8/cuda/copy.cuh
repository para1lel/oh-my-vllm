#pragma once

#include "format.cuh"

namespace fa_fp8 {
__global__ void copy_pages(uint8_t *codes, float *scales, int source, int destination, int valid) {
  int plane = blockIdx.y;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < valid * Heads * Width;
       i += gridDim.x * blockDim.x)
    codes[code_index(destination, plane, 0, 0, 0) + i] =
        codes[code_index(source, plane, 0, 0, 0) + i];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < valid * Heads;
       i += gridDim.x * blockDim.x)
    scales[scale_index(destination, plane, 0, 0) + i] =
        scales[scale_index(source, plane, 0, 0) + i];
}

inline void cache_contract(TensorView codes, TensorView scales, TensorView reference) {
  TVM_FFI_ICHECK(codes.ndim() == 5 && codes.size(0) > 1 && codes.size(0) <= INT32_MAX / PageRows);
  int64_t pages = codes.size(0);
  tensor_contract(codes, reference, kDLUInt, 8, {pages, 2, PageRows, Heads, Width});
  tensor_contract(scales, reference, kDLFloat, 32, {pages, 2, PageRows, Heads});
  disjoint(codes, scales);
}
} // namespace fa_fp8

inline void fa_fp8_copy_pages(TensorView codes, TensorView scales, int64_t source,
                              int64_t destination, int64_t valid_tokens) {
  fa_fp8::cache_contract(codes, scales, codes);
  TVM_FFI_ICHECK(source > 0 && source < codes.size(0) && destination > 0 &&
                 destination < codes.size(0) && source != destination);
  TVM_FFI_ICHECK(valid_tokens >= 1 && valid_tokens < fa_fp8::PageRows);
  tvm::ffi::CUDADeviceGuard guard(codes.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, codes.device().device_id));
  fa_fp8::copy_pages<<<dim3((valid_tokens * 1024 + 255) / 256, 2), 256, 0, stream>>>(
      static_cast<uint8_t *>(codes.data_ptr()), static_cast<float *>(scales.data_ptr()), source,
      destination, valid_tokens);
  check_cuda(cudaGetLastError());
}
