#pragma once

#include "../../primitives/cuda/common.cuh"

__global__ void sample_advance_presence_kernel(uint8_t *history, const int64_t *tokens) {
  int row = threadIdx.x;
  int64_t token = tokens[row];
  if (token < 0 || token >= Vocabulary) {
    asm volatile("trap;");
    return;
  }
  history[row * (Vocabulary / 8) + token / 8] |= uint8_t(1u << (token % 8));
}

void sample_advance_presence(TensorView history, TensorView tokens) {
  TVM_FFI_ICHECK(history.ndim() == 2 && history.size(0) > 0 && history.size(0) <= 20);
  int rows = history.size(0);
  tensor_contract(history, history, kDLUInt, 8, {rows, Vocabulary / 8});
  tensor_contract(tokens, history, kDLInt, 64, {rows});
  disjoint(history, tokens);
  tvm::ffi::CUDADeviceGuard guard(history.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, history.device().device_id));
  sample_advance_presence_kernel<<<1, rows, 0, stream>>>(
      static_cast<uint8_t *>(history.data_ptr()), static_cast<const int64_t *>(tokens.data_ptr()));
  check_cuda(cudaGetLastError());
}
