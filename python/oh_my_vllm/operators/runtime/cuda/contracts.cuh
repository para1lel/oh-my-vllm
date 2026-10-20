#pragma once

#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

using tvm::ffi::TensorView;

inline void check_cuda(cudaError_t status) {
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

inline void tensor_contract(TensorView tensor, TensorView reference, int code, int bits,
                            std::initializer_list<int64_t> shape) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA &&
                 tensor.device().device_id == reference.device().device_id);
  TVM_FFI_ICHECK(tensor.dtype().code == code && tensor.dtype().bits == bits &&
                 tensor.dtype().lanes == 1);
  TVM_FFI_ICHECK(tensor.ndim() == shape.size() && tensor.IsContiguous());
  int dimension = 0;
  for (auto extent : shape)
    TVM_FFI_ICHECK(tensor.size(dimension++) == extent);
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0);
}

inline void disjoint(TensorView left, TensorView right) {
  auto a = reinterpret_cast<uintptr_t>(left.data_ptr());
  auto b = reinterpret_cast<uintptr_t>(right.data_ptr());
  TVM_FFI_ICHECK(a + tvm::ffi::GetDataSize(left) <= b || b + tvm::ffi::GetDataSize(right) <= a)
      << "operator tensors overlap";
}
