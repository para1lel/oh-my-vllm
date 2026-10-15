#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
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

__device__ inline float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_xor_sync(0xffffffff, value, offset);
  return value;
}

__device__ inline float warp_max(float value) {
  return __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(value)));
}

__device__ inline void quantize_four(float4 value, __nv_fp8_e4m3 *output, float *scale) {
  float maximum =
      warp_max(fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)), fmaxf(fabsf(value.z), fabsf(value.w))));
  float divisor = fmaxf(maximum, 1e-10f) / 448.f;
  if ((threadIdx.x & 31) == 0)
    *scale = divisor;
  *reinterpret_cast<uint32_t *>(output) =
      __nv_fp8x4_e4m3(
          make_float4(value.x / divisor, value.y / divisor, value.z / divisor, value.w / divisor))
          .__x;
}
