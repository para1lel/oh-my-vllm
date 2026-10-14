#pragma once

#include "operators/common.cuh"

void groupwise_fp8_transposed(TensorView a, TensorView weight, TensorView scale_a,
                              TensorView scale_weight, TensorView output, TensorView workspace,
                              bool pdl) {
  TVM_FFI_ICHECK(a.device().device_type == kDLCUDA);
  tvm::ffi::CUDADeviceGuard device_guard(a.device().device_id);
  int major = 0, minor = 0;
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                        a.device().device_id) == cudaSuccess &&
                 cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                        a.device().device_id) == cudaSuccess &&
                 major == 10 && minor == 0);
  TVM_FFI_ICHECK(a.ndim() == 2 && a.size(0) > 0 && a.size(0) <= 32 && a.size(1) == 5120 &&
                 weight.ndim() == 2 && weight.size(1) == 5120 &&
                 (weight.size(0) == 34816 || weight.size(0) == 16384 || weight.size(0) == 14336));
  const int rows = a.size(0), columns = weight.size(0);
  TVM_FFI_ICHECK(output.ndim() == 2 && output.size(0) == rows && output.size(1) == columns &&
                 scale_a.ndim() == 2 && scale_a.size(0) == rows && scale_a.size(1) == 40 &&
                 scale_weight.ndim() == 2 && scale_weight.size(0) == columns / 128 &&
                 scale_weight.size(1) == 40 && workspace.ndim() == 1 && workspace.size(0) >= 16 &&
                 workspace.size(0) <= INT32_MAX);
  TVM_FFI_ICHECK(has_dtype(a, kDLFloat8_e4m3fn, 8) && has_dtype(weight, kDLFloat8_e4m3fn, 8) &&
                 has_dtype(scale_a, kDLFloat, 32) && has_dtype(scale_weight, kDLFloat, 32) &&
                 has_dtype(output, kDLBfloat, 16) && has_dtype(workspace, kDLUInt, 8));
  for (auto tensor : {a, weight, scale_a, scale_weight, output, workspace}) {
    TVM_FFI_ICHECK(same_cuda_device(tensor, a) && aligned(tensor, 16) &&
                   tensor.stride(tensor.ndim() - 1) == 1 &&
                   (tensor.ndim() != 2 || fits_int32_flat_offsets(tensor, true)));
    for (int dimension = tensor.ndim() - 2; dimension >= 0; --dimension)
      TVM_FFI_ICHECK(tensor.stride(dimension) ==
                     tensor.stride(dimension + 1) * tensor.size(dimension + 1));
  }
  for (auto source : {a, weight, scale_a, scale_weight, workspace})
    TVM_FFI_ICHECK(disjoint_storage(output, source));
  auto stream = stream_for(a, "groupwise_fp8_transposed");
  auto status =
      oh_my_vllm::gemm::CutlassGroupwiseScaledGEMMSM100<128, 1, 128, true, 1, cutlass::float_e4m3_t,
                                                        cutlass::bfloat16_t, 0, 1, 256, true, 128,
                                                        1, true, 32>(
          workspace.data_ptr(), workspace.size(0),
          static_cast<cutlass::float_e4m3_t *>(weight.data_ptr()),
          static_cast<cutlass::float_e4m3_t *>(a.data_ptr()),
          static_cast<float *>(scale_weight.data_ptr()), static_cast<float *>(scale_a.data_ptr()),
          static_cast<cutlass::bfloat16_t *>(output.data_ptr()), columns, rows, 5120, 1, stream,
          pdl);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  finish_cuda_launch(stream, "groupwise_fp8_transposed");
}
