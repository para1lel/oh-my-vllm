#pragma once

#include "operators/common.cuh"

namespace oh_my_vllm::gemm {
__global__ void merge_splitk_kernel(const float *partials, __nv_bfloat16 *output, int count,
                                    int splits, bool pdl) {
  if (pdl) {
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
  }
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    // Partial accumulators stay FP32 until the complete projection's BF16 boundary.
    float result = partials[index];
    for (int split = 1; split < splits; ++split)
      result += partials[int64_t(split) * count + index];
    output[index] = __float2bfloat16_rn(result);
  }
}

template <int Splits>
void run_splitk(TensorView a, TensorView weight, TensorView scale_a, TensorView scale_weight,
                TensorView partials, TensorView output, TensorView workspace, cudaStream_t stream,
                bool pdl) {
  auto status = CutlassGroupwiseScaledGEMMSM100<1, 128, 128, true, 1, cutlass::float_e4m3_t, float,
                                                0, 1, 128, true, 64, Splits>(
      workspace.data_ptr(), workspace.size(0), static_cast<cutlass::float_e4m3_t *>(a.data_ptr()),
      static_cast<cutlass::float_e4m3_t *>(weight.data_ptr()),
      static_cast<float *>(scale_a.data_ptr()), static_cast<float *>(scale_weight.data_ptr()),
      static_cast<float *>(partials.data_ptr()), a.size(0), weight.size(0), a.size(1) / Splits,
      Splits, stream, pdl);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  cudaLaunchConfig_t config{};
  const int count = a.size(0) * weight.size(0);
  config.gridDim = (count + 255) / 256;
  config.blockDim = 256;
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = pdl;
  config.attrs = &attribute;
  config.numAttrs = 1;
  TVM_FFI_ICHECK(cudaLaunchKernelEx(&config, merge_splitk_kernel,
                                    static_cast<const float *>(partials.data_ptr()),
                                    static_cast<__nv_bfloat16 *>(output.data_ptr()), count, Splits,
                                    pdl) == cudaSuccess);
}
} // namespace oh_my_vllm::gemm

void groupwise_fp8_splitk(TensorView a, TensorView weight, TensorView scale_a,
                          TensorView scale_weight, TensorView partials, TensorView output,
                          TensorView workspace, int64_t splits, bool pdl) {
  TVM_FFI_ICHECK(a.device().device_type == kDLCUDA);
  tvm::ffi::CUDADeviceGuard device_guard(a.device().device_id);
  int major = 0, minor = 0;
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                        a.device().device_id) == cudaSuccess &&
                 cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                        a.device().device_id) == cudaSuccess &&
                 major == 10 && minor == 0);
  TVM_FFI_ICHECK(a.ndim() == 2 && a.size(0) > 0 && a.size(0) <= 32 && weight.ndim() == 2 &&
                 weight.size(0) == 5120 && weight.size(1) == a.size(1) &&
                 (a.size(1) == 17408 || a.size(1) == 6144) && (splits == 2 || splits == 8) &&
                 a.size(1) % (splits * 128) == 0);
  const int rows = a.size(0), groups = a.size(1) / 128;
  TVM_FFI_ICHECK(output.ndim() == 2 && output.size(0) == rows && output.size(1) == 5120 &&
                 partials.ndim() == 3 && partials.size(0) == splits && partials.size(1) == rows &&
                 partials.size(2) == 5120 && scale_a.ndim() == 2 && scale_a.size(0) == rows &&
                 scale_a.size(1) == groups && scale_weight.ndim() == 2 &&
                 scale_weight.size(0) == 40 && scale_weight.size(1) == groups &&
                 workspace.ndim() == 1 && workspace.size(0) >= 16 &&
                 workspace.size(0) <= INT32_MAX);
  TVM_FFI_ICHECK(has_dtype(a, kDLFloat8_e4m3fn, 8) && has_dtype(weight, kDLFloat8_e4m3fn, 8) &&
                 has_dtype(scale_a, kDLFloat, 32) && has_dtype(scale_weight, kDLFloat, 32) &&
                 has_dtype(partials, kDLFloat, 32) && has_dtype(output, kDLBfloat, 16) &&
                 has_dtype(workspace, kDLUInt, 8));
  for (auto tensor : {a, weight, scale_a, scale_weight, partials, output, workspace}) {
    TVM_FFI_ICHECK(same_cuda_device(tensor, a) && aligned(tensor, 16) &&
                   tensor.stride(tensor.ndim() - 1) == 1 &&
                   (tensor.ndim() != 2 || fits_int32_flat_offsets(tensor, true)));
    for (int dimension = tensor.ndim() - 2; dimension >= 0; --dimension)
      TVM_FFI_ICHECK(tensor.stride(dimension) ==
                     tensor.stride(dimension + 1) * tensor.size(dimension + 1));
  }
  TVM_FFI_ICHECK(disjoint_storage(partials, output));
  for (auto destination : {partials, output})
    for (auto source : {a, weight, scale_a, scale_weight, workspace})
      TVM_FFI_ICHECK(disjoint_storage(destination, source));
  auto stream = stream_for(a, "groupwise_fp8_splitk");
  if (splits == 2)
    oh_my_vllm::gemm::run_splitk<2>(a, weight, scale_a, scale_weight, partials, output, workspace,
                                    stream, pdl);
  else
    oh_my_vllm::gemm::run_splitk<8>(a, weight, scale_a, scale_weight, partials, output, workspace,
                                    stream, pdl);
  finish_cuda_launch(stream, "groupwise_fp8_splitk");
}
