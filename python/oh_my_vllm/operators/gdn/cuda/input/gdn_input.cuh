#pragma once

#include "../recurrent/gdn_kernels.cuh"
#include "../../../runtime/cuda/launch.cuh"
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/cuda_host_adapter.hpp>
#include <cutlass/arch/barrier.h>
#include <map>
#include <tuple>

struct GDNInputConvolution {
  const __nv_bfloat16 *weight;
  __nv_bfloat16 *state;
  const int *starts;
  const int *reads;
  const int *writes;
  __nv_bfloat16 *output;
  int batch;
  int slots;
};

#include "gdn_input_pipeline.cuh"

template <int Rows, bool FuseConvolution>
static inline void gdn_input_launch(TensorView input, TensorView input_scale, TensorView weight,
                                    TensorView weight_scale, TensorView output,
                                    GDNInputConvolution convolution, cudaStream_t stream) {
  auto kernel = gdn_input_pipeline_kernel<Rows, FuseConvolution, true>;
  constexpr int SharedBytes = 106624, Threads = 192;
  using WeightKey = std::tuple<int, uintptr_t>;
  static thread_local std::map<WeightKey, CUtensorMap> weight_maps;
  WeightKey key{weight.device().device_id, reinterpret_cast<uintptr_t>(weight.data_ptr())};
  auto found = weight_maps.find(key);
  if (found == weight_maps.end()) {
    check_cuda(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes));
    CUtensorMap descriptor{};
    uint64_t dimensions[2] = {5120, 16384}, strides[1] = {5120};
    uint32_t box[2] = {128, 64}, elements[2] = {1, 1};
    auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, weight.data_ptr(), dimensions, strides, box,
        elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    found = weight_maps.emplace(key, descriptor).first;
  }
  launch_kernel(kernel, dim3(256), dim3(Threads), SharedBytes, stream,
                static_cast<const __nv_fp8_e4m3 *>(input.data_ptr()),
                static_cast<const float *>(input_scale.data_ptr()),
                static_cast<const float *>(weight_scale.data_ptr()),
                static_cast<__nv_bfloat16 *>(output.data_ptr()), found->second, convolution);
  check_cuda(cudaGetLastError());
}
