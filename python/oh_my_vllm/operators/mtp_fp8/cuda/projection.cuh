#pragma once

#include "../../mtp/cuda/mtp_projection.cuh"

namespace mtp_fp8 {
using ProjectionKey = std::tuple<int, uintptr_t>;

inline auto &projection_maps() {
  static thread_local std::map<ProjectionKey, CUtensorMap> maps;
  return maps;
}

inline void prepare_projection(TensorView weight) {
  auto key =
      ProjectionKey{weight.device().device_id, reinterpret_cast<uintptr_t>(weight.data_ptr())};
  auto &maps = projection_maps();
  if (maps.find(key) != maps.end())
    return;
  check_cuda(cudaFuncSetAttribute(mtp_projection_kernel<8>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  65536 + 8 * 512 + 128));
  check_cuda(cudaFuncSetAttribute(mtp_projection_kernel<16>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  65536 + 16 * 512 + 128));
  check_cuda(cudaFuncSetAttribute(mtp_projection_kernel<32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  65536 + 32 * 512 + 128));
  CUtensorMap descriptor{};
  uint64_t dimensions[2] = {10240, 5120}, strides[1] = {20480};
  uint32_t box[2] = {64, 64}, elements[2] = {1, 1};
  auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, weight.data_ptr(), dimensions, strides, box,
      elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
  maps.emplace(key, descriptor);
}

inline void project(TensorView input, TensorView weight, TensorView partials, int rows,
                    cudaStream_t stream) {
  auto key =
      ProjectionKey{weight.device().device_id, reinterpret_cast<uintptr_t>(weight.data_ptr())};
  auto &maps = projection_maps();
  auto found = maps.find(key);
  TVM_FFI_ICHECK(found != maps.end()) << "MTP input weight was not prepared on this thread";
  auto launch = [&](auto capacity) {
    constexpr int Capacity = decltype(capacity)::value;
    constexpr int Shared = 65536 + Capacity * 256 * 2 + 128;
    mtp_projection_kernel<Capacity><<<dim3(80, 4), 128, Shared, stream>>>(
        static_cast<const __nv_bfloat16 *>(input.data_ptr()),
        static_cast<float *>(partials.data_ptr()), rows, found->second);
    check_cuda(cudaGetLastError());
  };
  if (rows <= 8)
    launch(std::integral_constant<int, 8>{});
  else if (rows <= 16)
    launch(std::integral_constant<int, 16>{});
  else
    launch(std::integral_constant<int, 32>{});
}
} // namespace mtp_fp8

void mtp_fp8_prepare_input(TensorView weight, int rows) {
  TVM_FFI_ICHECK(rows > 0 && rows <= 32768);
  tensor_contract(weight, weight, kDLBfloat, 16, {5120, 10240});
  tvm::ffi::CUDADeviceGuard guard(weight.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, weight.device().device_id));
  cudaStreamCaptureStatus status;
  check_cuda(cudaStreamIsCapturing(stream, &status));
  TVM_FFI_ICHECK(status == cudaStreamCaptureStatusNone)
      << "prepare MTP input before Graph recording";
  if (rows <= 32)
    mtp_fp8::prepare_projection(weight);
}
