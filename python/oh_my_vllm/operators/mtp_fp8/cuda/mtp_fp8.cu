#include "input.cuh"
#include "../../mtp/cuda/accept.cuh"

namespace mtp_fp8 {
__global__ void endpoints_kernel(const __nv_bfloat16 *value, const __nv_bfloat16 *zero,
                                 const int32_t *sequence_rows, const int32_t *endpoint_rows,
                                 __nv_bfloat16 *endpoints, __nv_bfloat16 *endpoint_zero, int rows,
                                 int batch) {
  int request = blockIdx.x, first = sequence_rows[request], last = sequence_rows[request + 1];
  int selected = endpoint_rows[request];
  if (sequence_rows[0] != 0 || sequence_rows[batch] != rows || first < 0 || last <= first ||
      last > rows || selected < first || selected >= last) {
    asm volatile("trap;");
    return;
  }
  for (int vector = threadIdx.x; vector < 640; vector += blockDim.x) {
    reinterpret_cast<uint4 *>(endpoints)[request * 640 + vector] =
        reinterpret_cast<const uint4 *>(value)[int64_t(selected) * 640 + vector];
    reinterpret_cast<uint4 *>(endpoint_zero)[request * 640 + vector] =
        reinterpret_cast<const uint4 *>(zero)[int64_t(selected) * 640 + vector];
  }
}
} // namespace mtp_fp8

void mtp_fp8_endpoints(TensorView value, TensorView zero, TensorView sequence_rows,
                       TensorView endpoint_rows, TensorView endpoints, TensorView endpoint_zero) {
  TVM_FFI_ICHECK(value.ndim() == 2 && value.size(0) > 0 && value.size(0) <= 32768 &&
                 value.size(1) == 5120);
  TVM_FFI_ICHECK(sequence_rows.ndim() == 1 && sequence_rows.size(0) > 1 &&
                 sequence_rows.size(0) <= value.size(0) + 1);
  int rows = value.size(0), batch = sequence_rows.size(0) - 1;
  for (auto tensor : {value, zero})
    tensor_contract(tensor, value, kDLBfloat, 16, {rows, 5120});
  tensor_contract(sequence_rows, value, kDLInt, 32, {batch + 1});
  tensor_contract(endpoint_rows, value, kDLInt, 32, {batch});
  for (auto tensor : {endpoints, endpoint_zero}) {
    tensor_contract(tensor, value, kDLBfloat, 16, {batch, 5120});
    for (auto input : {value, zero, sequence_rows, endpoint_rows})
      disjoint(tensor, input);
  }
  disjoint(endpoints, endpoint_zero);
  tvm::ffi::CUDADeviceGuard guard(value.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, value.device().device_id));
  mtp_fp8::endpoints_kernel<<<batch, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(value.data_ptr()),
      static_cast<const __nv_bfloat16 *>(zero.data_ptr()),
      static_cast<const int32_t *>(sequence_rows.data_ptr()),
      static_cast<const int32_t *>(endpoint_rows.data_ptr()),
      static_cast<__nv_bfloat16 *>(endpoints.data_ptr()),
      static_cast<__nv_bfloat16 *>(endpoint_zero.data_ptr()), rows, batch);
  check_cuda(cudaGetLastError());
}
