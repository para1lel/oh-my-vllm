#include <cuda_bf16.h>
#include "../../runtime/cuda/contracts.cuh"

namespace {

__global__ void pool_ratio2_kernel(const float *kv, const float *gates, __nv_bfloat16 *output,
                                   int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  int first = (index / 512) * 1024 + index % 512;
  float a = gates[first], b = gates[first + 512];
  float maximum = fmaxf(a, b);
  float ea = expf(a - maximum), eb = expf(b - maximum);
  float inverse = 1.f / (ea + eb);
  float left = __fmul_rn(kv[first], __fmul_rn(ea, inverse));
  float right = __fmul_rn(kv[first + 512], __fmul_rn(eb, inverse));
  float value = __fadd_rn(left, right);
  output[index] = __float2bfloat16_rn(value);
}

} // namespace

void pool_ratio2(TensorView kv, TensorView gates, TensorView output) {
  TVM_FFI_ICHECK(kv.ndim() == 3 && kv.size(0) >= 0 && kv.size(0) <= 384);
  int rows = kv.size(0);
  tensor_contract(kv, kv, kDLFloat, 32, {rows, 2, 512});
  tensor_contract(gates, kv, kDLFloat, 32, {rows, 2, 512});
  tensor_contract(output, kv, kDLBfloat, 16, {rows, 512});
  disjoint(kv, output);
  disjoint(gates, output);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(kv.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, kv.device().device_id));
  int count = rows * 512;
  pool_ratio2_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
      static_cast<const float *>(kv.data_ptr()), static_cast<const float *>(gates.data_ptr()),
      static_cast<__nv_bfloat16 *>(output.data_ptr()), count);
  check_cuda(cudaGetLastError());
}
