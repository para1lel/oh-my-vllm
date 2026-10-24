#include <cuda_bf16.h>
#include "../../runtime/cuda/contracts.cuh"

namespace {

__global__ void widen_kernel(const uint16_t *values, uint32_t *output, uint8_t *status) {
  int row = blockIdx.x, invalid = 0;
  for (int column = threadIdx.x; column < 5120; column += blockDim.x) {
    uint16_t word = values[row * 5120 + column];
    bool nonfinite = (word & 0x7f80) == 0x7f80;
    invalid |= nonfinite;
    // Integer storage preserves signed zero and subnormals without floating arithmetic.
    output[row * 5120 + column] = nonfinite ? 0u : uint32_t(word) << 16;
  }
  int row_invalid = __syncthreads_or(invalid);
  if (threadIdx.x == 0)
    status[row] = row_invalid ? 1 : 0;
}

__device__ float warp_sum(float value) {
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_xor_sync(0xffffffff, value, offset);
  return value;
}

__global__ void rms_kernel(const __nv_bfloat162 *values, const __nv_bfloat162 *weight,
                           __nv_bfloat162 *output, int pairs, float epsilon) {
  int row = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  float sum = 0.f;
  for (int column = threadIdx.x; column < pairs; column += blockDim.x) {
    float2 value = __bfloat1622float2(values[row * pairs + column]);
    sum += value.x * value.x + value.y * value.y;
  }
  __shared__ float totals[8];
  sum = warp_sum(sum);
  if (lane == 0)
    totals[warp] = sum;
  __syncthreads();
  sum = warp_sum(lane < 8 ? totals[lane] : 0.f);
  float inverse = rsqrtf(sum / (2 * pairs) + epsilon);
  for (int column = threadIdx.x; column < pairs; column += blockDim.x) {
    float2 value = __bfloat1622float2(values[row * pairs + column]);
    float2 scale = __bfloat1622float2(weight[column]);
    output[row * pairs + column] =
        __floats2bfloat162_rn((value.x * inverse) * scale.x, (value.y * inverse) * scale.y);
  }
}

__global__ void rope_kernel(const __nv_bfloat162 *values, const float2 *coefficients,
                            __nv_bfloat162 *output, int pairs, int heads, bool inverse) {
  int row = blockIdx.x;
  for (int column = threadIdx.x; column < pairs; column += blockDim.x) {
    auto encoded = values[row * pairs + column];
    if (column < pairs - 32) {
      output[row * pairs + column] = encoded;
      continue;
    }
    float2 value = __bfloat1622float2(encoded);
    float2 coefficient = coefficients[(row / heads) * 32 + column - (pairs - 32)];
    float sine = inverse ? -coefficient.y : coefficient.y;
    output[row * pairs + column] = __floats2bfloat162_rn(value.x * coefficient.x - value.y * sine,
                                                         value.x * sine + value.y * coefficient.x);
  }
}

void values_contract(TensorView values, TensorView output) {
  TVM_FFI_ICHECK(values.ndim() == 2 && values.size(0) >= 0 && values.size(0) <= 384);
  tensor_contract(values, values, kDLBfloat, 16, {values.size(0), values.size(1)});
  tensor_contract(output, values, kDLBfloat, 16, {values.size(0), values.size(1)});
  disjoint(values, output);
}

} // namespace

void widen(TensorView values, TensorView output, TensorView status) {
  TVM_FFI_ICHECK(values.ndim() == 2 && values.size(0) >= 0 && values.size(0) <= 384);
  int rows = values.size(0);
  tensor_contract(values, values, kDLBfloat, 16, {rows, 5120});
  tensor_contract(output, values, kDLFloat, 32, {rows, 5120});
  tensor_contract(status, values, kDLUInt, 8, {rows});
  if (!rows)
    return;
  disjoint(values, output);
  disjoint(values, status);
  disjoint(output, status);
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  widen_kernel<<<rows, 256, 0, stream>>>(static_cast<const uint16_t *>(values.data_ptr()),
                                         static_cast<uint32_t *>(output.data_ptr()),
                                         static_cast<uint8_t *>(status.data_ptr()));
  check_cuda(cudaGetLastError());
}

void rms(TensorView values, TensorView weight, TensorView output, double epsilon) {
  values_contract(values, output);
  TVM_FFI_ICHECK(values.size(1) == 512 || values.size(1) == 1024 || values.size(1) == 1280 ||
                 values.size(1) == 5120);
  TVM_FFI_ICHECK(epsilon == 1e-20 || epsilon == 1e-6);
  int rows = values.size(0), width = values.size(1);
  tensor_contract(weight, values, kDLBfloat, 16, {width});
  disjoint(weight, output);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  rms_kernel<<<rows, 256, 0, stream>>>(static_cast<const __nv_bfloat162 *>(values.data_ptr()),
                                       static_cast<const __nv_bfloat162 *>(weight.data_ptr()),
                                       static_cast<__nv_bfloat162 *>(output.data_ptr()), width / 2,
                                       epsilon);
  check_cuda(cudaGetLastError());
}

void rope(TensorView values, TensorView coefficients, TensorView output, bool inverse) {
  TVM_FFI_ICHECK(values.ndim() == 3 && values.size(0) >= 0 && values.size(0) <= 384);
  TVM_FFI_ICHECK(values.size(1) == 1 || values.size(1) == 8 || values.size(1) == 16 ||
                 values.size(1) == 32 || values.size(1) == 64);
  TVM_FFI_ICHECK(values.size(2) == 128 || values.size(2) == 512);
  int rows = values.size(0), heads = values.size(1), width = values.size(2);
  tensor_contract(values, values, kDLBfloat, 16, {rows, heads, width});
  tensor_contract(output, values, kDLBfloat, 16, {rows, heads, width});
  tensor_contract(coefficients, values, kDLFloat, 32, {rows, 32, 2});
  disjoint(values, output);
  disjoint(coefficients, output);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  rope_kernel<<<rows * heads, 128, 0, stream>>>(
      static_cast<const __nv_bfloat162 *>(values.data_ptr()),
      static_cast<const float2 *>(coefficients.data_ptr()),
      static_cast<__nv_bfloat162 *>(output.data_ptr()), width / 2, heads, inverse);
  check_cuda(cudaGetLastError());
}
