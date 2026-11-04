#pragma once

#include <cstring>

namespace engram_lookup {

inline void host_contract(TensorView value, int code, int bits) {
  TVM_FFI_ICHECK(value.device().device_type == kDLCPU && value.IsContiguous());
  TVM_FFI_ICHECK(value.dtype().code == code && value.dtype().bits == bits &&
                 value.dtype().lanes == 1);
}

__global__ void decode_kernel(const uint8_t *packed, __nv_bfloat16 *output, int rows, int width,
                              int stride) {
  int row = blockIdx.x * 4 + threadIdx.x / 32;
  if (row >= rows)
    return;
  int lane = threadIdx.x % 32;
  const uint8_t *source = packed + int64_t(row) * stride;
  bool masked = source[width + width / 32] != 0;
  for (int group = 0; group < width / 32; ++group) {
    int channel = group * 32 + lane;
    float result = 0.f;
    if (!masked) {
      __nv_fp8_e4m3 value;
      value.__x = source[channel];
      __nv_fp8_e8m0 scale;
      scale.__x = source[width + group];
      result = __fmul_rn(static_cast<float>(value), static_cast<float>(scale));
    }
    output[int64_t(row) * width + channel] = __float2bfloat16_rn(result);
  }
}

} // namespace engram_lookup

void engram_lookup_stage(TensorView weight, TensorView scales, TensorView ids, TensorView packed,
                         int64_t start) {
  engram_lookup::host_contract(weight, kDLUInt, 8);
  engram_lookup::host_contract(scales, kDLUInt, 8);
  engram_lookup::host_contract(ids, kDLInt, 64);
  engram_lookup::host_contract(packed, kDLUInt, 8);
  TVM_FFI_ICHECK(weight.ndim() == 2 && scales.ndim() == 2 && ids.ndim() == 1 && packed.ndim() == 2);
  int64_t count = ids.size(0), rows = weight.size(0), width = weight.size(1);
  TVM_FFI_ICHECK(start >= 0 && rows >= 0 && start <= INT64_MAX - rows);
  TVM_FFI_ICHECK(width > 0 && width <= 8192 && width % 32 == 0);
  int64_t stride = ((width + width / 32 + 1 + 15) / 16) * 16;
  TVM_FFI_ICHECK(count >= 0 && count <= 65536 && scales.size(0) == rows &&
                 scales.size(1) == width / 32 && packed.size(0) == count &&
                 packed.size(1) == stride);
  disjoint(weight, packed);
  disjoint(scales, packed);
  disjoint(ids, packed);
  const auto *indices = static_cast<const int64_t *>(ids.data_ptr());
  const auto *values = static_cast<const uint8_t *>(weight.data_ptr());
  const auto *scale = static_cast<const uint8_t *>(scales.data_ptr());
  auto *destination = static_cast<uint8_t *>(packed.data_ptr());
  for (int64_t i = 0; i < count; ++i) {
    int64_t id = indices[i];
    auto *target = destination + i * stride;
    if (id >= start && id - start < rows) {
      int64_t local = id - start;
      std::memcpy(target, values + local * width, width);
      std::memcpy(target + width, scale + local * (width / 32), width / 32);
      target[width + width / 32] = 0;
    } else {
      std::memset(target, 0, stride);
      target[width + width / 32] = 1;
    }
  }
}

void engram_lookup_decode(TensorView packed, TensorView output) {
  TVM_FFI_ICHECK(output.ndim() == 2 && packed.ndim() == 2);
  int64_t rows = output.size(0), width = output.size(1);
  TVM_FFI_ICHECK(rows >= 0 && rows <= 65536 && width > 0 && width <= 8192 && width % 32 == 0);
  int stride = ((width + width / 32 + 1 + 15) / 16) * 16;
  tensor_contract(output, output, kDLBfloat, 16, {rows, width});
  tensor_contract(packed, output, kDLUInt, 8, {rows, stride});
  disjoint(packed, output);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(output.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, output.device().device_id));
  engram_lookup::decode_kernel<<<(rows + 3) / 4, 128, 0, stream>>>(
      static_cast<const uint8_t *>(packed.data_ptr()),
      static_cast<__nv_bfloat16 *>(output.data_ptr()), rows, width, stride);
  check_cuda(cudaGetLastError());
}
