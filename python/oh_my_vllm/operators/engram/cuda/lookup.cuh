#pragma once

#include "../../runtime/cuda/contracts.cuh"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__global__ void engram_lookup_kernel(const uint8_t *table, const uint8_t *scales,
                                     const int64_t *indices, __nv_bfloat16 *output, uint8_t *status,
                                     int64_t table_rows) {
  int row = blockIdx.x, channel = threadIdx.x;
  int64_t index = indices[row];
  if (index < 0 || index >= table_rows) {
    output[row * 256 + channel] = __float2bfloat16_rn(0.f);
    if (channel == 0)
      status[row] = 1;
    return;
  }
  uint8_t code = table[index * 256 + channel];
  uint8_t scale = scales[index * 8 + channel / 32];
  float value = __half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
  auto rounded = __float2bfloat16_rn(ldexpf(value, static_cast<int>(scale) - 127));
  bool invalid = (code & 0x7f) == 0x7f || scale == 255 || !isfinite(__bfloat162float(rounded));
  bool rejected = __syncthreads_or(invalid);
  output[row * 256 + channel] = rejected ? __float2bfloat16_rn(0.f) : rounded;
  if (channel == 0)
    status[row] = rejected ? 2 : 0;
}

void engram_lookup(TensorView table, TensorView indices, TensorView output, TensorView status,
                   int64_t table_rows, int64_t table_offset, int64_t scale_offset) {
  TVM_FFI_ICHECK(table.ndim() == 1 && table.size(0) > 0 && table_rows > 0 &&
                 table_rows <= INT32_MAX && table_offset >= 0 && scale_offset >= 0);
  int64_t bytes = table.size(0);
  TVM_FFI_ICHECK(table_offset <= bytes && table_rows <= (bytes - table_offset) / 256 &&
                 scale_offset <= bytes && table_rows <= (bytes - scale_offset) / 8);
  TVM_FFI_ICHECK(table_offset + table_rows * 256 <= scale_offset ||
                 scale_offset + table_rows * 8 <= table_offset);
  TVM_FFI_ICHECK(indices.ndim() == 2 && indices.size(0) >= 0 && indices.size(0) <= 384 &&
                 indices.size(1) >= 1 && indices.size(1) <= 24);
  int rows = indices.size(0), heads = indices.size(1);
  tensor_contract(table, indices, kDLUInt, 8, {bytes});
  tensor_contract(indices, indices, kDLInt, 64, {rows, heads});
  tensor_contract(output, indices, kDLBfloat, 16, {rows, heads, 256});
  tensor_contract(status, indices, kDLUInt, 8, {rows, heads});
  if (!rows)
    return;
  for (auto written : {output, status})
    for (auto read : {table, indices})
      disjoint(written, read);
  disjoint(output, status);
  tvm::ffi::CUDADeviceGuard guard(indices.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, indices.device().device_id));
  auto data = static_cast<const uint8_t *>(table.data_ptr());
  engram_lookup_kernel<<<rows * heads, 256, 0, stream>>>(
      data + table_offset, data + scale_offset, static_cast<const int64_t *>(indices.data_ptr()),
      static_cast<__nv_bfloat16 *>(output.data_ptr()), static_cast<uint8_t *>(status.data_ptr()),
      table_rows);
  check_cuda(cudaGetLastError());
}
