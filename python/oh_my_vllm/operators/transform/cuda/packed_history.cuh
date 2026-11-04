#pragma once

namespace {

__global__ void gather_fp4_kernel(const uint8_t *codes, const uint8_t *scales, const int64_t *table,
                                  const int64_t *slots, __nv_bfloat16 *output, int32_t *status,
                                  int64_t elements, int width, int length, int table_slots,
                                  int table_width, int64_t pool_rows, int page_rows, int group_size,
                                  int format) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
       i += int64_t(gridDim.x) * blockDim.x) {
    int channel = i % width;
    int position = (i / width) % length;
    int request = i / width / length;
    int64_t slot = slots[request];
    int64_t block = 0;
    bool invalid = slot < 0 || slot >= table_slots || position / page_rows >= table_width;
    if (!invalid)
      block = table[slot * table_width + position / page_rows];
    invalid |= block < 0 || block >= pool_rows / page_rows;
    int64_t row = invalid ? 0 : block * page_rows + position % page_rows;
    float restored = __uint_as_float(0x7fc00000u);
    if (!invalid) {
      uint8_t packed = codes[row * (width / 2) + channel / 2];
      uint8_t code = (packed >> ((channel % 2) * 4)) & 15;
      uint8_t scale_code = scales[row * (width / group_size) + channel / group_size];
      float scale;
      if (format == 2) {
        __nv_fp8_e4m3 value;
        value.__x = scale_code;
        scale = static_cast<float>(value);
        invalid |= (scale_code & 128) != 0 || scale_code == 0 || scale_code == 127;
      } else {
        scale = ldexpf(1.f, int(scale_code) - 127);
        invalid |= scale_code == 255;
      }
      float value = __half2float(static_cast<__half>(__nv_cvt_fp4_to_halfraw(code, __NV_E2M1)));
      restored = invalid ? __uint_as_float(0x7fc00000u) : __fmul_rn(value, scale);
    }
    auto result = __float2bfloat16_rn(restored);
    invalid |= !isfinite(__bfloat162float(result));
    output[i] = invalid ? __float2bfloat16_rn(__uint_as_float(0x7fc00000u)) : result;
    if (invalid)
      atomicExch(status, 1);
  }
}

} // namespace

void gather_fp4(TensorView codes, TensorView scales, TensorView table, TensorView slots,
                TensorView output, TensorView status, int64_t page_rows, int64_t group_size,
                int64_t format) {
  TVM_FFI_ICHECK(codes.ndim() == 2 && scales.ndim() == 2 && table.ndim() == 2 &&
                 slots.ndim() == 1 && output.ndim() == 3);
  TVM_FFI_ICHECK(page_rows > 0 && page_rows <= INT_MAX &&
                 (group_size == 16 || group_size == 32 || group_size == 128) &&
                 (format == 1 || format == 2));
  int64_t batch = output.size(0), length = output.size(1), width = output.size(2);
  TVM_FFI_ICHECK(batch >= 0 && length >= 0 && length <= INT_MAX && width > 0 && width <= 8192 &&
                 width % group_size == 0 && table.size(0) > 0 && table.size(0) <= INT_MAX &&
                 table.size(1) > 0 && table.size(1) <= INT_MAX &&
                 length <= table.size(1) * page_rows && codes.size(0) > 0 &&
                 codes.size(0) % page_rows == 0);
  TVM_FFI_ICHECK(length == 0 || batch <= std::numeric_limits<int64_t>::max() / length / width);
  tensor_contract(codes, output, kDLUInt, 8, {codes.size(0), width / 2});
  tensor_contract(scales, output, kDLUInt, 8, {codes.size(0), width / group_size});
  tensor_contract(table, output, kDLInt, 64, {table.size(0), table.size(1)});
  tensor_contract(slots, output, kDLInt, 64, {batch});
  tensor_contract(output, output, kDLBfloat, 16, {batch, length, width});
  tensor_contract(status, output, kDLInt, 32, {1});
  for (auto input : {codes, scales, table, slots}) {
    disjoint(input, output);
    disjoint(input, status);
  }
  disjoint(output, status);
  int64_t elements = batch * length * width;
  if (!elements)
    return;
  tvm::ffi::CUDADeviceGuard guard(output.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, output.device().device_id));
  unsigned blocks = static_cast<unsigned>(std::min<int64_t>((elements + 255) / 256, 65535));
  gather_fp4_kernel<<<blocks, 256, 0, stream>>>(static_cast<const uint8_t *>(codes.data_ptr()),
                                                static_cast<const uint8_t *>(scales.data_ptr()),
                                                static_cast<const int64_t *>(table.data_ptr()),
                                                static_cast<const int64_t *>(slots.data_ptr()),
                                                static_cast<__nv_bfloat16 *>(output.data_ptr()),
                                                static_cast<int32_t *>(status.data_ptr()), elements,
                                                width, length, table.size(0), table.size(1),
                                                codes.size(0), page_rows, group_size, format);
  check_cuda(cudaGetLastError());
}
