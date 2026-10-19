#pragma once
#include "common.cuh"

namespace gdn_mx_native {
__global__ void convert_kernel(const __nv_fp8_e4m3 *original, const float *scales,
                               __nv_fp8_e4m3 *prepared, unsigned char *prepared_scales, int rows,
                               int padded_rows, int width, bool weights) {
  int vector = blockIdx.x * blockDim.x + threadIdx.x;
  if (vector >= padded_rows * (width / 4))
    return;
  int row = vector / (width / 4), column = vector % (width / 4) * 4;
  unsigned exponent = 0;
  float factor = 0.f;
  if (row < rows) {
    float scale = scales[int64_t(weights ? row / 128 : row) * (width / 128) + column / 128];
    unsigned bits = __float_as_uint(scale);
    if (!(scale >= 0.f && isfinite(scale)))
      asm volatile("trap;");
    exponent = (bits >> 23) & 255u;
    unsigned mantissa = bits & 0x7fffffu;
    if (exponent == 0u)
      exponent = mantissa > 0x400000u ? 1u : 0u;
    else
      exponent += mantissa != 0u && exponent < 254u;
    float rounded_scale = __uint_as_float(exponent ? exponent << 23 : 0x400000u);
    factor = scale / rounded_scale;
  }
  uint32_t words =
      row < rows ? *reinterpret_cast<const uint32_t *>(original + int64_t(row) * width + column)
                 : 0u;
  __nv_fp8_e4m3 values[4];
  for (int item = 0; item < 4; ++item) {
    __nv_fp8_e4m3 value;
    value.__x = (words >> (item * 8)) & 255u;
    values[item] = __nv_fp8_e4m3(float(value) * factor);
  }
  *reinterpret_cast<uint32_t *>(prepared + int64_t(row) * width + column) =
      *reinterpret_cast<uint32_t *>(values);
  if ((column & 31) == 0) {
    int group = column / 32;
    int64_t tile = int64_t(row / 128) * (width / 128) + group / 4;
    int64_t offset = tile * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 + group % 4;
    prepared_scales[offset] = exponent;
  }
}

} // namespace gdn_mx_native
