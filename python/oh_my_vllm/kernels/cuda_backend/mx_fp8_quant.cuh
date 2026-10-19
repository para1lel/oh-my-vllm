#pragma once
#include "common.cuh"

namespace ffn_mx_native {
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

__global__ void silu_table_initialize_kernel(float *table) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 65536)
    return;
  float gate = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(index)));
  float exponent = expf(-fabsf(gate));
  float sigmoid = (gate >= 0 ? 1.f : exponent) / (1.f + exponent);
  table[index] = gate * sigmoid;
}

__global__ void activate_mx_kernel(const __nv_bfloat16 *gate_up, __nv_fp8_e4m3 *prepared,
                                   unsigned char *scales, int rows, int padded_rows,
                                   const float *silu_table) {
  int group = blockIdx.x * 8 + threadIdx.x / 32;
  if (group >= padded_rows * 136)
    return;
  int row = group / 136, inner_group = group % 136, lane = threadIdx.x & 31;
  int column = inner_group * 128 + lane * 4;
  float values[4] = {0.f, 0.f, 0.f, 0.f};
  if (row < rows) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      unsigned gate = __bfloat16_as_ushort(gate_up[int64_t(row) * 34816 + column + j]);
      float up = __bfloat162float(gate_up[int64_t(row) * 34816 + 17408 + column + j]);
      values[j] = __bfloat162float(__float2bfloat16_rn(silu_table[gate] * up));
    }
  }
  float maximum = warp_max(
      fmaxf(fmaxf(fabsf(values[0]), fabsf(values[1])), fmaxf(fabsf(values[2]), fabsf(values[3]))));
  float divisor = fmaxf(maximum, 1e-10f) / 448.f;
  unsigned bits = __float_as_uint(divisor);
  unsigned exponent = (bits >> 23) & 255u;
  exponent += (bits & 0x7fffffu) != 0u && exponent < 254u;
  float inverse_scale = 1.f / __uint_as_float(exponent << 23);
  *reinterpret_cast<uint32_t *>(prepared + int64_t(row) * 17408 + column) =
      __nv_fp8x4_e4m3(make_float4(values[0] * inverse_scale, values[1] * inverse_scale,
                                  values[2] * inverse_scale, values[3] * inverse_scale))
          .__x;
  if (lane == 0) {
    int64_t tile = int64_t(row / 128) * 136 + inner_group;
    int64_t offset = tile * 512 + (row % 32) * 16 + (row % 128 / 32) * 4;
    *reinterpret_cast<uint32_t *>(scales + offset) = row < rows ? exponent * 0x01010101u : 0u;
  }
}

} // namespace ffn_mx_native
