#pragma once
#include "mx_fp8_convert.cuh"

namespace ffn_mx_native {
using mx_fp8::convert_kernel;

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
