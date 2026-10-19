#pragma once
#include "fa_input.cuh"

__global__ void fa_output_mx_kernel(const __nv_bfloat16 *attended, const __nv_bfloat16 *gate,
                                    __nv_fp8_e4m3 *prepared, unsigned char *prepared_scales,
                                    int rows) {
  int group = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31;
  int64_t offset = int64_t(group) * 128 + lane * 4;
  int row = group / 48, head = group % 48;
  if (row >= rows) {
    *reinterpret_cast<uint32_t *>(prepared + offset) = 0u;
    if (lane == 0) {
      int64_t sf = (int64_t(row / 128) * 48 + head) * 512 + row % 32 * 16 + row % 128 / 32 * 4;
      *reinterpret_cast<uint32_t *>(prepared_scales + sf) = 0u;
    }
    return;
  }
  uint2 input = *reinterpret_cast<const uint2 *>(attended + offset);
  float values[4] = {__uint_as_float(input.x << 16), __uint_as_float(input.x & 0xffff0000u),
                     __uint_as_float(input.y << 16), __uint_as_float(input.y & 0xffff0000u)};
  uint2 packed = *reinterpret_cast<const uint2 *>(gate + offset);
  float gates[4] = {__uint_as_float(packed.x << 16), __uint_as_float(packed.x & 0xffff0000u),
                    __uint_as_float(packed.y << 16), __uint_as_float(packed.y & 0xffff0000u)};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float g = gates[j];
    float exponent = expf(-fabsf(g));
    float sigmoid = fa_bf16((g >= 0 ? 1.f : exponent) / (1.f + exponent));
    values[j] = fa_bf16(fa_bf16(values[j]) * sigmoid);
  }
  float maximum = warp_max(
      fmaxf(fmaxf(fabsf(values[0]), fabsf(values[1])), fmaxf(fabsf(values[2]), fabsf(values[3]))));
  float divisor = fmaxf(maximum, 1e-10f) / 448.f;
  if (!(divisor >= 0.f && isfinite(divisor)))
    asm volatile("trap;");
  unsigned bits = __float_as_uint(divisor);
  unsigned exponent = (bits >> 23) & 255u;
  exponent += (bits & 0x7fffffu) != 0u && exponent < 254u;
  float inverse_scale = 1.f / __uint_as_float(exponent << 23);
  *reinterpret_cast<uint32_t *>(prepared + offset) =
      __nv_fp8x4_e4m3(make_float4(values[0] * inverse_scale, values[1] * inverse_scale,
                                  values[2] * inverse_scale, values[3] * inverse_scale))
          .__x;
  if (lane == 0) {
    int64_t offset =
        (int64_t(row / 128) * 48 + head) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4;
    *reinterpret_cast<uint32_t *>(prepared_scales + offset) = exponent * 0x01010101u;
  }
}
