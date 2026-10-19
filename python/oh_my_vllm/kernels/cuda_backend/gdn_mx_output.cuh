#pragma once
#include "gdn_kernels.cuh"

namespace gdn_mx_native {

__global__ void gdn_output_mx_kernel(const __nv_bfloat16 *recurrent, const __nv_bfloat16 *qkvz,
                                     const float *norm, __nv_fp8_e4m3 *prepared,
                                     unsigned char *prepared_scales, int rows) {
  int row = blockIdx.x / 6, head = (blockIdx.x % 6) * 8 + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) {
    *reinterpret_cast<uint32_t *>(prepared + int64_t(row) * 6144 + head * 128 + lane * 4) = 0u;
    if (lane == 0) {
      int64_t offset =
          (int64_t(row / 128) * 48 + head) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4;
      *reinterpret_cast<uint32_t *>(prepared_scales + offset) = 0u;
    }
    return;
  }
  const auto *source = recurrent + (int64_t(row) * 48 + head) * 128;
  float partial[4];
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    float value = __bfloat162float(source[group * 32 + lane]);
    partial[group] = value;
  }
#pragma unroll
  for (int pair = 0; pair < 2; ++pair) {
    float2 x = make_float2(partial[pair * 2], partial[pair * 2 + 1]);
    float2 squared = __fmul2_rn(x, x);
    partial[pair * 2] = squared.x;
    partial[pair * 2 + 1] = squared.y;
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      float2 x = make_float2(partial[pair * 2], partial[pair * 2 + 1]);
      float2 other = make_float2(__shfl_xor_sync(0xffffffffu, x.x, offset),
                                 __shfl_xor_sync(0xffffffffu, x.y, offset));
      float2 sum = __fadd2_rn(x, other);
      partial[pair * 2] = sum.x;
      partial[pair * 2 + 1] = sum.y;
    }
  }
  float total = __shfl_sync(0xffffffffu, (partial[0] + partial[2]) + (partial[1] + partial[3]), 0);
  float inverse = rsqrtf(total / 128.f + 1e-6f);
  int column = lane * 4;
  uint2 input = *reinterpret_cast<const uint2 *>(source + column);
  uint2 gate =
      *reinterpret_cast<const uint2 *>(qkvz + int64_t(row) * 16384 + 10240 + head * 128 + column);
  float values[4] = {__uint_as_float(input.x << 16), __uint_as_float(input.x & 0xffff0000u),
                     __uint_as_float(input.y << 16), __uint_as_float(input.y & 0xffff0000u)};
  float gates[4] = {__uint_as_float(gate.x << 16), __uint_as_float(gate.x & 0xffff0000u),
                    __uint_as_float(gate.y << 16), __uint_as_float(gate.y & 0xffff0000u)};
  float4 raw_weight = *reinterpret_cast<const float4 *>(norm + column);
  float weights[4] = {raw_weight.x, raw_weight.y, raw_weight.z, raw_weight.w};
#pragma unroll
  for (int pair = 0; pair < 2; ++pair) {
    int element = pair * 2;
    float2 normalized = __fmul2_rn(make_float2(values[element], values[element + 1]),
                                   make_float2(inverse, inverse));
    normalized = __fmul2_rn(normalized, make_float2(weights[element], weights[element + 1]));
    float2 gate = make_float2(gates[element], gates[element + 1]);
    float2 sigmoid = make_float2(gdn_sigmoid(gate.x), gdn_sigmoid(gate.y));
    float2 activated = __fmul2_rn(gate, sigmoid);
    float2 product = __fmul2_rn(normalized, activated);
    __nv_bfloat162 rounded = __floats2bfloat162_rn(product.x, product.y);
    unsigned bits = *reinterpret_cast<const unsigned *>(&rounded);
    values[element] = __uint_as_float(bits << 16);
    values[element + 1] = __uint_as_float(bits & 0xffff0000u);
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
  *reinterpret_cast<uint32_t *>(prepared + int64_t(row) * 6144 + head * 128 + column) =
      __nv_fp8x4_e4m3(make_float4(values[0] * inverse_scale, values[1] * inverse_scale,
                                  values[2] * inverse_scale, values[3] * inverse_scale))
          .__x;
  if (lane == 0) {
    int64_t offset =
        (int64_t(row / 128) * 48 + head) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4;
    *reinterpret_cast<uint32_t *>(prepared_scales + offset) = exponent * 0x01010101u;
  }
}

} // namespace gdn_mx_native
