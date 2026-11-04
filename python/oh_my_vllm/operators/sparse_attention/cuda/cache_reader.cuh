#pragma once

template <bool Bf16>
__device__ __forceinline__ __nv_bfloat16 read_compressed(const uint8_t *data, const uint8_t *scales,
                                                         int64_t row, int channel, bool &invalid) {
  if constexpr (Bf16) {
    auto value = reinterpret_cast<const __nv_bfloat16 *>(data)[row * 512 + channel];
    invalid |= !isfinite(__bfloat162float(value));
    return value;
  } else {
    uint8_t packed = data[row * 256 + channel / 2];
    uint8_t code = (packed >> ((channel % 2) * 4)) & 15;
    uint8_t scale = scales[row * 32 + channel / 16];
    invalid |= (scale & 128) != 0 || scale == 0 || scale == 127;
    float decoded = __half2float(static_cast<__half>(__nv_cvt_fp4_to_halfraw(code, __NV_E2M1)));
    auto value = __float2bfloat16_rn(__fmul_rn(decoded, fp8(scale)));
    invalid |= !isfinite(__bfloat162float(value));
    return value;
  }
}
