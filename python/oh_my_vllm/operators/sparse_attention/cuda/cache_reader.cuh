/*
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Pair dequantization adapted from FlashMLA csrc/cuda_kernels/sm100/dequant_utils.cuh,
revision 2e5429fc5653bab6e081f09477126f731882a6a9. Uses CUDA primitives without kerutils.
*/

#pragma once

__device__ __forceinline__ void read_compressed_group16(const uint8_t *data, const uint8_t *scales,
                                                        int64_t row, int channel,
                                                        uint4 (&values)[2], bool &invalid) {
  uint2 packed = *reinterpret_cast<const uint2 *>(data + row * 256 + channel / 2);
  uint8_t scale = scales[row * 32 + channel / 16];
  invalid |= (scale & 128) != 0 || scale == 0 || scale == 127;
  auto multiplier = __float2bfloat162_rn(fp8(scale));
  auto *decoded = reinterpret_cast<__nv_bfloat162 *>(values);
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    uint32_t word = c < 4 ? packed.x : packed.y;
    uint8_t codes = word >> ((c & 3) * 8);
    auto value = static_cast<__nv_bfloat162>(__nv_cvt_fp4x2_to_bf162raw(codes, __NV_E2M1));
    // E2M1 times E4M3 has at most six significant bits and is exact in BF16.
    decoded[c] = __hmul2(value, multiplier);
  }
}

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
