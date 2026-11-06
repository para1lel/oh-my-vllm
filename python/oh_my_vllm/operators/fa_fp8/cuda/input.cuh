#pragma once

#include "format.cuh"
#include "../../fa/cuda/input/fa_input.cuh"

namespace fa_fp8 {
__global__ void kv_heads(const __nv_bfloat16 *packed, const float *norm, const int64_t *positions,
                         const int32_t *slots, uint8_t *codes, float *scales, int rows,
                         int source_start, int packed_start, int64_t cache_tokens) {
  int row = source_start + blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
  int64_t input_base = int64_t(packed_start + blockIdx.x) * 2048 + head * 256;
  int slot = slots[row];
  if (slot < PageRows || slot >= cache_tokens) {
    asm volatile("trap;");
    return;
  }
  float values[8], total = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    values[j] = __bfloat162float(packed[input_base + lane + j * 32]);
    total += values[j] * values[j];
  }
  float inverse = rsqrtf(warp_sum(total) / 256.f + 1e-6f);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    values[j] = bf16(values[j] * inverse * (norm[lane + j * 32] + 1.f));
  int axis = lane % 3;
  if ((axis == 1 && lane >= 33) || (axis == 2 && lane >= 30))
    axis = 0;
  float angle =
      float(positions[int64_t(axis) * rows + row]) * powf(10000000.f, -float(lane) / 32.f);
  float cosine = bf16(cosf(angle)), sine = bf16(sinf(angle));
  float a = values[0], b = values[1];
  values[0] = bf16(a * cosine - b * sine);
  values[1] = bf16(b * cosine + a * sine);
  int page = slot / PageRows, token = slot % PageRows;
  encode_vector(values, codes + code_index(page, 0, token, head, 0),
                scales + scale_index(page, 0, token, head));
#pragma unroll
  for (int j = 0; j < 8; ++j)
    values[j] = __bfloat162float(packed[input_base + 1024 + lane + j * 32]);
  encode_vector(values, codes + code_index(page, 1, token, head, 0),
                scales + scale_index(page, 1, token, head));
}
} // namespace fa_fp8
