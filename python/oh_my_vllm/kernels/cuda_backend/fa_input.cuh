#pragma once

#include "common.cuh"

__device__ inline float fa_bf16(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void fa_heads_kernel(const __nv_bfloat16 *packed, const float *q_norm,
                                const float *k_norm, const int64_t *positions, const int32_t *slots,
                                __nv_bfloat16 *cache, __nv_bfloat16 *query, __nv_bfloat16 *gate,
                                int rows, int columns, int variant, int64_t cache_tokens) {
  int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
  bool is_query = variant != 1 && head < 24;
  int kh = variant == 1 ? head : head - 24;
  int offset = is_query ? head * 512 : (variant == 1 ? 0 : 12288) + kh * 256;
  const float *norm = is_query ? q_norm : k_norm;
  float values[8], total = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    values[j] = __bfloat162float(packed[int64_t(row) * columns + offset + lane + j * 32]);
    total += values[j] * values[j];
  }
  float inverse = rsqrtf(warp_sum(total) / 256.f + 1e-6f);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    values[j] = fa_bf16(values[j] * inverse * (norm[lane + j * 32] + 1.f));
  int axis = lane % 3;
  if ((axis == 1 && lane >= 33) || (axis == 2 && lane >= 30))
    axis = 0;
  float angle =
      float(positions[int64_t(axis) * rows + row]) * powf(10000000.f, -float(lane) / 32.f);
  float cosine = fa_bf16(cosf(angle)), sine = fa_bf16(sinf(angle));
  float a = values[0], b = values[1];
  values[0] = fa_bf16(a * cosine - b * sine);
  values[1] = fa_bf16(b * cosine + a * sine);
  int slot = is_query ? 0 : slots[row];
  if (!is_query && (slot < 0 || slot >= cache_tokens)) {
    asm volatile("trap;");
    return;
  }
  int64_t key_base = ((int64_t(slot / 784) * 1568 + slot % 784) * 4 + kh) * 256;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int dim = lane + j * 32;
    if (is_query) {
      query[(int64_t(row) * 24 + head) * 256 + dim] = __float2bfloat16_rn(values[j]);
      gate[(int64_t(row) * 24 + head) * 256 + dim] =
          packed[int64_t(row) * columns + offset + 256 + dim];
    } else {
      cache[key_base + dim] = __float2bfloat16_rn(values[j]);
      cache[key_base + int64_t(784) * 4 * 256 + dim] =
          packed[int64_t(row) * columns + (variant == 1 ? 1024 : 13312) + kh * 256 + dim];
    }
  }
}

__device__ inline float fa_maximum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  return value;
}

template <int Warps>
__global__ void fa_merge_quant_kernel(const float *partial, const float *lse,
                                      const __nv_bfloat16 *gate, __nv_fp8_e4m3 *output,
                                      float *scales, int splits) {
  int group = blockIdx.x, head_row = group / 2, half = group % 2;
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  __shared__ float weights[128];
  __shared__ float4 totals[Warps][32];
  float denominator = 0;
  if (warp == 0) {
    float maximum = -INFINITY;
    for (int i = lane; i < splits; i += 32)
      maximum = fmaxf(maximum, lse[int64_t(head_row) * splits + i]);
    maximum = fa_maximum(maximum);
    float safe = maximum == -INFINITY ? 0.f : maximum;
    for (int i = lane; i < splits; i += 32) {
      float weight = fa_exp2(lse[int64_t(head_row) * splits + i] - safe);
      weights[i] = weight;
      denominator += weight;
    }
    denominator = warp_sum(denominator);
  }
  if constexpr (Warps == 1)
    __syncwarp();
  else
    __syncthreads();
  float values[4] = {};
  for (int i = warp; i < splits; i += Warps) {
    float4 part = *reinterpret_cast<const float4 *>(
        partial + (int64_t(head_row) * splits + i) * 256 + half * 128 + lane * 4);
    values[0] += part.x * weights[i];
    values[1] += part.y * weights[i];
    values[2] += part.z * weights[i];
    values[3] += part.w * weights[i];
  }
  if constexpr (Warps > 1) {
    totals[warp][lane] = make_float4(values[0], values[1], values[2], values[3]);
    __syncthreads();
    if (warp != 0)
      return;
    values[0] = values[1] = values[2] = values[3] = 0;
#pragma unroll
    for (int w = 0; w < Warps; ++w) {
      float4 part = totals[w][lane];
      values[0] += part.x;
      values[1] += part.y;
      values[2] += part.z;
      values[3] += part.w;
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float g = __bfloat162float(gate[int64_t(head_row) * 256 + half * 128 + lane * 4 + j]);
    float exponent = expf(-fabsf(g));
    float sigmoid = fa_bf16((g >= 0 ? 1.f : exponent) / (1.f + exponent));
    values[j] = fa_bf16(fa_bf16(denominator > 0 ? values[j] / denominator : 0.f) * sigmoid);
  }
  quantize_four(make_float4(values[0], values[1], values[2], values[3]),
                output + int64_t(group) * 128 + lane * 4, scales + group);
}
