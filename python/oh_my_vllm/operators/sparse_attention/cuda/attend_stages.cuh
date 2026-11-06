#pragma once

template <bool Bf16Window, bool Bf16Compressed>
__device__ __forceinline__ void
attend_load_cache(const int32_t *ids, const uint8_t *window, const uint8_t *window_scales,
                  const uint8_t *compressed, const uint8_t *compressed_scales, __nv_bfloat16 *kv,
                  int64_t window_rows, bool &invalid) {
  if constexpr (Bf16Window && Bf16Compressed) {
#pragma unroll 1
    for (int group = 0; group < 16; group += 4) {
      uint4 values[4] = {};
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        int i = (threadIdx.x + (group + part) * 256) * 8;
        int id = ids[i / 512], channel = i % 512;
        if (id >= 0) {
          const auto *table =
              reinterpret_cast<const __nv_bfloat16 *>(id < window_rows ? window : compressed);
          int64_t index = id < window_rows ? id : id - window_rows;
          values[part] = *reinterpret_cast<const uint4 *>(table + index * 512 + channel);
        }
      }
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        int i = (threadIdx.x + (group + part) * 256) * 8;
        invalid |= !finite_bf16_vector(values[part]);
        *reinterpret_cast<uint4 *>(kv + address(i / 512, i % 512, 512)) = values[part];
      }
    }
  } else if constexpr (Bf16Window && !Bf16Compressed) {
    for (int group = threadIdx.x; group < 64 * 32; group += blockDim.x) {
      int slot = group / 32, channel = (group % 32) * 16;
      int id = ids[slot];
      uint4 values[2] = {};
      if (id >= 0 && id < window_rows) {
        const auto *source =
            reinterpret_cast<const __nv_bfloat16 *>(window) + int64_t(id) * 512 + channel;
        values[0] = *reinterpret_cast<const uint4 *>(source);
        values[1] = *reinterpret_cast<const uint4 *>(source + 8);
      } else if (id >= 0) {
        read_compressed_group16(compressed, compressed_scales, id - window_rows, channel, values,
                                invalid);
      }
      invalid |= !finite_bf16_vector(values[0]) || !finite_bf16_vector(values[1]);
      *reinterpret_cast<uint4 *>(kv + address(slot, channel, 512)) = values[0];
      *reinterpret_cast<uint4 *>(kv + address(slot, channel + 8, 512)) = values[1];
    }
  } else {
    for (int i = threadIdx.x; i < 64 * 512; i += blockDim.x) {
      int id = ids[i / 512], channel = i % 512;
      auto value = __float2bfloat16_rn(0.f);
      if (id >= 0 && id < window_rows) {
        if constexpr (Bf16Window) {
          value = reinterpret_cast<const __nv_bfloat16 *>(window)[int64_t(id) * 512 + channel];
        } else {
          uint8_t code = window[int64_t(id) * 512 + channel];
          uint8_t scale = window_scales[int64_t(id) * 16 + channel / 32];
          invalid |= (code & 127) == 127 || scale == 255;
          value = __float2bfloat16_rn(ldexpf(fp8(code), int(scale) - 127));
        }
      } else if (id >= 0) {
        value = read_compressed<Bf16Compressed>(compressed, compressed_scales, id - window_rows,
                                                channel, invalid);
      }
      invalid |= !isfinite(__bfloat162float(value));
      kv[address(i / 512, channel, 512)] = value;
    }
  }
}

__device__ __forceinline__ bool attend_scores(const __nv_bfloat16 *q, const __nv_bfloat16 *kv,
                                              const int32_t *ids, float *maxima, float (&scores)[4],
                                              __nv_bfloat16 *output, uint8_t *status, bool &invalid,
                                              int lane, int warp, int fr, int fc) {
  for (int inner = 0; inner < 512; inner += 16) {
    unsigned a[4], b[2];
    load_a(a, q, inner, lane, 512);
    load_key(b, kv, inner, warp, lane);
    mma(scores, a, b);
  }
#pragma unroll
  for (int rg = 0; rg < 2; ++rg) {
    float local = -INFINITY;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int slot = warp * 8 + fc + j;
      float value = scores[rg * 2 + j] * 0.044194173824159216f;
      invalid |= ids[slot] != -1 && !isfinite(value);
      value = ids[slot] != -1 ? value : -INFINITY;
      scores[rg * 2 + j] = value;
      local = fmaxf(local, value);
    }
    local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, 1));
    local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, 2));
    if (lane % 4 == 0)
      maxima[warp * 16 + fr + rg * 8] = local;
  }
  if (__syncthreads_or(invalid)) {
    reject(output, status);
    return false;
  }
  return true;
}

__device__ __forceinline__ void attend_mix(const float (&scores)[4], const __nv_bfloat16 *kv,
                                           __nv_bfloat16 *p, float *maxima, float *sums,
                                           float (&maximum)[2], float (&denominator)[2],
                                           float (&accumulated)[8][4], int lane, int warp, int fr,
                                           int fc) {
  float alpha[2], block_sum[2];
#pragma unroll
  for (int rg = 0; rg < 2; ++rg) {
    int r = fr + rg * 8;
    float next = maximum[rg];
#pragma unroll
    for (int w = 0; w < 8; ++w)
      next = fmaxf(next, maxima[w * 16 + r]);
    alpha[rg] = expf(maximum[rg] - next);
    maximum[rg] = next;
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float probability = expf(scores[rg * 2 + j] - next);
      sum += probability;
      p[address(r, warp * 8 + fc + j, 64)] = __float2bfloat16_rn(probability);
    }
    block_sum[rg] = sum;
  }
  released_block_sums(block_sum, sums, maxima);
#pragma unroll
  for (int rg = 0; rg < 2; ++rg)
    denominator[rg] = __fmaf_rn(denominator[rg], alpha[rg], block_sum[rg]);
#pragma unroll
  for (int tile = 0; tile < 8; ++tile)
#pragma unroll
    for (int j = 0; j < 4; ++j)
      accumulated[tile][j] *= alpha[j / 2];
  for (int inner = 0; inner < 64; inner += 16) {
    unsigned a[4];
    load_a(a, p, inner, lane, 64);
#pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
      unsigned b[2];
      load_value(b, kv, inner, (tile * 8 + warp) * 8, lane);
      mma(accumulated[tile], a, b);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void attend_store(const float *sink, __nv_bfloat16 *output,
                                             uint8_t *status, const float (&maximum)[2],
                                             float (&denominator)[2],
                                             const float (&accumulated)[8][4], bool &invalid,
                                             int head, int fr, int fc, int warp) {
#pragma unroll
  for (int rg = 0; rg < 2; ++rg)
    denominator[rg] += expf(sink[head + fr + rg * 8] - maximum[rg]);
#pragma unroll
  for (int tile = 0; tile < 8; ++tile)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // A finite sink can underflow for an all-masked row.
      float normalized =
          denominator[j / 2] == 0.f ? 0.f : accumulated[tile][j] / denominator[j / 2];
      auto value = __float2bfloat16_rn(normalized);
      invalid |= !isfinite(__bfloat162float(value));
      output[(fr + (j / 2) * 8) * 512 + (tile * 8 + warp) * 8 + fc + j % 2] = value;
    }
  if (__syncthreads_or(invalid))
    reject(output, status);
  else if (threadIdx.x == 0)
    *status = 0;
}
