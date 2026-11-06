#pragma once

struct DecodeScratch {
  float *scores;
  float *maxima;
  float *sums;
  float *values;
  int32_t *errors;
};

constexpr int DecodeFloats = 16 * 64 + 16 + 16 + 16 * 512 + 1;

inline DecodeScratch decode_scratch(void *data, int tiles) {
  auto *scores = static_cast<float *>(data);
  auto *maxima = scores + int64_t(tiles) * 16 * 64;
  auto *sums = maxima + int64_t(tiles) * 16;
  auto *values = sums + int64_t(tiles) * 16;
  return {scores, maxima, sums, values,
          reinterpret_cast<int32_t *>(values + int64_t(tiles) * 16 * 512)};
}

__device__ __forceinline__ void decode_dot_scores(float (&scores)[4], const __nv_bfloat16 *q,
                                                  const __nv_bfloat16 *kv, int lane, int warp) {
  for (int inner = 0; inner < 512; inner += 16) {
    unsigned a[4], b[2];
    load_a(a, q, inner, lane, 512);
    load_key(b, kv, inner, warp, lane);
    mma(scores, a, b);
  }
}

template <bool Bf16Compressed>
__device__ __forceinline__ void
decode_load_values(__nv_bfloat16 *values, const __nv_bfloat16 *window, const uint8_t *compressed,
                   const uint8_t *compressed_scales, const int32_t *indices, int64_t window_rows,
                   int b, int lane, int row, int slots, int column, bool &invalid) {
#pragma unroll
  for (int step = 0; step < 2; ++step) {
    int selected = step * 32 + lane, slot = b * 64 + selected;
    int id = slot < slots ? indices[int64_t(row) * slots + slot] : -1;
    uint4 value = {};
    if (id >= 0 && id < window_rows) {
      value = *reinterpret_cast<const uint4 *>(window + int64_t(id) * 512 + column);
    } else if (id >= 0) {
      if constexpr (Bf16Compressed) {
        const auto *table = reinterpret_cast<const __nv_bfloat16 *>(compressed);
        value = *reinterpret_cast<const uint4 *>(table + int64_t(id - window_rows) * 512 + column);
      } else {
        auto *decoded = reinterpret_cast<__nv_bfloat16 *>(&value);
#pragma unroll
        for (int c = 0; c < 8; ++c)
          decoded[c] = read_compressed<false>(compressed, compressed_scales, id - window_rows,
                                              column + c, invalid);
      }
    }
    *reinterpret_cast<uint4 *>(values + selected * 8) = value;
  }
}

template <bool Bf16Compressed>
__global__ void decode_scores(const __nv_bfloat16 *query, const float *sink, const int32_t *indices,
                              const __nv_bfloat16 *window, const uint8_t *compressed,
                              const uint8_t *compressed_scales, DecodeScratch scratch, int heads,
                              int slots, int64_t window_rows, int64_t compressed_rows) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto *q = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto *kv = q + 16 * 512;
  auto *maxima = reinterpret_cast<float *>(kv + 64 * 512);
  auto *ids = reinterpret_cast<int32_t *>(maxima + 8 * 16);
  int blocks = (slots + 63) / 64;
  int group = blockIdx.x, part = blockIdx.y;
  int row = group / (heads / 16), head = (group % (heads / 16)) * 16;
  int tile = group * blocks + part;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int fr = lane / 4, fc = (lane % 4) * 2;
  bool invalid = false;
  for (int i = threadIdx.x; i < 16 * 512; i += blockDim.x) {
    auto value = query[(int64_t(row) * heads + head) * 512 + i];
    invalid |= !isfinite(__bfloat162float(value));
    q[address(i / 512, i % 512, 512)] = value;
  }
  if (threadIdx.x < 16)
    invalid |= !isfinite(sink[head + threadIdx.x]);
  if (threadIdx.x < 64) {
    int slot = part * 64 + threadIdx.x;
    int id = slot < slots ? indices[int64_t(row) * slots + slot] : -1;
    invalid |= id < -1 || int64_t(id) >= window_rows + compressed_rows;
    ids[threadIdx.x] = id;
  }
  bool rejected = __syncthreads_or(invalid);
  if (!rejected) {
    for (int i = threadIdx.x; i < 64 * 512; i += blockDim.x) {
      int id = ids[i / 512], channel = i % 512;
      auto value = __float2bfloat16_rn(0.f);
      if (id >= 0 && id < window_rows) {
        value = window[int64_t(id) * 512 + channel];
        invalid |= !isfinite(__bfloat162float(value));
      } else if (id >= 0) {
        value = read_compressed<Bf16Compressed>(compressed, compressed_scales, id - window_rows,
                                                channel, invalid);
      }
      kv[address(i / 512, channel, 512)] = value;
    }
    rejected = __syncthreads_or(invalid);
  }
  float scores[4] = {};
  if (!rejected) {
    decode_dot_scores(scores, q, kv, lane, warp);
  }
#pragma unroll
  for (int rg = 0; rg < 2; ++rg) {
    float maximum = -1e30f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int slot = warp * 8 + fc + j;
      float value = scores[rg * 2 + j] * 0.044194173824159216f;
      invalid |= !rejected && ids[slot] >= 0 && !isfinite(value);
      value = !rejected && ids[slot] >= 0 ? value : -INFINITY;
      scratch.scores[int64_t(tile) * 1024 + (fr + rg * 8) * 64 + slot] = value;
      maximum = fmaxf(maximum, value);
    }
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2));
    if (lane % 4 == 0)
      maxima[warp * 16 + fr + rg * 8] = maximum;
  }
  rejected |= __syncthreads_or(invalid);
  if (threadIdx.x < 16) {
    float maximum = -1e30f;
    for (int w = 0; w < 8; ++w)
      maximum = fmaxf(maximum, maxima[w * 16 + threadIdx.x]);
    scratch.maxima[tile * 16 + threadIdx.x] = maximum;
  }
  if (threadIdx.x == 0)
    scratch.errors[tile] = rejected;
}

__global__ void decode_probabilities(DecodeScratch scratch, int slots) {
  __shared__ float sums[2 * 256];
  int blocks = (slots + 63) / 64;
  int group = blockIdx.x / blocks, part = blockIdx.x % blocks;
  int tile = group * blocks + part;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int fr = lane / 4, fc = (lane % 4) * 2;
  bool rejected = false;
  for (int b = 0; b < blocks; ++b)
    rejected |= scratch.errors[group * blocks + b] != 0;
  if (rejected)
    return;
  float maximum[2] = {-1e30f, -1e30f};
#pragma unroll
  for (int rg = 0; rg < 2; ++rg)
    for (int b = 0; b <= part; ++b)
      maximum[rg] = fmaxf(maximum[rg], scratch.maxima[(group * blocks + b) * 16 + fr + rg * 8]);
  float block_sum[2] = {};
#pragma unroll
  for (int rg = 0; rg < 2; ++rg) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int slot = warp * 8 + fc + j, head = fr + rg * 8;
      float probability =
          expf(scratch.scores[int64_t(tile) * 1024 + head * 64 + slot] - maximum[rg]);
      block_sum[rg] += probability;
      scratch.scores[int64_t(tile) * 1024 + head * 64 + slot] =
          __bfloat162float(__float2bfloat16_rn(probability));
    }
  }
  {
#pragma unroll
    for (int offset = 128; offset >= 32; offset >>= 1) {
      sums[threadIdx.x] = block_sum[0];
      sums[256 + threadIdx.x] = block_sum[1];
      __syncthreads();
      block_sum[0] = __fadd_rn(block_sum[0], sums[threadIdx.x ^ offset]);
      block_sum[1] = __fadd_rn(block_sum[1], sums[256 + (threadIdx.x ^ offset)]);
      if (offset != 32)
        __syncthreads();
    }
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
      float sum = block_sum[rg];
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, 2));
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, 1));
      if (warp == 0 && lane % 4 == 0)
        scratch.sums[tile * 16 + fr + rg * 8] = sum;
    }
  }
}

template <bool Bf16Compressed>
__global__ void decode_ordered_values(const float *sink, const int32_t *indices,
                                      const __nv_bfloat16 *window, const uint8_t *compressed,
                                      const uint8_t *compressed_scales, DecodeScratch scratch,
                                      __nv_bfloat16 *output, int groups, int heads, int slots,
                                      int64_t window_rows) {
  __shared__ __align__(16) __nv_bfloat16 probabilities[16 * 64];
  __shared__ __align__(16) __nv_bfloat16 values[64 * 8];
  int blocks = (slots + 63) / 64;
  int group = blockIdx.x, column = blockIdx.y * 8;
  int row = group / (heads / 16), head_base = (group % (heads / 16)) * 16;
  int lane = threadIdx.x, fr = lane / 4, fc = (lane % 4) * 2;
  bool invalid = false;
  for (int b = 0; b < blocks; ++b)
    invalid |= scratch.errors[group * blocks + b] != 0;
  float accumulated[4] = {}, maximum[2] = {-1e30f, -1e30f}, denominator[2] = {};
  if (!invalid) {
    for (int b = 0; b < blocks; ++b) {
      int tile = group * blocks + b;
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        int i = (step * 32 + lane) * 4;
        float4 loaded =
            *reinterpret_cast<const float4 *>(scratch.scores + int64_t(tile) * 1024 + i);
        auto *destination = probabilities + address(i / 64, i % 64, 64);
        destination[0] = __float2bfloat16_rn(loaded.x);
        destination[1] = __float2bfloat16_rn(loaded.y);
        destination[2] = __float2bfloat16_rn(loaded.z);
        destination[3] = __float2bfloat16_rn(loaded.w);
      }
      decode_load_values<Bf16Compressed>(values, window, compressed, compressed_scales, indices,
                                         window_rows, b, lane, row, slots, column, invalid);
      __syncwarp();
#pragma unroll
      for (int rg = 0; rg < 2; ++rg) {
        float next = fmaxf(maximum[rg], scratch.maxima[tile * 16 + fr + rg * 8]);
        float alpha = expf(maximum[rg] - next);
        maximum[rg] = next;
        denominator[rg] = __fmaf_rn(denominator[rg], alpha, scratch.sums[tile * 16 + fr + rg * 8]);
        accumulated[rg * 2] *= alpha;
        accumulated[rg * 2 + 1] *= alpha;
      }
      for (int inner = 0; inner < 64; inner += 16) {
        unsigned a[4], v[2];
        load_a(a, probabilities, inner, lane, 64);
        unsigned pointer = __cvta_generic_to_shared(values + (inner + lane % 16) * 8);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                     : "=r"(v[0]), "=r"(v[1])
                     : "r"(pointer));
        mma(accumulated, a, v);
      }
      __syncwarp();
    }
#pragma unroll
    for (int rg = 0; rg < 2; ++rg)
      denominator[rg] += expf(sink[head_base + fr + rg * 8] - maximum[rg]);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float normalized = denominator[j / 2] == 0.f ? 0.f : accumulated[j] / denominator[j / 2];
    auto value = __float2bfloat16_rn(normalized);
    invalid |= !isfinite(__bfloat162float(value));
    output[int64_t(group) * 8192 + (fr + (j / 2) * 8) * 512 + column + fc + j % 2] = value;
  }
  bool failed = __any_sync(0xffffffff, invalid);
  if (lane == 0)
    scratch.errors[groups * blocks + group * 64 + blockIdx.y] = failed;
}

__global__ void decode_status(DecodeScratch scratch, __nv_bfloat16 *output, uint8_t *status,
                              int groups, int slots) {
  int group = blockIdx.x, blocks = (slots + 63) / 64;
  bool rejected = false;
  for (int column = 0; column < 64; ++column)
    rejected |= scratch.errors[groups * blocks + group * 64 + column] != 0;
  if (rejected)
    for (int i = threadIdx.x; i < 8192; i += blockDim.x)
      output[int64_t(group) * 8192 + i] = __float2bfloat16_rn(0.f);
  if (threadIdx.x == 0)
    status[group] = rejected;
}
