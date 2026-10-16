#pragma once

#include "fa_attention.cuh"

constexpr int FAWarpTransposeSharedBytes = 73088;

__device__ __forceinline__ void fa_q1_load_value_a(unsigned (&a)[4], const __nv_bfloat16 *value,
                                                   int inner, int dimension, int lane) {
  int token = inner + (lane / 16) * 8 + lane % 8;
  int column = dimension + ((lane / 8) % 2) * 8;
  unsigned address = __cvta_generic_to_shared(value + fa_shared_index(token, column));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(address));
}

__global__ void fa_q1_warp_partial_kernel(
    const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ cache,
    const int32_t *__restrict__ tables, const int32_t *__restrict__ lengths,
    const int32_t *__restrict__ starts, float *__restrict__ partial, float *__restrict__ lse,
    int table_width, int splits, int first, int query_rows, int64_t cache_pages) {
  extern __shared__ __align__(32) unsigned char q1_warp_storage[];
  auto *q = reinterpret_cast<__nv_bfloat16 *>(q1_warp_storage);
  auto *key = q + 8 * 256;
  auto *value = key + 64 * 256;
  auto *probability = value + 64 * 256;
  auto *scores = reinterpret_cast<float *>(probability + 8 * 64);
  auto *maxima = scores + 64 * 8;
  auto *sums = maxima + 4 * 8;
  auto *alphas = sums + 4 * 8;
  auto *inverses = alphas + 8;
  int thread = threadIdx.x, warp = thread / 32, lane = thread & 31;
  int fr = lane / 4, fc = (lane % 4) * 2;
  int sequence = blockIdx.x, head = blockIdx.y, split = blockIdx.z;
  int start = starts[sequence], end = starts[sequence + 1];
  if (start < 0 || end > query_rows || end - start != 1 || starts[0] != 0 ||
      starts[gridDim.x] != query_rows) {
    asm volatile("trap;");
    return;
  }
  int last = lengths[start];
  if (last < first || int64_t(last) > int64_t(table_width) * 784) {
    asm volatile("trap;");
    return;
  }
  int chunk = ((max(last - first, 0) + splits * 64 - 1) / (splits * 64)) * 64;
  int begin = first + split * chunk, limit = min(begin + chunk, last);
  float accumulated[4][4] = {};
  float denominator = 0, maximum = -INFINITY;
#pragma unroll
  for (int index = thread; index < 8 * 32; index += 128) {
    int row = index / 32, column = (index % 32) * 8;
    auto *destination = q + fa_shared_index(row, column);
    int64_t offset = (int64_t(start) * 24 + head * 6 + row) * 256 + column;
    if ((reinterpret_cast<uintptr_t>(query) & 15) == 0) {
      uint4 values =
          row < 6 ? *reinterpret_cast<const uint4 *>(query + offset) : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4 *>(destination) = values;
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        destination[j] = row < 6 ? query[offset + j] : __float2bfloat16(0);
    }
  }
  __syncthreads();
  if (begin < limit)
    fa_stage_kv<64>(cache, tables, sequence, table_width, cache_pages, begin, last, 4, head, false,
                    key, value);
  for (int base = begin; base < limit; base += 64) {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    {
      float score[4] = {};
#pragma unroll 8
      for (int inner = 0; inner < 256; inner += 16) {
        unsigned ka[4], qb[2];
        fa_load_a(ka, key + warp * 16 * 256, inner, lane);
        fa_load_k(qb, q, inner, 0, lane);
        fa_mma_bf16(score, ka[0], ka[1], ka[2], ka[3], qb[0], qb[1]);
      }
      *reinterpret_cast<float2 *>(scores + (warp * 16 + fr) * 8 + fc) =
          make_float2(score[0], score[1]);
      *reinterpret_cast<float2 *>(scores + (warp * 16 + fr + 8) * 8 + fc) =
          make_float2(score[2], score[3]);
    }
    __syncthreads();
    float score[4], m = -INFINITY;
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int column = kt * 32 + warp * 8 + fc + j;
        float current = fr < 6 && base + column < last
                            ? scores[column * 8 + fr] * 0.09016844005556021f
                            : -INFINITY;
        score[kt * 2 + j] = current;
        m = fmaxf(m, current);
      }
    }
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    if (lane % 4 == 0)
      maxima[warp * 8 + fr] = m;
    __syncthreads();
    m = maximum;
#pragma unroll
    for (int w = 0; w < 4; ++w)
      m = fmaxf(m, maxima[w * 8 + fr]);
    float safe = m == -INFINITY ? 0.f : m;
    float alpha = fa_exp2(maximum - safe);
    maximum = m;
    float sum = 0;
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int column = kt * 32 + warp * 8 + fc + j;
        float current = fa_exp2(score[kt * 2 + j] - safe);
        sum += current;
        probability[fa_shared_index(fr, column, 64)] = __float2bfloat16_rn(current);
      }
    }
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    if (lane % 4 == 0) {
      sums[warp * 8 + fr] = sum;
      if (warp == 0)
        alphas[fr] = alpha;
    }
    __syncthreads();
    sum = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w)
      sum += sums[w * 8 + fr];
    denominator = denominator * alpha + sum;
    float correction[2] = {alphas[fc], alphas[fc + 1]};
#pragma unroll
    for (int tile = 0; tile < 4; ++tile)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        accumulated[tile][j] *= correction[j % 2];
#pragma unroll
    for (int inner = 0; inner < 64; inner += 16) {
      unsigned pb[2];
      fa_load_k(pb, probability, inner, 0, lane, 64);
#pragma unroll
      for (int tile = 0; tile < 4; ++tile) {
        unsigned va[4];
        fa_q1_load_value_a(va, value, inner, warp * 64 + tile * 16, lane);
        fa_mma_bf16(accumulated[tile], va[0], va[1], va[2], va[3], pb[0], pb[1]);
      }
    }
    __syncthreads();
    if (base + 64 < limit)
      fa_stage_kv<64>(cache, tables, sequence, table_width, cache_pages, base + 64, last, 4, head,
                      false, key, value);
  }
  float inverse = 1.f / (denominator > 0 ? denominator : 1.f);
  if (warp == 0 && lane % 4 == 0) {
    inverses[fr] = inverse;
    if (fr < 6)
      lse[(int64_t(start) * 24 + head * 6 + fr) * splits + split] = maximum - log2f(inverse);
  }
  __syncthreads();
  float inverse_for_query[2] = {inverses[fc], inverses[fc + 1]};
#pragma unroll
  for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int query_head = fc + j % 2;
      if (query_head < 6) {
        int dimension = warp * 64 + tile * 16 + fr + (j / 2) * 8;
        int64_t offset =
            ((int64_t(start) * 24 + head * 6 + query_head) * splits + split) * 256 + dimension;
        partial[offset] = accumulated[tile][j] * inverse_for_query[j % 2];
      }
    }
  }
}
