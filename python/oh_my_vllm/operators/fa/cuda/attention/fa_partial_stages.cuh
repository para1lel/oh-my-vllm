#pragma once

template <int BQ>
__device__ __forceinline__ void fa_partial_load_query(const __nv_bfloat16 *query, __nv_bfloat16 *q,
                                                      int start, int end, int qbase, int kh) {
  using Position = int32_t;
  constexpr int BK = 64, h = 24, ratio = 6;
  constexpr int RowGroups = BQ / 8, ScoreTiles = BQ * BK / 512, AccTiles = BQ / 2;
#pragma unroll
  for (int i = threadIdx.x; i < BQ * 32; i += 128) {
    int r = i / 32, col = (i % 32) * 8;
    int row = start + (qbase + r) / ratio;
    auto *destination = q + fa_shared_index(r, col);
    int64_t offset = (static_cast<int64_t>(row) * h + kh * ratio + (qbase + r) % ratio) * 256 + col;
    if ((reinterpret_cast<uintptr_t>(query) & 15) == 0) {
      uint4 values =
          row < end ? *reinterpret_cast<const uint4 *>(query + offset) : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4 *>(destination) = values;
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        destination[j] = row < end ? query[offset + j] : __float2bfloat16(0);
    }
  }
}

template <int BQ>
__device__ __forceinline__ void fa_partial_scores(const __nv_bfloat16 *q, const __nv_bfloat16 *k,
                                                  float (&score)[BQ * 64 / 512][4], int warp,
                                                  int lane) {
  using Position = int32_t;
  constexpr int BK = 64, h = 24, ratio = 6;
  constexpr int RowGroups = BQ / 8, ScoreTiles = BQ * BK / 512, AccTiles = BQ / 2;
#pragma unroll 1
  for (int inner = 0; inner < 256; inner += 16) {
    unsigned qa[BQ / 16][4], kb[BK / 32][2];
#pragma unroll
    for (int rg = 0; rg < BQ / 16; ++rg)
      fa_load_a(qa[rg], q + rg * 16 * 256, inner, lane);
#pragma unroll
    for (int kt = 0; kt < BK / 32; ++kt)
      fa_load_k<true, BK>(kb[kt], k, inner, kt * 4 + warp, lane);
#pragma unroll
    for (int rg = 0; rg < BQ / 16; ++rg) {
#pragma unroll
      for (int kt = 0; kt < BK / 32; ++kt) {
        fa_mma_bf16(score[rg * (BK / 32) + kt], qa[rg][0], qa[rg][1], qa[rg][2], qa[rg][3],
                    kb[kt][0], kb[kt][1]);
      }
    }
  }
  __syncthreads();
}

template <int BQ>
__device__ __forceinline__ bool
fa_partial_probabilities(float (&score)[BQ * 64 / 512][4], __nv_bfloat16 *p, float *maxima,
                         float *row_sums, float (&maximum)[BQ / 8], float (&alpha)[BQ / 8],
                         const int32_t *lengths, int start, int end, int qbase, int base, int last,
                         int fr, int fc, int warp, int lane) {
  using Position = int32_t;
  constexpr int BK = 64, h = 24, ratio = 6;
  constexpr int RowGroups = BQ / 8, ScoreTiles = BQ * BK / 512, AccTiles = BQ / 2;
#pragma unroll
  for (int rg = 0; rg < RowGroups; ++rg) {
    int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
    int row = start + (qbase + r) / ratio;
    Position length = row < end ? lengths[row] : 0;
    if (length < 0 || length > last) {
      asm volatile("trap;");
      return false;
    }
    float m = -INFINITY;
#pragma unroll
    for (int kt = 0; kt < BK / 32; ++kt) {
      int tile = (rg / 2) * (BK / 32) + kt;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int col = (kt * 4 + warp) * 8 + fc + j;
        float value = row < end && base + col < length
                          ? score[tile][(rg % 2) * 2 + j] * 0.09016844005556021f
                          : -INFINITY;
        score[tile][(rg % 2) * 2 + j] = value;
        m = fmaxf(m, value);
      }
    }
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
    if (lane % 4 == 0)
      maxima[warp * BQ + r] = m;
  }
  __syncthreads();
#pragma unroll
  for (int rg = 0; rg < RowGroups; ++rg) {
    int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
    float m = maximum[rg];
#pragma unroll
    for (int w = 0; w < 4; ++w)
      m = fmaxf(m, maxima[w * BQ + r]);
    float safe = m == -INFINITY ? 0.f : m;
    alpha[rg] = fa_exp2(maximum[rg] - safe);
    maximum[rg] = m;
    float sum = 0.f;
#pragma unroll
    for (int kt = 0; kt < BK / 32; ++kt) {
      int tile = (rg / 2) * (BK / 32) + kt;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int col = (kt * 4 + warp) * 8 + fc + j;
        float probability = fa_exp2(score[tile][(rg % 2) * 2 + j] - safe);
        sum += probability;
        p[fa_shared_index(r, col, 64)] = __float2bfloat16_rn(probability);
      }
    }
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    if (lane % 4 == 0)
      row_sums[warp * BQ + r] = sum;
  }
  __syncthreads();
  return true;
}

template <int BQ>
__device__ __forceinline__ bool
fa_partial_tiles(const __nv_bfloat16 *cache, const int32_t *tables, const int32_t *lengths,
                 __nv_bfloat16 *q, __nv_bfloat16 *key_buffers, __nv_bfloat16 *value_buffers,
                 __nv_bfloat16 *p, float *maxima, float *row_sums, uint64_t *ready,
                 float (&acc)[BQ / 2][4], float (&denominator)[BQ / 8], float (&maximum)[BQ / 8],
                 const CUtensorMap &key_map, int seq, int table_width, int64_t cache_pages,
                 int begin, int limit, int last, int kh, int start, int end, int qbase, int fr,
                 int fc, int warp, int lane) {
  using Position = int32_t;
  constexpr int BK = 64, h = 24, ratio = 6;
  constexpr int RowGroups = BQ / 8, ScoreTiles = BQ * BK / 512, AccTiles = BQ / 2;
  for (Position base = begin; base < limit; base += BK) {
    auto *k = key_buffers;
    auto *v = value_buffers;
    cute::wait_barrier(*ready, ((base - begin) / BK) & 1);
    __syncthreads();
    float score[ScoreTiles][4] = {};
    fa_partial_scores<BQ>(q, k, score, warp, lane);
    if (base + BK < limit)
      fa_stage_tiled_kv<0, BK>(cache, tables, seq, table_width, cache_pages, base + BK, last, kh,
                               key_buffers, value_buffers, key_map, ready);
    float alpha[RowGroups];
    if (!fa_partial_probabilities<BQ>(score, p, maxima, row_sums, maximum, alpha, lengths, start,
                                      end, qbase, base, last, fr, fc, warp, lane))
      return false;
#pragma unroll
    for (int rg = 0; rg < RowGroups; ++rg) {
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      denominator[rg] *= alpha[rg];
#pragma unroll
      for (int w = 0; w < 4; ++w)
        denominator[rg] += row_sums[w * BQ + r];
    }
#pragma unroll
    for (int tile = 0; tile < AccTiles; ++tile) {
      int row_base = (tile / 8) * 16;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[tile][j] *= alpha[row_base / 8 + j / 2];
    }
    cute::wait_barrier(ready[1], ((base - begin) / BK) & 1);
    __syncthreads();
#pragma unroll
    for (int inner = 0; inner < BK; inner += 16) {
      unsigned pa[BQ / 16][4];
#pragma unroll
      for (int rg = 0; rg < BQ / 16; ++rg) {
        fa_load_a(pa[rg], p + rg * 16 * 64, inner, lane, 64);
      }
#pragma unroll
      for (int ct = 0; ct < 8; ++ct) {
        unsigned vb[2];
        fa_load_v<true, BK>(vb, v, inner, (ct * 4 + warp) * 8, lane);
#pragma unroll
        for (int rg = 0; rg < BQ / 16; ++rg)
          fa_mma_bf16(acc[rg * 8 + ct], pa[rg][0], pa[rg][1], pa[rg][2], pa[rg][3], vb[0], vb[1]);
      }
    }
    __syncthreads();
    if (base + BK < limit)
      fa_stage_tiled_kv<1, BK>(cache, tables, seq, table_width, cache_pages, base + BK, last, kh,
                               key_buffers, value_buffers, key_map, ready + 1);
  }
  return true;
}

template <int BQ>
__device__ __forceinline__ void
fa_partial_store(float *partial, float *lse, const float (&acc)[BQ / 2][4],
                 float (&denominator)[BQ / 8], const float (&maximum)[BQ / 8], int start, int end,
                 int qbase, int kh, int splits, int split, int fr, int fc, int warp, int lane) {
  using Position = int32_t;
  constexpr int BK = 64, h = 24, ratio = 6;
  constexpr int RowGroups = BQ / 8, ScoreTiles = BQ * BK / 512, AccTiles = BQ / 2;
#pragma unroll
  for (int rg = 0; rg < RowGroups; ++rg)
    denominator[rg] = 1.f / (denominator[rg] > 0 ? denominator[rg] : 1.f);
#pragma unroll
  for (int tile = 0; tile < AccTiles; ++tile) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int rg = (tile / 8) * 2 + j / 2;
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      int qr = qbase + r, row = start + qr / ratio;
      if (row < end) {
        int col = ((tile % 8) * 4 + warp) * 8 + fc + j % 2;
        int64_t dst =
            ((static_cast<int64_t>(row) * h + kh * ratio + qr % ratio) * splits + split) * 256 +
            col;
        partial[dst] = acc[tile][j] * denominator[rg];
      }
    }
  }
  if (warp == 0 && lane % 4 == 0) {
#pragma unroll
    for (int rg = 0; rg < RowGroups; ++rg) {
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      int qr = qbase + r, row = start + qr / ratio;
      if (row < end)
        lse[(static_cast<int64_t>(row) * h + kh * ratio + qr % ratio) * splits + split] =
            maximum[rg] - log2f(denominator[rg]);
    }
  }
}
