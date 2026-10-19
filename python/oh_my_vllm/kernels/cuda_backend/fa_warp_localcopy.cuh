#pragma once

#include "fa_attention.cuh"

template <bool Value, int QueryWarps>
__device__ __forceinline__ void
fa_warp_localcopy_stage(const __nv_bfloat16 *cache, const int32_t *tables, int sequence,
                        int table_width, int64_t cache_pages, int64_t base, int64_t last, int head,
                        __nv_bfloat16 *destination) {
  int64_t first_page = min(base, last - 1) / 784;
  int64_t final_page = min(base + 63, last - 1) / 784;
  if (first_page < 0 || final_page >= table_width) {
    asm volatile("trap;");
    return;
  }
  int64_t table_base = int64_t(sequence) * table_width;
  int64_t first_block = tables[table_base + first_page];
  int64_t final_block = first_page == final_page ? first_block : tables[table_base + final_page];
  if (first_block < 0 || first_block >= cache_pages || final_block < 0 ||
      final_block >= cache_pages) {
    asm volatile("trap;");
    return;
  }
  int64_t offset = base % 784;
  const auto *first_pointer = cache + (first_block * 1568 * 4 + head + (Value ? 784 * 4 : 0)) * 256;
  const auto *final_pointer = cache + (final_block * 1568 * 4 + head + (Value ? 784 * 4 : 0)) * 256;
#pragma unroll
  for (int i = threadIdx.x % (QueryWarps * 32); i < QueryWarps * 16 * 32; i += QueryWarps * 32) {
    int row = (threadIdx.x / (QueryWarps * 32)) * (QueryWarps * 16) + i / 32;
    int64_t position = base + row;
    int64_t local = offset + row;
    bool crossing = local >= 784;
    const auto *page = crossing ? final_pointer : first_pointer;
    local -= crossing ? 784 : 0;
    const auto *source = page + local * 4 * 256 + (i % 32) * 8;
    unsigned target = __cvta_generic_to_shared(destination + fa_shared_index(row, (i % 32) * 8));
    int bytes = position < last ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;" ::"r"(target),
                 "l"(source), "r"(bytes)
                 : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int QueryWarps> __device__ __forceinline__ void fa_warp_localcopy_sync(int kv_warp) {
  if constexpr (QueryWarps == 1)
    __syncwarp(0xffffffff);
  else
    asm volatile("barrier.sync %0, 64;" ::"r"(kv_warp + 1) : "memory");
}

template <int QueryWarps>
__global__ void fa_warp_localcopy_kernel(
    const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ cache,
    const int32_t *__restrict__ tables, const int32_t *__restrict__ lengths,
    const int32_t *__restrict__ starts, float *__restrict__ partial, float *__restrict__ lse,
    int table_width, int splits, int first, int max_queries, int query_rows, int64_t cache_pages) {
  static_assert(QueryWarps == 1 || QueryWarps == 2);
  constexpr int KVWarps = 4 / QueryWarps, QueryTile = QueryWarps * 16;
  constexpr int ScoreTiles = 4 / KVWarps, KVPerWarp = 64 / KVWarps;
  constexpr int OutputSlots = 128, StageFloats = 4 * OutputSlots * 32;
  constexpr int StageBytes = StageFloats * sizeof(float) + 4 * 16 * 2 * sizeof(float);
  constexpr int QKVBytes = (QueryTile * 256 + 2 * 64 * 256) * sizeof(__nv_bfloat16);
  static_assert(StageBytes == 66048 && StageBytes <= QKVBytes);
  extern __shared__ __align__(32) unsigned char warp_localcopy_storage[];
  auto *q = reinterpret_cast<__nv_bfloat16 *>(warp_localcopy_storage);
  auto *key = q + QueryTile * 256;
  auto *value = key + 64 * 256;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  const int query_warp = warp % QueryWarps, kv_warp = warp / QueryWarps;
  const int fr = lane / 4, fc = (lane % 4) * 2;
  const int sequence = blockIdx.x, head = blockIdx.y, split = blockIdx.z;
  const int start = starts[sequence], end = starts[sequence + 1];
  if (start < 0 || end > query_rows || end <= start || end - start > max_queries ||
      starts[0] != 0 || starts[gridDim.x] != query_rows) {
    asm volatile("trap;");
    return;
  }
  const int64_t last = lengths[end - 1];
  if (last < first || first < 0 || last > int64_t(table_width) * 784) {
    asm volatile("trap;");
    return;
  }
  int64_t row_lengths[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    int r = query_warp * 16 + fr + j * 8;
    int row = start + r / 6;
    row_lengths[j] = row < end ? lengths[row] : 0;
    if (row_lengths[j] < 0 || row_lengths[j] > last) {
      asm volatile("trap;");
      return;
    }
  }
  const int64_t chunk = ((last - first + splits * 64 - 1) / (splits * 64)) * 64;
  const int64_t begin = first + int64_t(split) * chunk;
  const int64_t limit = min(begin + chunk, last);
  float output[32][4] = {};
  float denominator[2] = {}, maximum[2] = {-INFINITY, -INFINITY};
#pragma unroll
  for (int i = threadIdx.x; i < QueryTile * 32; i += 128) {
    int r = i / 32, column = (i % 32) * 8, row = start + r / 6;
    auto *destination = q + fa_shared_index(r, column);
    int64_t index = (int64_t(row) * 24 + head * 6 + r % 6) * 256 + column;
    if ((reinterpret_cast<uintptr_t>(query) & 15) == 0) {
      uint4 values =
          row < end ? *reinterpret_cast<const uint4 *>(query + index) : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4 *>(destination) = values;
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        destination[j] = row < end ? query[index + j] : __float2bfloat16(0);
    }
  }
  __syncthreads();
  if (begin < limit) {
    fa_warp_localcopy_stage<false, QueryWarps>(cache, tables, sequence, table_width, cache_pages,
                                               begin, last, head, key);
    fa_warp_localcopy_stage<true, QueryWarps>(cache, tables, sequence, table_width, cache_pages,
                                              begin, last, head, value);
  }
  for (int64_t base = begin; base < limit; base += 64) {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    fa_warp_localcopy_sync<QueryWarps>(kv_warp);
    float score[ScoreTiles][8] = {};
#pragma unroll 8
    for (int inner = 0; inner < 256; inner += 16) {
      unsigned qa[4];
      fa_load_a(qa, q + query_warp * 16 * 256, inner, lane);
#pragma unroll
      for (int kt = 0; kt < ScoreTiles; ++kt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          unsigned kb[2];
          int krow = kv_warp * KVPerWarp + kt * 16 + half * 8;
          fa_load_k(kb, key, inner, krow / 8, lane);
          auto &fragment = *reinterpret_cast<float (*)[4]>(score[kt] + half * 4);
          fa_mma_bf16(fragment, qa[0], qa[1], qa[2], qa[3], kb[0], kb[1]);
        }
      }
    }
    fa_warp_localcopy_sync<QueryWarps>(kv_warp);
    bool next = base + 64 < limit;
    if (next)
      fa_warp_localcopy_stage<false, QueryWarps>(cache, tables, sequence, table_width, cache_pages,
                                                 base + 64, last, head, key);
    float alpha[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float m = maximum[j];
      int r = query_warp * 16 + fr + j * 8;
      bool valid_row = start + r / 6 < end;
#pragma unroll
      for (int kt = 0; kt < ScoreTiles; ++kt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
          for (int element = 0; element < 2; ++element) {
            int slot = half * 4 + j * 2 + element;
            int column = kv_warp * KVPerWarp + kt * 16 + half * 8 + fc + element;
            float scaled = valid_row && base + column < row_lengths[j]
                               ? score[kt][slot] * 0.09016844005556021f
                               : -INFINITY;
            score[kt][slot] = scaled;
            m = fmaxf(m, scaled);
          }
        }
      }
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
      float safe = m == -INFINITY ? 0.f : m;
      alpha[j] = fa_exp2(maximum[j] - safe);
      maximum[j] = m;
      denominator[j] *= alpha[j];
#pragma unroll
      for (int kt = 0; kt < ScoreTiles; ++kt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
          for (int element = 0; element < 2; ++element) {
            int slot = half * 4 + j * 2 + element;
            score[kt][slot] = fa_exp2(score[kt][slot] - safe);
          }
        }
      }
    }
#pragma unroll
    for (int tile = 0; tile < 32; ++tile) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        output[tile][j] *= alpha[j / 2];
    }
    if (next)
      asm volatile("cp.async.wait_group 1;" ::: "memory");
    else
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    fa_warp_localcopy_sync<QueryWarps>(kv_warp);
#pragma unroll
    for (int kt = 0; kt < ScoreTiles; ++kt) {
      unsigned probability[4];
#pragma unroll
      for (int word = 0; word < 4; ++word)
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;"
            : "=r"(probability[word])
            : "f"(score[kt][word * 2]), "f"(score[kt][word * 2 + 1]));
      fa_rowsum_bf16(denominator, probability);
#pragma unroll
      for (int tile = 0; tile < 32; ++tile) {
        unsigned vb[2];
        fa_load_v(vb, value, kv_warp * KVPerWarp + kt * 16, tile * 8, lane);
        fa_mma_bf16(output[tile], probability[0], probability[1], probability[2], probability[3],
                    vb[0], vb[1]);
      }
    }
    fa_warp_localcopy_sync<QueryWarps>(kv_warp);
    if (next)
      fa_warp_localcopy_stage<true, QueryWarps>(cache, tables, sequence, table_width, cache_pages,
                                                base + 64, last, head, value);
  }
  __syncthreads();
  auto *stage = reinterpret_cast<float *>(warp_localcopy_storage);
  auto *states = stage + StageFloats;
#pragma unroll
  for (int tile = 0; tile < 32; ++tile) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      stage[(warp * OutputSlots + tile * 4 + j) * 32 + lane] = output[tile][j];
  }
  if (lane % 4 == 0) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int row = fr + j * 8;
      states[(warp * 16 + row) * 2] = maximum[j];
      states[(warp * 16 + row) * 2 + 1] = denominator[j];
    }
  }
  __syncthreads();
  if (kv_warp != 0)
    return;
  float weights[2][KVWarps], inverse[2], final_maximum[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    int row = fr + j * 8;
    float m = -INFINITY, d = 0.f;
#pragma unroll
    for (int kw = 0; kw < KVWarps; ++kw)
      m = fmaxf(m, states[((kw * QueryWarps + query_warp) * 16 + row) * 2]);
    float safe = m == -INFINITY ? 0.f : m;
#pragma unroll
    for (int kw = 0; kw < KVWarps; ++kw) {
      int state = ((kw * QueryWarps + query_warp) * 16 + row) * 2;
      weights[j][kw] = fa_exp2(states[state] - safe);
      d += states[state + 1] * weights[j][kw];
    }
    inverse[j] = 1.f / (d > 0.f ? d : 1.f);
    final_maximum[j] = m;
  }
#pragma unroll
  for (int tile = 0; tile < 32; ++tile) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int r = query_warp * 16 + fr + (j / 2) * 8;
      int row = start + r / 6;
      if (row < end) {
        float total = 0.f;
#pragma unroll
        for (int kw = 0; kw < KVWarps; ++kw) {
          int source_warp = kw * QueryWarps + query_warp;
          total +=
              stage[(source_warp * OutputSlots + tile * 4 + j) * 32 + lane] * weights[j / 2][kw];
        }
        int column = tile * 8 + fc + j % 2;
        int64_t index = ((int64_t(row) * 24 + head * 6 + r % 6) * splits + split) * 256 + column;
        partial[index] = total * inverse[j / 2];
      }
    }
  }
  if (lane % 4 == 0) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int r = query_warp * 16 + fr + j * 8;
      int row = start + r / 6;
      if (row < end)
        lse[(int64_t(row) * 24 + head * 6 + r % 6) * splits + split] =
            final_maximum[j] - log2f(inverse[j]);
    }
  }
}
