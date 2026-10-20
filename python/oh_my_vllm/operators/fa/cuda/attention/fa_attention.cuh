#pragma once

#include "../../../primitives/cuda/common.cuh"
#include <mutex>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>

__device__ __forceinline__ float fa_exp2(float value) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ int fa_shared_index(int row, int col, int stride = 256) {
  return row * stride + (col ^ ((row & 7) * 8));
}

template <bool Tiled = false, int TileRows = 64>
__device__ __forceinline__ int fa_kv_index(int row, int col, int stride = 256) {
  if constexpr (Tiled)
    return (col / 64) * (TileRows * 64) + row * 64 + ((col % 64) ^ ((row & 7) * 8));
  return fa_shared_index(row, col, stride);
}

template <bool Tiled = false, int TileRows = 64>
__device__ __forceinline__ void fa_load_a(unsigned (&a)[4], const __nv_bfloat16 *data, int inner,
                                          int lane, int stride = 256) {
  unsigned address = __cvta_generic_to_shared(
      data + fa_kv_index<Tiled, TileRows>(lane % 16, inner + (lane / 16) * 8, stride));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(address));
}

template <bool Tiled = false, int TileRows = 64>
__device__ __forceinline__ void fa_load_k(unsigned (&b)[2], const __nv_bfloat16 *data, int inner,
                                          int warp, int lane, int stride = 256) {
  unsigned address = __cvta_generic_to_shared(
      data +
      fa_kv_index<Tiled, TileRows>(warp * 8 + lane % 8, inner + ((lane / 8) % 2) * 8, stride));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(address));
}

template <bool Tiled = false, int TileRows = 64>
__device__ __forceinline__ void fa_load_v(unsigned (&b)[2], const __nv_bfloat16 *data, int inner,
                                          int col, int lane, int stride = 256) {
  unsigned address =
      __cvta_generic_to_shared(data + fa_kv_index<Tiled, TileRows>(inner + lane % 16, col, stride));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(address));
}

__device__ __forceinline__ void fa_mma_bf16(float (&c)[4], unsigned a0, unsigned a1, unsigned a2,
                                            unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int BK, bool Tiled = false, int Operand = -1, typename Position>
__device__ __forceinline__ void fa_stage_kv(const __nv_bfloat16 *cache, const int32_t *tables,
                                            int start, int table_width, int64_t cache_pages,
                                            Position base, Position last, int hk, int kh,
                                            __nv_bfloat16 *k, __nv_bfloat16 *v) {
  Position first_page = min(base, last - 1) / 784;
  Position final_page = min(base + BK - 1, last - 1) / 784;
  if (first_page < 0 || final_page >= table_width) {
    asm volatile("trap;");
    return;
  }
  int64_t table_row = static_cast<int64_t>(start) * table_width;
  int64_t first_block = tables[table_row + first_page];
  int64_t final_block = first_page == final_page ? first_block : tables[table_row + final_page];
  if (first_block < 0 || first_block >= cache_pages || final_block < 0 ||
      final_block >= cache_pages) {
    asm volatile("trap;");
    return;
  }
  Position base_offset = base % 784;
  const __nv_bfloat16 *first_pointer = cache + (first_block * 1568 * hk + kh) * 256;
  const __nv_bfloat16 *final_pointer = cache + (final_block * 1568 * hk + kh) * 256;
#pragma unroll
  for (int i = threadIdx.x; i < BK * 32; i += 128) {
    Position pos = base + i / 32;
    Position local = base_offset + i / 32;
    bool crossing = local >= 784;
    const __nv_bfloat16 *page_pointer = crossing ? final_pointer : first_pointer;
    local -= crossing ? 784 : 0;
    const __nv_bfloat16 *source = page_pointer + int64_t(local) * hk * 256 + (i % 32) * 8;
    unsigned kd = __cvta_generic_to_shared(k + fa_kv_index<Tiled, BK>(i / 32, (i % 32) * 8));
    unsigned vd = __cvta_generic_to_shared(v + fa_kv_index<Tiled, BK>(i / 32, (i % 32) * 8));
    int bytes = pos < last ? 16 : 0;
    if constexpr (Operand != 1)
      asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;" ::"r"(kd), "l"(source),
                   "r"(bytes)
                   : "memory");
    if constexpr (Operand != 0)
      asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;" ::"r"(vd),
                   "l"(source + int64_t(784) * hk * 256), "r"(bytes)
                   : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int Operand, int BK = 64>
__device__ inline void fa_stage_tiled_kv(const __nv_bfloat16 *cache, const int32_t *tables, int seq,
                                         int table_width, int64_t cache_pages, int base, int last,
                                         int kh, __nv_bfloat16 *key, __nv_bfloat16 *value,
                                         const CUtensorMap &map, uint64_t *ready) {
  if (base + BK <= last && base % 784 + BK <= 784) {
    if (threadIdx.x == 0) {
      int page = tables[seq * table_width + base / 784];
      if (page < 0 || page >= cache_pages) {
        asm volatile("trap;");
        return;
      }
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(ready, BK * 512);
      if constexpr (Operand == 0)
        cute::SM90_TMA_LOAD_5D::copy(&map, ready, uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                     key, 0, base % 784, 0, kh, page * 2);
      if constexpr (Operand == 1)
        cute::SM90_TMA_LOAD_5D::copy(&map, ready, uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                     value, 0, base % 784, 0, kh, page * 2 + 1);
    }
  } else {
    fa_stage_kv<BK, true, Operand>(cache, tables, seq, table_width, cache_pages, base, last, 4, kh,
                                   key, value);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0)
      cutlass::arch::ClusterBarrier::arrive(ready);
  }
}

template <int QueryTile>
__global__ __launch_bounds__(128, 3) void fa_partial_kernel(
    const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ cache,
    const int32_t *__restrict__ tables, const int32_t *__restrict__ lengths,
    const int32_t *__restrict__ starts, float *__restrict__ partial, float *__restrict__ lse,
    int table_width, int splits, int first, int query_tiles, int max_query_len, int query_rows,
    int64_t cache_pages, const __grid_constant__ CUtensorMap key_map) {
  using Position = int32_t;
  constexpr int BQ = QueryTile, BK = 64, h = 24, hk = 4;
  constexpr int RowGroups = BQ / 8, ScoreTiles = BQ * BK / 512, AccTiles = BQ / 2;
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *q = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto *key_buffers = q + BQ * 256;
  auto *value_buffers = key_buffers + BK * 256;
  auto *p = value_buffers + BK * 256;
  auto *maxima = reinterpret_cast<float *>(p + BQ * 64);
  auto *row_sums = maxima + 4 * BQ;
  auto *ready = reinterpret_cast<uint64_t *>(maxima + 8 * BQ);
  if (threadIdx.x == 0) {
    cute::initialize_barrier(ready[0], 1);
    cute::initialize_barrier(ready[1], 1);
    cutlass::arch::fence_barrier_init();
  }
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  const int fr = lane / 4, fc = (lane % 4) * 2;
  const int seq = blockIdx.x / query_tiles, qbase = (blockIdx.x % query_tiles) * BQ;
  const int kh = blockIdx.y, split = blockIdx.z, ratio = h / hk;
  const int64_t start64 = starts[seq];
  const int64_t end64 = starts[seq + 1];
  const int sequences = gridDim.x / query_tiles;
  if (start64 < 0 || end64 > query_rows || end64 <= start64 || end64 - start64 > max_query_len ||
      starts[0] != 0 || starts[sequences] != query_rows) {
    asm volatile("trap;");
    return;
  }
  const int start = start64, end = end64;
  if (start + qbase / ratio >= end)
    return;
  const int64_t last64 = lengths[min(end - 1, start + (qbase + BQ - 1) / ratio)];
  if (last64 < first || last64 > int64_t(table_width) * 784) {
    asm volatile("trap;");
    return;
  }
  const Position last = last64;
  const Position chunk = ((max(last - first, Position(0)) + splits * BK - 1) / (splits * BK)) * BK;
  const Position begin = first + split * chunk;
  float acc[AccTiles][4] = {}, denominator[RowGroups] = {}, maximum[RowGroups];
#pragma unroll
  for (int rg = 0; rg < RowGroups; ++rg)
    maximum[rg] = -INFINITY;
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
  __syncthreads();
  const Position limit = min(begin + chunk, last);
  if (begin < limit) {
    fa_stage_tiled_kv<0, BK>(cache, tables, seq, table_width, cache_pages, begin, last, kh,
                             key_buffers, value_buffers, key_map, ready);
    fa_stage_tiled_kv<1, BK>(cache, tables, seq, table_width, cache_pages, begin, last, kh,
                             key_buffers, value_buffers, key_map, ready + 1);
  }
  for (Position base = begin; base < limit; base += BK) {
    auto *k = key_buffers;
    auto *v = value_buffers;
    cute::wait_barrier(*ready, ((base - begin) / BK) & 1);
    __syncthreads();
    float score[ScoreTiles][4] = {};
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
    if (base + BK < limit)
      fa_stage_tiled_kv<0, BK>(cache, tables, seq, table_width, cache_pages, base + BK, last, kh,
                               key_buffers, value_buffers, key_map, ready);
#pragma unroll
    for (int rg = 0; rg < RowGroups; ++rg) {
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      int row = start + (qbase + r) / ratio;
      Position length = row < end ? lengths[row] : 0;
      if (length < 0 || length > last) {
        asm volatile("trap;");
        return;
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
    float alpha[RowGroups];
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
