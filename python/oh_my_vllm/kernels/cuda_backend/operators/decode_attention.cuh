#pragma once

#include "attention_math.cuh"

// Keep the complete PV accumulator in registers across all KV tiles. Only
// scores/probabilities and online-softmax statistics cross warp boundaries.
template <int BK, typename Position>
__device__ __forceinline__ void stage_kv(const __nv_bfloat16 *cache, const void *tables, int start,
                                         int table_width, Position base, Position last, int hk,
                                         int kh, bool tw, __nv_bfloat16 *k, __nv_bfloat16 *v) {
  // A BK-wide tile can touch at most two 784-token pages. Each thread loads
  // those table entries once per tile, rather than once per K/V chunk.
  Position first_page = min(base, last - 1) / 784;
  Position final_page = min(base + BK - 1, last - 1) / 784;
  int64_t table_row = static_cast<int64_t>(start) * table_width;
  int64_t first_block = index_at(tables, tw, table_row + first_page);
  int64_t final_block =
      first_page == final_page ? first_block : index_at(tables, tw, table_row + final_page);
#pragma unroll
  for (int i = threadIdx.x; i < BK * 32; i += 128) {
    Position pos = base + i / 32;
    int64_t page = pos / 784 == first_page ? first_block : final_block;
    int64_t offset = ((page * 1568 + pos % 784) * hk + kh) * 256 + (i % 32) * 8;
    unsigned kd = __cvta_generic_to_shared(k + shared_index(i / 32, (i % 32) * 8));
    unsigned vd = __cvta_generic_to_shared(v + shared_index(i / 32, (i % 32) * 8));
    int bytes = pos < last ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;" ::"r"(kd),
                 "l"(cache + offset), "r"(bytes)
                 : "memory");
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;" ::"r"(vd),
                 "l"(cache + offset + 784 * hk * 256), "r"(bytes)
                 : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <bool ModelShape, bool Grouped, typename Position, int Buffers>
__global__ void
attention_partial_kernel(const __nv_bfloat16 *__restrict__ query,
                         const __nv_bfloat16 *__restrict__ cache, const void *__restrict__ tables,
                         const void *__restrict__ lengths, const void *__restrict__ starts,
                         float *__restrict__ partial, float *__restrict__ lse, int runtime_h,
                         int runtime_hk, int table_width, int splits, int first, bool tw, bool lw,
                         bool sw, int query_tiles, int max_query_len) {
  pdl_dependency_wait();
  pdl_launch_next();
  constexpr int BQ = Grouped ? 32 : 16, BK = Grouped ? 64 : 32;
  constexpr int RowGroups = BQ / 8, ScoreTiles = BQ * BK / 512, AccTiles = BQ / 2;
  const int h = ModelShape ? 24 : runtime_h, hk = ModelShape ? 4 : runtime_hk;
  extern __shared__ __align__(32) unsigned char storage[];
  auto *q = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto *key_buffers = q + BQ * 256;
  auto *value_buffers = key_buffers + Buffers * BK * 256;
  auto *p = value_buffers + Buffers * BK * 256;
  auto *maxima = reinterpret_cast<float *>(p + BQ * 64);
  auto *sums = maxima + 4 * BQ;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  const int fr = lane / 4, fc = (lane % 4) * 2;
  const int seq = blockIdx.x / query_tiles, qbase = (blockIdx.x % query_tiles) * BQ;
  const int kh = blockIdx.y, split = blockIdx.z, ratio = h / hk;
  const int start = Grouped ? index_at(starts, sw, seq) : seq;
  const int end = Grouped ? index_at(starts, sw, seq + 1) : seq + 1;
  if constexpr (Grouped) {
    if (end <= start || end - start > max_query_len)
      asm volatile("trap;");
  }
  const Position last = index_at(lengths, lw, end - 1);
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
    auto *destination = q + shared_index(r, col);
    int64_t offset = (static_cast<int64_t>(row) * h + kh * ratio + (qbase + r) % ratio) * 256 + col;
    if ((reinterpret_cast<uintptr_t>(query) & 15) == 0) {
      uint4 values =
          row < end ? *reinterpret_cast<const uint4 *>(query + offset) : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4 *>(destination) = values;
    } else {
      // A contiguous tensor may begin at a BF16 storage offset, not a16B boundary.
#pragma unroll
      for (int j = 0; j < 8; ++j)
        destination[j] = row < end ? query[offset + j] : __float2bfloat16(0);
    }
  }
  __syncthreads();
  const Position limit = min(begin + chunk, last);
  if (begin < limit)
    stage_kv<BK>(cache, tables, start, table_width, begin, last, hk, kh, tw, key_buffers,
                 value_buffers);
  int step = 0;
  for (Position base = begin; base < limit; base += BK, ++step) {
    auto *k = key_buffers + (step % Buffers) * BK * 256;
    auto *v = value_buffers + (step % Buffers) * BK * 256;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if constexpr (Buffers == 2) {
      if (base + BK < limit)
        stage_kv<BK>(cache, tables, start, table_width, base + BK, last, hk, kh, tw,
                     key_buffers + ((step + 1) % Buffers) * BK * 256,
                     value_buffers + ((step + 1) % Buffers) * BK * 256);
    }
    float score[ScoreTiles][4] = {};
#pragma unroll 8
    for (int inner = 0; inner < 256; inner += 16) {
      unsigned qa[BQ / 16][4], kb[BK / 32][2];
#pragma unroll
      for (int rg = 0; rg < BQ / 16; ++rg)
        load_a(qa[rg], q + rg * 16 * 256, inner, lane);
#pragma unroll
      for (int kt = 0; kt < BK / 32; ++kt)
        load_k(kb[kt], k, inner, kt * 4 + warp, lane);
#pragma unroll
      for (int rg = 0; rg < BQ / 16; ++rg) {
#pragma unroll
        for (int kt = 0; kt < BK / 32; ++kt) {
          mma_bf16(score[rg * (BK / 32) + kt], qa[rg][0], qa[rg][1], qa[rg][2], qa[rg][3],
                   kb[kt][0], kb[kt][1]);
        }
      }
    }
#pragma unroll
    for (int rg = 0; rg < RowGroups; ++rg) {
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      int row = start + (qbase + r) / ratio;
      Position length = row < end ? index_at(lengths, lw, row) : 0;
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
      alpha[rg] = softmax_exp2(maximum[rg] - safe);
      maximum[rg] = m;
      float sum = 0;
#pragma unroll
      for (int kt = 0; kt < BK / 32; ++kt) {
        int tile = (rg / 2) * (BK / 32) + kt;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          int col = (kt * 4 + warp) * 8 + fc + j;
          float probability = softmax_exp2(score[tile][(rg % 2) * 2 + j] - safe);
          sum += probability;
          p[shared_index(r, col, 64)] = __float2bfloat16_rn(probability);
        }
      }
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      sum += __shfl_xor_sync(0xffffffff, sum, 2);
      if (lane % 4 == 0)
        sums[warp * BQ + r] = sum;
    }
    __syncthreads();
#pragma unroll
    for (int rg = 0; rg < RowGroups; ++rg) {
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      float sum = 0;
#pragma unroll
      for (int w = 0; w < 4; ++w)
        sum += sums[w * BQ + r];
      denominator[rg] = denominator[rg] * alpha[rg] + sum;
    }
#pragma unroll
    for (int tile = 0; tile < AccTiles; ++tile) {
      // Four column warps make the row group independent of warp. Keep this
      // index static so alpha and denominator remain register arrays.
      int row_base = (tile / 8) * 16;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[tile][j] *= alpha[row_base / 8 + j / 2];
    }
#pragma unroll
    for (int inner = 0; inner < BK; inner += 16) {
      unsigned pa[BQ / 16][4];
#pragma unroll
      for (int rg = 0; rg < BQ / 16; ++rg)
        load_a(pa[rg], p + rg * 16 * 64, inner, lane, 64);
#pragma unroll
      for (int ct = 0; ct < 8; ++ct) {
        unsigned vb[2];
        load_v(vb, v, inner, (ct * 4 + warp) * 8, lane);
#pragma unroll
        for (int rg = 0; rg < BQ / 16; ++rg)
          mma_bf16(acc[rg * 8 + ct], pa[rg][0], pa[rg][1], pa[rg][2], pa[rg][3], vb[0], vb[1]);
      }
    }
    __syncthreads();
    if constexpr (Buffers == 1) {
      if (base + BK < limit)
        stage_kv<BK>(cache, tables, start, table_width, base + BK, last, hk, kh, tw, key_buffers,
                     value_buffers);
    }
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
template <bool ModelShape, bool Grouped, typename Position, int Buffers>
void configure_attention_shared_memory(int shared_bytes) {
  // ModelShape, Grouped and Buffers determine this kernel's shared footprint.
  // Once-only setup also keeps subsequent CUDA Graph captures free of setters.
  static std::once_flag configured;
  std::call_once(configured, [shared_bytes] {
    TVM_FFI_ICHECK(cudaFuncSetAttribute(
                       attention_partial_kernel<ModelShape, Grouped, Position, Buffers>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes) == cudaSuccess);
  });
}
template <typename Position>
void launch_attention(TensorView q, TensorView cache, TensorView tables, TensorView lengths,
                      TensorView starts, TensorView partial, TensorView lse, int64_t first,
                      bool grouped, int max_query_len, cudaStream_t stream) {
  int h = q.size(1), hk = cache.size(3), splits = lse.size(2);
  int bq = grouped ? 32 : 16, bk = grouped ? 64 : 32;
  int tiles = ((grouped ? max_query_len : 1) * (h / hk) + bq - 1) / bq;
  // Only four ungrouped queries benefit from double buffering on B200.
  // Larger eager batches need the occupancy afforded by one shared KV tile.
  int shared_bytes =
      (bq * 256 + 2 * ((!grouped && q.size(0) == 4) ? 2 : 1) * bk * 256 + bq * 64) * 2 +
      (8 * bq) * 4;
  dim3 grid((grouped ? starts.size(0) - 1 : q.size(0)) * tiles, hk, splits);
#define ATTENTION(M, G, B)                                                                         \
  configure_attention_shared_memory<M, G, Position, B>(shared_bytes);                              \
  launch_kernel(attention_partial_kernel<M, G, Position, B>, grid, 128, shared_bytes, stream,      \
                static_cast<const __nv_bfloat16 *>(q.data_ptr()),                                  \
                static_cast<const __nv_bfloat16 *>(cache.data_ptr()), tables.data_ptr(),           \
                lengths.data_ptr(), starts.data_ptr(), static_cast<float *>(partial.data_ptr()),   \
                static_cast<float *>(lse.data_ptr()), h, hk, tables.size(1), splits, first,        \
                tables.dtype().bits == 64, lengths.dtype().bits == 64, starts.dtype().bits == 64,  \
                tiles, max_query_len)
  if (h == 24 && hk == 4) {
    if (grouped) {
      ATTENTION(true, true, 1);
    } else {
      if (q.size(0) == 4) {
        ATTENTION(true, false, 2);
      } else {
        ATTENTION(true, false, 1);
      }
    }
  } else {
    if (grouped) {
      ATTENTION(false, true, 1);
    } else {
      if (q.size(0) == 4) {
        ATTENTION(false, false, 2);
      } else {
        ATTENTION(false, false, 1);
      }
    }
  }
#undef ATTENTION
}
void attention_partial(TensorView q, TensorView cache, TensorView tables, TensorView lengths,
                       TensorView starts, TensorView partial, TensorView lse, int64_t first,
                       bool grouped, bool position64, int64_t max_query_len) {
  TVM_FFI_ICHECK(max_query_len >= 1 && max_query_len <= 8);
  auto stream = stream_for(q, "attention_partial");
  if (position64)
    launch_attention<int64_t>(q, cache, tables, lengths, starts, partial, lse, first, grouped,
                              max_query_len, stream);
  else
    launch_attention<int32_t>(q, cache, tables, lengths, starts, partial, lse, first, grouped,
                              max_query_len, stream);
  finish_cuda_launch(stream, "attention_partial");
}
template <int Splits>
__global__ void attention_merge_kernel(const float *__restrict__ partial,
                                       const float *__restrict__ lse,
                                       __nv_bfloat16 *__restrict__ out) {
  pdl_dependency_wait();
  pdl_launch_next();
  int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
  __shared__ float weights[Splits], scratch[8];
  float log = tid < Splits ? lse[row * Splits + tid] : -INFINITY;
  float maximum = warp_max(log);
  if (lane == 0)
    scratch[warp] = maximum;
  __syncthreads();
  maximum = warp_max(lane < 4 ? scratch[lane] : -INFINITY);
  float safe_maximum = maximum == -INFINITY ? 0.f : maximum;
  float weight = tid < Splits ? softmax_exp2(log - safe_maximum) : 0.f;
  if (tid < Splits)
    weights[tid] = weight;
  float total = warp_sum(weight);
  if (lane == 0)
    scratch[warp + 4] = total;
  __syncthreads();
  total = warp_sum(lane < 4 ? scratch[lane + 4] : 0.f);
  float2 accum = make_float2(0.f, 0.f);
#pragma unroll 128
  for (int i = 0; i < Splits; ++i) {
    float2 value = reinterpret_cast<const float2 *>(
        partial)[(static_cast<int64_t>(row) * Splits + i) * 128 + tid];
    float w = weights[i];
    accum.x += value.x * w;
    accum.y += value.y * w;
  }
  reinterpret_cast<__nv_bfloat162 *>(out)[row * 128 + tid] = __floats2bfloat162_rn(
      total > 0.f ? accum.x / total : 0.f, total > 0.f ? accum.y / total : 0.f);
}

void attention_merge(TensorView partial, TensorView lse, TensorView out) {
  auto stream = stream_for(out, "attention_merge");
#define MERGE(S)                                                                                   \
  launch_kernel(attention_merge_kernel<S>, lse.size(0) * lse.size(1), 128, 0, stream,              \
                static_cast<const float *>(partial.data_ptr()),                                    \
                static_cast<const float *>(lse.data_ptr()),                                        \
                static_cast<__nv_bfloat16 *>(out.data_ptr()))
  if (lse.size(2) == 16) {
    MERGE(16);
  } else if (lse.size(2) == 64) {
    MERGE(64);
  } else {
    TVM_FFI_ICHECK(lse.size(2) == 128);
    MERGE(128);
  }
#undef MERGE
  finish_cuda_launch(stream, "attention_merge");
}

// DSpark keeps the retained prefix in pages and the bidirectional seven-row
// noise block in separate tensors. No noise KV is written to the shared pages.
