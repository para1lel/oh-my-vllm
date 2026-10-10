#pragma once

#include "attention_math.cuh"

__device__ __forceinline__ void stage_dspark_copy(__nv_bfloat16 *destination,
                                                  const __nv_bfloat16 *source, bool valid) {
  if ((reinterpret_cast<uintptr_t>(source) & 15) == 0) {
    unsigned target = __cvta_generic_to_shared(destination);
    int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;" ::"r"(target),
                 "l"(source), "r"(bytes)
                 : "memory");
  } else {
    // A contiguous BF16 view can start at an odd storage offset. Scalar
    // staging preserves that legal layout without changing the CUDA provider.
#pragma unroll
    for (int j = 0; j < 8; ++j)
      destination[j] = valid ? source[j] : __float2bfloat16_rn(0.f);
  }
}

__device__ __forceinline__ void
stage_dspark_kv(const __nv_bfloat16 *cache, const int *tables, const __nv_bfloat16 *block_key,
                const __nv_bfloat16 *block_value, int seq, int table_width, int cache_pages,
                int base, int context, int kh, __nv_bfloat16 *key, __nv_bfloat16 *value) {
#pragma unroll
  for (int i = threadIdx.x; i < 64 * 16; i += 128) {
    int pos = base + i / 16, col = (i % 16) * 8;
    const __nv_bfloat16 *source_k = block_key, *source_v = block_value;
    bool valid = pos < context + 7;
    if (pos < context) {
      int page = tables[seq * table_width + pos / 784];
      if (page < 0 || page >= cache_pages)
        asm volatile("trap;");
      int64_t offset = ((static_cast<int64_t>(page) * 1568 + pos % 784) * 8 + kh) * 128 + col;
      source_k = cache + offset;
      source_v = source_k + 784 * 8 * 128;
    } else if (valid) {
      int64_t offset = ((static_cast<int64_t>(seq) * 7 + pos - context) * 8 + kh) * 128 + col;
      source_k = block_key + offset;
      source_v = block_value + offset;
    }
    stage_dspark_copy(key + shared_index(i / 16, col, 128), source_k, valid);
    stage_dspark_copy(value + shared_index(i / 16, col, 128), source_v, valid);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__global__ void dspark_attention_partial_kernel(
    const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ cache,
    const int *__restrict__ tables, const int *__restrict__ contexts,
    const __nv_bfloat16 *__restrict__ block_key, const __nv_bfloat16 *__restrict__ block_value,
    float *__restrict__ partial, float *__restrict__ lse, int table_width, int cache_pages,
    int splits) {
  pdl_dependency_wait();
  pdl_launch_next();
  constexpr int BQ = 32, BK = 64, Buffers = 1;
  constexpr int RowGroups = BQ / 8, ScoreTiles = BQ * BK / 512, AccTiles = BQ / 4;
  constexpr int h = 32, hk = 8;
  extern __shared__ __align__(32) unsigned char storage[];
  auto *q = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto *key_buffers = q + BQ * 128;
  auto *value_buffers = key_buffers + Buffers * BK * 128;
  auto *p = value_buffers + Buffers * BK * 128;
  auto *maxima = reinterpret_cast<float *>(p + BQ * 64);
  auto *sums = maxima + 4 * BQ;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  const int fr = lane / 4, fc = (lane % 4) * 2;
  const int seq = blockIdx.x, qbase = 0;
  const int kh = blockIdx.y, split = blockIdx.z, ratio = h / hk;
  const int start = seq * 7, end = start + 7;
  const int context = contexts[seq], last = context + 7;
  if (context < 0 || context > table_width * 784)
    asm volatile("trap;");
  const int chunk = ((last + splits * BK - 1) / (splits * BK)) * BK;
  const int begin = split * chunk;
  float acc[AccTiles][4] = {}, denominator[RowGroups] = {}, maximum[RowGroups];
#pragma unroll
  for (int rg = 0; rg < RowGroups; ++rg)
    maximum[rg] = -INFINITY;
#pragma unroll
  for (int i = threadIdx.x; i < BQ * 16; i += 128) {
    int r = i / 16, col = (i % 16) * 8;
    int row = start + (qbase + r) / ratio;
    auto *destination = q + shared_index(r, col, 128);
    int64_t offset = (static_cast<int64_t>(row) * h + kh * ratio + (qbase + r) % ratio) * 128 + col;
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
  const int limit = min(begin + chunk, last);
  if (begin < limit)
    stage_dspark_kv(cache, tables, block_key, block_value, seq, table_width, cache_pages, begin,
                    context, kh, key_buffers, value_buffers);
  int step = 0;
  for (int base = begin; base < limit; base += BK, ++step) {
    auto *k = key_buffers + (step % Buffers) * BK * 128;
    auto *v = value_buffers + (step % Buffers) * BK * 128;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if constexpr (Buffers == 2) {
      if (base + BK < limit)
        stage_dspark_kv(cache, tables, block_key, block_value, seq, table_width, cache_pages,
                        base + BK, context, kh, key_buffers, value_buffers);
    }
    float score[ScoreTiles][4] = {};
#pragma unroll 8
    for (int inner = 0; inner < 128; inner += 16) {
      unsigned qa[BQ / 16][4], kb[BK / 32][2];
#pragma unroll
      for (int rg = 0; rg < BQ / 16; ++rg)
        load_a(qa[rg], q + rg * 16 * 128, inner, lane, 128);
#pragma unroll
      for (int kt = 0; kt < BK / 32; ++kt)
        load_k(kb[kt], k, inner, kt * 4 + warp, lane, 128);
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
      int length = row < end ? last : 0;
      float m = -INFINITY;
#pragma unroll
      for (int kt = 0; kt < BK / 32; ++kt) {
        int tile = (rg / 2) * (BK / 32) + kt;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          int col = (kt * 4 + warp) * 8 + fc + j;
          float value = row < end && base + col < length
                            ? score[tile][(rg % 2) * 2 + j] * 0.12751743082459868f
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
      int row_base = (tile / 4) * 16;
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
      for (int ct = 0; ct < 4; ++ct) {
        unsigned vb[2];
        load_v(vb, v, inner, (ct * 4 + warp) * 8, lane, 128);
#pragma unroll
        for (int rg = 0; rg < BQ / 16; ++rg)
          mma_bf16(acc[rg * 4 + ct], pa[rg][0], pa[rg][1], pa[rg][2], pa[rg][3], vb[0], vb[1]);
      }
    }
    __syncthreads();
    if constexpr (Buffers == 1) {
      if (base + BK < limit)
        stage_dspark_kv(cache, tables, block_key, block_value, seq, table_width, cache_pages,
                        base + BK, context, kh, key_buffers, value_buffers);
    }
  }
#pragma unroll
  for (int rg = 0; rg < RowGroups; ++rg)
    denominator[rg] = 1.f / (denominator[rg] > 0 ? denominator[rg] : 1.f);
#pragma unroll
  for (int tile = 0; tile < AccTiles; ++tile) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int rg = (tile / 4) * 2 + j / 2;
      int r = (rg / 2) * 16 + fr + (rg % 2) * 8;
      int qr = qbase + r, row = start + qr / ratio;
      if (row < end) {
        int col = ((tile % 4) * 4 + warp) * 8 + fc + j % 2;
        int64_t dst =
            ((static_cast<int64_t>(row) * h + kh * ratio + qr % ratio) * splits + split) * 128 +
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

void dspark_attention_partial(TensorView query, TensorView cache, TensorView tables,
                              TensorView contexts, TensorView block_key, TensorView block_value,
                              TensorView partial, TensorView lse) {
  TVM_FFI_ICHECK(query.ndim() == 4 && query.size(1) == 7 && query.size(2) == 32 &&
                 query.size(3) == 128);
  TVM_FFI_ICHECK(cache.ndim() == 5 && cache.size(1) == 2 && cache.size(2) == 784 &&
                 cache.size(3) == 8 && cache.size(4) == 128);
  TVM_FFI_ICHECK(has_dtype(tables, kDLInt, 32) && has_dtype(contexts, kDLInt, 32));
  TVM_FFI_ICHECK(lse.size(2) == 16 || lse.size(2) == 64);
  auto stream = stream_for(query, "dspark_attention_partial");
  constexpr int shared_bytes = (32 * 128 + 2 * 64 * 128 + 32 * 64) * 2 + 8 * 32 * 4;
  dim3 grid(query.size(0), 8, lse.size(2));
  launch_kernel(dspark_attention_partial_kernel, grid, 128, shared_bytes, stream,
                static_cast<const __nv_bfloat16 *>(query.data_ptr()),
                static_cast<const __nv_bfloat16 *>(cache.data_ptr()),
                static_cast<const int *>(tables.data_ptr()),
                static_cast<const int *>(contexts.data_ptr()),
                static_cast<const __nv_bfloat16 *>(block_key.data_ptr()),
                static_cast<const __nv_bfloat16 *>(block_value.data_ptr()),
                static_cast<float *>(partial.data_ptr()), static_cast<float *>(lse.data_ptr()),
                tables.size(1), cache.size(0), lse.size(2));
  finish_cuda_launch(stream, "dspark_attention_partial");
}
template <int Splits>
__global__ void dspark_attention_merge_kernel(const float *__restrict__ partial,
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
    float2 value = make_float2(0.f, 0.f);
    if (tid < 64)
      value = reinterpret_cast<const float2 *>(
          partial)[(static_cast<int64_t>(row) * Splits + i) * 64 + tid];
    float w = weights[i];
    accum.x += value.x * w;
    accum.y += value.y * w;
  }
  if (tid < 64)
    reinterpret_cast<__nv_bfloat162 *>(out)[row * 64 + tid] = __floats2bfloat162_rn(
        total > 0.f ? accum.x / total : 0.f, total > 0.f ? accum.y / total : 0.f);
}

void dspark_attention_merge(TensorView partial, TensorView lse, TensorView output) {
  auto stream = stream_for(output, "dspark_attention_merge");
#define DSPARK_MERGE(S)                                                                            \
  launch_kernel(dspark_attention_merge_kernel<S>, lse.size(0) * lse.size(1), 128, 0, stream,       \
                static_cast<const float *>(partial.data_ptr()),                                    \
                static_cast<const float *>(lse.data_ptr()),                                        \
                static_cast<__nv_bfloat16 *>(output.data_ptr()))
  if (lse.size(2) == 16) {
    DSPARK_MERGE(16);
  } else {
    TVM_FFI_ICHECK(lse.size(2) == 64);
    DSPARK_MERGE(64);
  }
#undef DSPARK_MERGE
  finish_cuda_launch(stream, "dspark_attention_merge");
}
