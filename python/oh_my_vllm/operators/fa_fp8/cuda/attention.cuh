#pragma once

#include "format.cuh"
#include "../../fa/cuda/attention/fa_attention.cuh"

namespace fa_fp8 {
template <int BQ>
__device__ inline void load_kv(const uint8_t *codes, const float *scales, const int32_t *tables,
                               int seq, int width, int pages, int base, int last, int head,
                               __nv_bfloat16 *keys, __nv_bfloat16 *values) {
  for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
    int row = i / 16, dim = i % 16 * 16, pos = base + row;
    alignas(16) __nv_bfloat16 key[16], value[16];
    if (pos < last) {
      int logical = pos / PageRows;
      if (logical < 0 || logical >= width) {
        asm volatile("trap;");
        return;
      }
      int page = tables[int64_t(seq) * width + logical];
      if (page <= 0 || page >= pages) {
        asm volatile("trap;");
        return;
      }
      int64_t index = scale_index(page, 0, pos % PageRows, head);
      uint4 key_codes = *reinterpret_cast<const uint4 *>(codes + index * Width + dim);
      uint4 value_codes =
          *reinterpret_cast<const uint4 *>(codes + (index + PageRows * Heads) * Width + dim);
      const auto *key_words = reinterpret_cast<const uint32_t *>(&key_codes);
      const auto *value_words = reinterpret_cast<const uint32_t *>(&value_codes);
      float key_scale = scales[index], value_scale = scales[index + PageRows * Heads];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        key[j] = decode(uint8_t(key_words[j / 4] >> (j % 4 * 8)), key_scale);
        value[j] = decode(uint8_t(value_words[j / 4] >> (j % 4 * 8)), value_scale);
      }
    } else {
#pragma unroll
      for (int j = 0; j < 16; ++j)
        key[j] = value[j] = __float2bfloat16_rn(0.f);
    }
#pragma unroll
    for (int group = 0; group < 2; ++group) {
      int offset = group * 8;
      *reinterpret_cast<uint4 *>(keys + fa_kv_index<true, 64>(row, dim + offset)) =
          *reinterpret_cast<const uint4 *>(key + offset);
      *reinterpret_cast<uint4 *>(values + fa_kv_index<true, 64>(row, dim + offset)) =
          *reinterpret_cast<const uint4 *>(value + offset);
    }
  }
  __syncthreads();
}

template <int BQ>
__device__ inline bool
tiles(const uint8_t *codes, const float *scales, const int32_t *tables, const int32_t *lengths,
      __nv_bfloat16 *q, __nv_bfloat16 *keys, __nv_bfloat16 *values, __nv_bfloat16 *p, float *maxima,
      float *sums, float (&acc)[BQ / 2][4], float (&denominator)[BQ / 8], float (&maximum)[BQ / 8],
      int seq, int width, int pages, int begin, int limit, int last, int head, int start, int end,
      int qbase, int warp, int lane) {
  int fr = lane / 4, fc = lane % 4 * 2;
  for (int base = begin; base < limit; base += 64) {
    load_kv<BQ>(codes, scales, tables, seq, width, pages, base, last, head, keys, values);
    float score[BQ * 64 / 512][4] = {}, alpha[BQ / 8];
    fa_partial_scores<BQ>(q, keys, score, warp, lane);
    if (!fa_partial_probabilities<BQ>(score, p, maxima, sums, maximum, alpha, lengths, start, end,
                                      qbase, base, last, fr, fc, warp, lane))
      return false;
#pragma unroll
    for (int rg = 0; rg < BQ / 8; ++rg) {
      int r = rg / 2 * 16 + fr + rg % 2 * 8;
      denominator[rg] *= alpha[rg];
#pragma unroll
      for (int w = 0; w < 4; ++w)
        denominator[rg] += sums[w * BQ + r];
    }
#pragma unroll
    for (int tile = 0; tile < BQ / 2; ++tile)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[tile][j] *= alpha[tile / 8 * 2 + j / 2];
#pragma unroll
    for (int inner = 0; inner < 64; inner += 16) {
      unsigned pa[BQ / 16][4];
#pragma unroll
      for (int rg = 0; rg < BQ / 16; ++rg)
        fa_load_a(pa[rg], p + rg * 16 * 64, inner, lane, 64);
#pragma unroll
      for (int ct = 0; ct < 8; ++ct) {
        unsigned vb[2];
        fa_load_v<true, 64>(vb, values, inner, (ct * 4 + warp) * 8, lane);
#pragma unroll
        for (int rg = 0; rg < BQ / 16; ++rg)
          fa_mma_bf16(acc[rg * 8 + ct], pa[rg][0], pa[rg][1], pa[rg][2], pa[rg][3], vb[0], vb[1]);
      }
    }
    __syncthreads();
  }
  return true;
}

template <int BQ>
__global__ void partial(const __nv_bfloat16 *query, const uint8_t *codes, const float *scales,
                        const int32_t *tables, const int32_t *lengths, const int32_t *starts,
                        float *out, float *lse, int width, int splits, int first, int query_tiles,
                        int max_queries, int rows, int pages) {
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *q = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto *keys = q + BQ * 256, *values = keys + 64 * 256, *p = values + 64 * 256;
  auto *maxima = reinterpret_cast<float *>(p + BQ * 64), *sums = maxima + 4 * BQ;
  int seq = blockIdx.x / query_tiles, qbase = blockIdx.x % query_tiles * BQ;
  int head = blockIdx.y, split = blockIdx.z, start = starts[seq], end = starts[seq + 1];
  if (start < 0 || end > rows || end <= start || end - start > max_queries || starts[0] != 0 ||
      starts[gridDim.x / query_tiles] != rows) {
    asm volatile("trap;");
    return;
  }
  if (start + qbase / 6 >= end)
    return;
  int last = lengths[min(end - 1, start + (qbase + BQ - 1) / 6)];
  if (last < first || last > width * PageRows) {
    asm volatile("trap;");
    return;
  }
  int chunk = ((max(last - first, 0) + splits * 64 - 1) / (splits * 64)) * 64;
  int begin = first + split * chunk, limit = min(begin + chunk, last);
  float acc[BQ / 2][4] = {}, denominator[BQ / 8] = {}, maximum[BQ / 8];
#pragma unroll
  for (int rg = 0; rg < BQ / 8; ++rg)
    maximum[rg] = -INFINITY;
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  fa_partial_load_query<BQ>(query, q, start, end, qbase, head);
  __syncthreads();
  if (!tiles<BQ>(codes, scales, tables, lengths, q, keys, values, p, maxima, sums, acc, denominator,
                 maximum, seq, width, pages, begin, limit, last, head, start, end, qbase, warp,
                 lane))
    return;
  fa_partial_store<BQ>(out, lse, acc, denominator, maximum, start, end, qbase, head, splits, split,
                       lane / 4, lane % 4 * 2, warp, lane);
}

inline void attend(TensorView query, TensorView codes, TensorView scales, TensorView tables,
                   TensorView lengths, TensorView starts, TensorView out, TensorView lse,
                   int max_queries, int first, cudaStream_t stream) {
  auto launch = [&](auto tile) {
    constexpr int BQ = decltype(tile)::value;
    constexpr int Bytes = (BQ * 256 + 2 * 64 * 256 + BQ * 64) * 2 + 8 * BQ * 4;
    static std::once_flag configured;
    std::call_once(configured, [] {
      check_cuda(
          cudaFuncSetAttribute(partial<BQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes));
    });
    int tiles = (max_queries * 6 + BQ - 1) / BQ;
    partial<BQ><<<dim3(tables.size(0) * tiles, 4, lse.size(2)), 128, Bytes, stream>>>(
        static_cast<const __nv_bfloat16 *>(query.data_ptr()),
        static_cast<const uint8_t *>(codes.data_ptr()),
        static_cast<const float *>(scales.data_ptr()),
        static_cast<const int32_t *>(tables.data_ptr()),
        static_cast<const int32_t *>(lengths.data_ptr()),
        static_cast<const int32_t *>(starts.data_ptr()), static_cast<float *>(out.data_ptr()),
        static_cast<float *>(lse.data_ptr()), tables.size(1), lse.size(2), first, tiles,
        max_queries, query.size(0), codes.size(0));
  };
  if (max_queries == 1)
    launch(std::integral_constant<int, 16>{});
  else
    launch(std::integral_constant<int, 32>{});
  check_cuda(cudaGetLastError());
}
} // namespace fa_fp8
