#pragma once
#include "fa_decode_cache.cuh"
#include "fa_input.cuh"
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>

template <int W> constexpr int fa_native_storage() { return 16384 + 32768 + W * 8 * 64 * 4; }

__device__ inline unsigned fa_pack4(float a, float b, float c, float d) {
  unsigned short x, y;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(x) : "f"(a), "f"(b));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(y) : "f"(c), "f"(d));
  return unsigned(x) | (unsigned(y) << 16);
}

template <int N> __device__ inline void fa_native_load(uint32_t address, float (&value)[N]) {
  unsigned w[N];
  if constexpr (N == 8)
    cute::SM100_TMEM_LOAD_32dp32b8x::copy(address, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
  if constexpr (N == 16)
    cute::SM100_TMEM_LOAD_32dp32b16x::copy(address, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7],
                                           w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15]);
  if constexpr (N == 32)
    cute::SM100_TMEM_LOAD_32dp32b32x::copy(address, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7],
                                           w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15],
                                           w[16], w[17], w[18], w[19], w[20], w[21], w[22], w[23],
                                           w[24], w[25], w[26], w[27], w[28], w[29], w[30], w[31]);
#pragma unroll
  for (int i = 0; i < N; ++i)
    value[i] = __uint_as_float(w[i]);
}

template <int N, size_t... I>
__device__ inline void fa_native_store(uint32_t address, const float (&value)[N],
                                       std::index_sequence<I...>) {
  if constexpr (N == 8)
    cute::SM100_TMEM_STORE_32dp32b8x::copy(__float_as_uint(value[I])..., address);
  if constexpr (N == 16)
    cute::SM100_TMEM_STORE_32dp32b16x::copy(__float_as_uint(value[I])..., address);
  if constexpr (N == 32)
    cute::SM100_TMEM_STORE_32dp32b32x::copy(__float_as_uint(value[I])..., address);
}

template <int W>
__global__ __launch_bounds__(512, 1) void fa_native_fp8_kernel(
    const __nv_bfloat16 *__restrict__ original_query, const __nv_fp8_e4m3 *__restrict__ query,
    const float *__restrict__ query_scales, const __nv_fp8_e4m3 *__restrict__ cache,
    const float *__restrict__ cache_scales, const int32_t *__restrict__ tables,
    const int32_t *__restrict__ lengths, const int32_t *__restrict__ starts,
    float *__restrict__ partial, float *__restrict__ lse, int table_width, int splits, int first,
    int max_queries, int query_rows, int64_t pages, const __grid_constant__ CUtensorMap map,
    const __grid_constant__ CUtensorMap scale_map) {
  using namespace cute;
  constexpr int N = W * 8;
  using E = cutlass::float_e4m3_t;
  using QKMma = decltype(make_tiled_mma(
      SM100_MMA_F8F6F4_SS<E, E, float, 64, N, UMMA::Major::K, UMMA::Major::K>{}));
  using PVMma = decltype(make_tiled_mma(
      SM100_MMA_F8F6F4_SS<E, E, float, 128, N, UMMA::Major::MN, UMMA::Major::K>{}));
  using QLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<E>{}, partition_shape_B(QKMma{}, Shape<Int<N>, _256>{})));
  using KLayout = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<E>{},
                                                   partition_shape_A(QKMma{}, Shape<_64, _256>{})));
  using PLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW64_Atom<E>{}, partition_shape_B(PVMma{}, Shape<Int<N>, _64>{})));
  using VLayout = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_MN_SW128_Atom<E>{},
                                                   partition_shape_A(PVMma{}, Shape<_128, _64>{})));
  static_assert(cosize_v<QLayout> <= 8192 && cosize_v<KLayout> == 16384 &&
                cosize_v<PLayout> <= 4096 && cosize_v<VLayout> == 8192);
  extern __shared__ __align__(1024) unsigned char native_storage[];
  int group = threadIdx.x / 128;
  auto *storage = native_storage + group * fa_native_storage<W>();
  auto sQ = make_tensor(make_smem_ptr(reinterpret_cast<E *>(storage)), QLayout{});
  auto sP = make_tensor(make_smem_ptr(reinterpret_cast<E *>(storage + 8192)), PLayout{});
  auto *qs = reinterpret_cast<float *>(storage + 12288);
  auto *ps = qs + 32, *alphas = ps + 32, *inverses = alphas + 32;
  auto *ready = reinterpret_cast<uint64_t *>(inverses + 32);
  auto *tmem = reinterpret_cast<uint32_t *>(native_storage + 12832);
  auto *encoded = storage + 16384;
  auto *vs = reinterpret_cast<float *>(encoded - 512) + (first & 3);
  auto *scores = reinterpret_cast<float *>(encoded + 32768);
  auto *flags = reinterpret_cast<unsigned *>(storage + 12928);
  int tid = threadIdx.x % 128, warp = tid / 32, lane = tid % 32;
  int sequence = blockIdx.x, head = blockIdx.y, split = blockIdx.z * 4 + group;
  int start = starts[sequence], end = starts[sequence + 1];
  if (start < 0 || end > query_rows || end <= start || end - start > max_queries ||
      starts[0] != 0 || starts[gridDim.x] != query_rows) {
    asm volatile("trap;");
    return;
  }
  int last = lengths[end - 1];
  if (first < 0 || last < first || int64_t(last) > int64_t(table_width) * 784) {
    asm volatile("trap;");
    return;
  }
  int qr = tid / 4, row = start + qr / 6;
  int row_length = qr < N && row < end ? lengths[row] : 0;
  if (row_length < 0 || row_length > last || (qr < N && row < end && row_length < lengths[start])) {
    asm volatile("trap;");
    return;
  }
  int chunk = ((last - first + splits * 64 - 1) / (splits * 64)) * 64,
      begin = first + split * chunk, limit = min(begin + chunk, last);
  TMEM::Allocator1Sm allocator;
  if (group == 0 && warp == 0) {
    allocator.allocate(N * 16, tmem);
    allocator.release_allocation_lock();
  }
  if (tid == 0) {
    for (int i = 0; i < 4; ++i)
      initialize_barrier(ready[i], 1);
    cutlass::arch::fence_barrier_init();
  }
  float maximum = -INFINITY, denominator = 0;
  float key_scale = 0.f;
  if (begin < limit) {
    int page = tables[sequence * table_width + begin / 784];
    if (page < 0 || page >= pages) {
      asm volatile("trap;");
      return;
    }
    key_scale = cache_scales[(int64_t(page) * 8 + head) * 784 + begin % 784];
  }
  if (qr < N) {
    int64_t qo = (int64_t(row) * 24 + head * 6 + qr % 6) * 256;
    if constexpr (W == 1) {
      float m = 0;
      for (int c = tid % 4; c < 256; c += 4)
        m = fmaxf(m, row < end ? fabsf(__bfloat162float(original_query[qo + c])) : 0.f);
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
      float scale = fmaxf(m, 1e-10f) / 448.f;
      float inverse_scale = __frcp_rn(scale);
      if (tid % 4 == 0)
        qs[qr] = scale * key_scale * 0.09016844005556021f;
      for (int c = (tid % 4) * 4; c < 256; c += 16) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          v[j] = row < end ? fa_scale_operand(__bfloat162float(original_query[qo + c + j]), scale,
                                              inverse_scale)
                           : 0.f;
        *reinterpret_cast<unsigned *>(&sQ(make_coord(qr, c % 32), 0, c / 32)) =
            fa_pack4(v[0], v[1], v[2], v[3]);
      }
    } else {
      float scale = row < end ? query_scales[qo / 256] : 0.f;
      if (tid % 4 == 0)
        qs[qr] = scale * key_scale * 0.09016844005556021f;
      for (int c = (tid % 4) * 4; c < 256; c += 16)
        *reinterpret_cast<unsigned *>(&sQ(make_coord(qr, c % 32), 0, c / 32)) =
            row < end ? *reinterpret_cast<const unsigned *>(query + qo + c) : 0u;
    }
  }
  int common_limit = min(limit, lengths[start]);
  float max_value_scale = 0.f;
  for (int position = begin + tid; position < common_limit; position += 128) {
    int page = tables[sequence * table_width + position / 784];
    if (page < 0 || page >= pages) {
      asm volatile("trap;");
      return;
    }
    int64_t offset = ((int64_t(page) * 2 + 1) * 4 + head) * 784 + position % 784;
    max_value_scale = fmaxf(max_value_scale, cache_scales[offset]);
  }
  unsigned maximum_bits = __reduce_max_sync(0xffffffff, __float_as_uint(max_value_scale));
  if (lane == 0)
    flags[warp] = maximum_bits;
  fa_native_group_sync();
  maximum_bits = max(max(flags[0], flags[1]), max(flags[2], flags[3]));
  float local_maximum = __uint_as_float(maximum_bits);
  if (qr < N && row < end) {
    for (int position = max(begin, common_limit); position < min(limit, row_length); ++position) {
      int page = tables[sequence * table_width + position / 784];
      if (page < 0 || page >= pages) {
        asm volatile("trap;");
        return;
      }
      local_maximum = fmaxf(
          local_maximum, cache_scales[((int64_t(page) * 2 + 1) * 4 + head) * 784 + position % 784]);
    }
  }
  float probability_scale = fmaxf(local_maximum, 1e-20f) / 448.f;
  float probability_inverse = __frcp_rn(probability_scale);
  if (qr < N && tid % 4 == 0)
    ps[qr] = probability_scale;
  cutlass::arch::fence_view_async_shared();
  __syncthreads();
  if (begin < limit) {
    fa_stage_encoded_kv<0>(cache, cache_scales, tables, sequence, table_width, pages, begin, last,
                           head, map, scale_map, ready, encoded);
    fa_stage_encoded_kv<1>(cache, cache_scales, tables, sequence, table_width, pages, begin, last,
                           head, map, scale_map, ready + 1, encoded);
  }
  QKMma qk;
  PVMma pv;
  auto q_fragment = qk.get_slice(0).make_fragment_B(sQ);
  auto p_fragment = pv.get_slice(0).make_fragment_B(sP);
  auto sK = make_tensor(make_smem_ptr(reinterpret_cast<E *>(encoded)), KLayout{});
  auto k_fragment = qk.get_slice(0).make_fragment_A(sK);
  auto score_fragment = partition_fragment_C(qk, Shape<_64, Int<N>>{});
  auto output_fragment = partition_fragment_C(pv, Shape<_128, Int<N>>{});
  uint32_t group_tmem = *tmem + group * 4 * N;
  uint32_t address = group_tmem + uint32_t(warp * 32) * 65536u;
  for (int base = begin, iteration = 0; base < limit; base += 64, ++iteration) {
    wait_barrier(ready[0], iteration & 1);
    cutlass::arch::fence_view_async_shared();
    fa_native_group_sync();
    if (warp == 0) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      score_fragment.data() = group_tmem;
      qk.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        gemm(qk, k_fragment(_, _, k), q_fragment(_, _, k), score_fragment);
        qk.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(ready + 2);
    }
    wait_barrier(ready[2], iteration & 1);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    float dot[N];
    fa_native_load(address, dot);
    if (lane < 16) {
      int token = warp * 16 + lane;
#pragma unroll
      for (int i = 0; i < N; ++i)
        scores[((token >> 4) * 4 + (token & 3)) * N * 4 + (i ^ (token & 3)) * 4 +
               ((token >> 2) & 3)] = dot[i] * qs[i];
    }
    cutlass::arch::fence_view_async_tmem_load();
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    fa_native_group_sync();
    if (base + 64 < limit)
      fa_stage_encoded_kv<0>(cache, cache_scales, tables, sequence, table_width, pages, base + 64,
                             last, head, map, scale_map, ready, encoded);
    wait_barrier(ready[1], iteration & 1);
    float row_alpha = 1.f;
    if (qr < N) {
      float probabilities[16], m = maximum;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        int token = tid % 4 * 4 + i / 4 * 16 + i % 4;
        float x = row < end && base + token < row_length
                      ? scores[((token >> 4) * 4 + (token & 3)) * N * 4 + (qr ^ (token & 3)) * 4 +
                               ((token >> 2) & 3)]
                      : -INFINITY;
        probabilities[i] = x;
        m = fmaxf(m, x);
      }
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
      float safe = m == -INFINITY ? 0.f : m, alpha = fa_exp2(maximum - safe);
      maximum = m;
      row_alpha = alpha;
      float total = 0;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        int token = tid % 4 * 4 + i / 4 * 16 + i % 4;
        float x = fa_exp2(probabilities[i] - safe);
        total += x;
        probabilities[i] = x * vs[token];
      }
      total += __shfl_xor_sync(0xffffffff, total, 1);
      total += __shfl_xor_sync(0xffffffff, total, 2);
      denominator = denominator * alpha + total;
      float inverse_scale = probability_inverse;
      if (tid % 4 == 0)
        alphas[qr] = alpha;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int c = tid % 4 * 4 + i * 16;
        *reinterpret_cast<unsigned *>(&sP(make_coord(qr, c % 32), 0, c / 32)) =
            fa_pack4(__fmul_rn(probabilities[i * 4], inverse_scale),
                     __fmul_rn(probabilities[i * 4 + 1], inverse_scale),
                     __fmul_rn(probabilities[i * 4 + 2], inverse_scale),
                     __fmul_rn(probabilities[i * 4 + 3], inverse_scale));
      }
    }
    unsigned changed = __ballot_sync(0xffffffff, qr < N && row < end && row_alpha != 1.f);
    if (lane == 0)
      flags[warp] = changed;
    cutlass::arch::fence_view_async_shared();
    fa_native_group_sync();
    if (iteration > 0 && (flags[0] | flags[1] | flags[2] | flags[3])) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
      for (int part = 0; part < 2; ++part) {
        float values[N];
        fa_native_load(address + N * (part + 1), values);
        cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
        for (int i = 0; i < N; ++i)
          values[i] *= alphas[i];
        fa_native_store(address + N * (part + 1), values, std::make_index_sequence<N>{});
      }
      cutlass::arch::fence_view_async_tmem_store();
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      fa_native_group_sync();
    }
    if (warp == 0) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
      for (int part = 0; part < 2; ++part) {
        auto sV = make_tensor(make_smem_ptr(reinterpret_cast<E *>(encoded + 16384 + part * 8192)),
                              VLayout{});
        auto v_fragment = pv.get_slice(0).make_fragment_A(sV);
        output_fragment.data() = group_tmem + N * (part + 1);
        pv.accumulate_ = iteration > 0 ? UMMA::ScaleOut::One : UMMA::ScaleOut::Zero;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          gemm(pv, v_fragment(_, _, k), p_fragment(_, _, k), output_fragment);
          pv.accumulate_ = UMMA::ScaleOut::One;
        }
      }
      cutlass::arch::umma_arrive(ready + 3);
    }
    wait_barrier(ready[3], iteration & 1);
    if (base + 64 < limit)
      fa_stage_encoded_kv<1>(cache, cache_scales, tables, sequence, table_width, pages, base + 64,
                             last, head, map, scale_map, ready + 1, encoded);
  }
  if (qr < N && tid % 4 == 0) {
    float inv = 1.f / (denominator > 0 ? denominator : 1.f);
    inverses[qr] = inv;
    if (row < end)
      lse[(int64_t(row) * 24 + head * 6 + qr % 6) * splits + split] = maximum - log2f(inv);
  }
  fa_native_group_sync();
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    float values[N] = {};
    if (begin < limit) {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      fa_native_load(address + N * (part + 1), values);
      cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
      for (int i = 0; i < N; ++i)
        values[i] *= ps[i];
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      int r = start + i / 6;
      if (r < end)
        partial[((int64_t(r) * 24 + head * 6 + i % 6) * splits + split) * 256 + part * 128 + tid] =
            values[i] * inverses[i];
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (group == 0 && warp == 0)
    allocator.free(*tmem, N * 16);
}
