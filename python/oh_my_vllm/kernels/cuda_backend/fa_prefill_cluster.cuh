#pragma once

#include "fa_prefill.cuh"
#include <utility>

inline constexpr int FAPrefillClusterSharedBytes = 199680;

template <size_t... Indices>
__device__ inline void fa_cluster_tmem_load64_impl(uint32_t address, float (&values)[64],
                                                   std::index_sequence<Indices...>) {
  uint32_t words[64];
  cute::SM100_TMEM_LOAD_32dp32b64x::copy(address, words[Indices]...);
#pragma unroll
  for (int i = 0; i < 64; ++i)
    values[i] = __uint_as_float(words[i]);
}

__device__ inline void fa_cluster_tmem_load64(uint32_t address, float (&values)[64]) {
  fa_cluster_tmem_load64_impl(address, values, std::make_index_sequence<64>{});
}

template <size_t... Indices>
__device__ inline void fa_cluster_tmem_store64_impl(uint32_t address, const float (&values)[64],
                                                    std::index_sequence<Indices...>) {
  cute::SM100_TMEM_STORE_32dp32b64x::copy(__float_as_uint(values[Indices])..., address);
}

__device__ inline void fa_cluster_tmem_store64(uint32_t address, const float (&values)[64]) {
  fa_cluster_tmem_store64_impl(address, values, std::make_index_sequence<64>{});
}

template <size_t... I>
__device__ inline void fa_cluster_pair_load_impl(uint32_t address, float (&values)[64],
                                                 std::index_sequence<I...>) {
  uint32_t words[64];
  cute::SM100_TMEM_LOAD_16dp32b64x::copy(address, words[I]...);
#pragma unroll
  for (int i = 0; i < 64; ++i)
    values[i] = __uint_as_float(words[i]);
}

__device__ inline void fa_cluster_pair_load(uint32_t a, float (&v)[64]) {
  fa_cluster_pair_load_impl(a, v, std::make_index_sequence<64>{});
}

template <size_t... I>
__device__ inline void fa_cluster_pair_store_impl(uint32_t address, const float (&values)[32],
                                                  std::index_sequence<I...>) {
  cute::SM100_TMEM_STORE_16dp32b32x::copy(__float_as_uint(values[I])..., address);
}

__device__ inline void fa_cluster_pair_store(uint32_t a, const float (&v)[32]) {
  fa_cluster_pair_store_impl(a, v, std::make_index_sequence<32>{});
}

struct FAPrefillClusterPageSpan {
  int first_logical_page;
  int first_page;
  int last_page;
};

__device__ inline FAPrefillClusterPageSpan
fa_prefill_cluster_page_span(const int32_t *tables, int seq, int table_width, int base, int last) {
  int first_logical_page = min(base, last - 1) / 784;
  int last_logical_page = min(base + 127, last - 1) / 784;
  int first_page = tables[seq * table_width + first_logical_page];
  int last_page = first_page;
  if (last_logical_page != first_logical_page)
    last_page = tables[seq * table_width + last_logical_page];
  return {first_logical_page, first_page, last_page};
}

__device__ inline bool fa_prefill_cluster_pages_valid(const FAPrefillClusterPageSpan &pages,
                                                      int cache_pages) {
  return pages.first_page >= 0 && pages.first_page < cache_pages && pages.last_page >= 0 &&
         pages.last_page < cache_pages;
}

__device__ inline int fa_prefill_cluster_cached_page(const FAPrefillClusterPageSpan &pages,
                                                     int logical) {
  return logical / 784 == pages.first_logical_page ? pages.first_page : pages.last_page;
}

template <bool ExportBFloat16, bool Decouple>
__global__ __launch_bounds__(Decouple ? 480 : 448, 1) void fa_prefill_cluster_kernel(
    const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ cache,
    const __nv_fp8_e4m3 *__restrict__ encoded_cache, const float *__restrict__ cache_scales,
    const int32_t *__restrict__ tables, const int32_t *__restrict__ lengths,
    const int32_t *__restrict__ starts, float *__restrict__ partial, float *__restrict__ lse,
    int table_width, int tiles, int max_queries, int query_rows, int cache_pages, int first,
    const __grid_constant__ CUtensorMap key_map, const __grid_constant__ CUtensorMap value_map,
    const __grid_constant__ CUtensorMap value16_map, __nv_bfloat16 *attended) {
  using namespace cute;
  using Element = cutlass::bfloat16_t;
  using QKElement = cutlass::float_e4m3_t;
  using Barrier = cutlass::arch::ClusterBarrier;
  using QKMma =
      decltype(make_tiled_mma(SM100_MMA_F8F6F4_2x1SM_SS<QKElement, QKElement, float, 256, 128,
                                                        UMMA::Major::K, UMMA::Major::K>{}));
  using PVMma =
      decltype(make_tiled_mma(SM100_MMA_F16BF16_2x1SM_TS<Element, Element, float, 256, 256,
                                                         UMMA::Major::K, UMMA::Major::MN>{}));
  using QLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<QKElement>{}, partition_shape_A(QKMma{}, Shape<_256, _256>{})));
  using KLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<QKElement>{}, partition_shape_B(QKMma{}, Shape<_128, _256>{})));
  using PLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_A(PVMma{}, Shape<_256, _128>{})));
  using VLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_MN_SW128_Atom<Element>{}, partition_shape_B(PVMma{}, Shape<_256, _128>{})));
  static_assert((cosize_v<QLayout> + 2 * cosize_v<KLayout>)*sizeof(QKElement) +
                    2 * cosize_v<VLayout> * sizeof(Element) + 512 * sizeof(float) +
                    21 * sizeof(uint64_t) + sizeof(uint32_t) <=
                FAPrefillClusterSharedBytes);
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *query_storage = reinterpret_cast<QKElement *>(storage);
  auto *key_storage = query_storage + cosize_v<QLayout>;
  auto *value_storage = reinterpret_cast<Element *>(key_storage + 2 * cosize_v<KLayout>);
  auto *probability_storage = value_storage + 2 * cosize_v<VLayout>;
  auto *alpha_storage = reinterpret_cast<float *>(probability_storage);
  auto *inverse_storage = alpha_storage + 256;
  auto *query_scales = inverse_storage + 128;
  auto *barriers = reinterpret_cast<uint64_t *>(query_scales + 128);
  auto *key_ready = barriers;
  auto *value_ready = barriers + 2;
  auto *consumed = barriers + 4;
  auto *score_ready = barriers + 6;
  auto *score_released = barriers + 7;
  auto *key_consumed = barriers + 18;
  auto *remote_score_released = barriers + 20;
  auto *score_free = barriers + 8;
  auto *probability_ready = barriers + 10;
  auto *corrected = barriers + 12;
  auto *tmem_base = reinterpret_cast<uint32_t *>(barriers + (Decouple ? 21 : 18));
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  int rank = blockIdx.x % 2, cluster = blockIdx.x / 2;
  int seq = cluster / tiles, cluster_qbase = (cluster % tiles) * 256;
  int qbase = cluster_qbase + rank * 128, kh = blockIdx.y;
  auto *remote_key_ready = barriers + 14;
  auto *remote_pv_ready = barriers + 16;
  int start = starts[seq], end = starts[seq + 1];
  if (start < 0 || end > query_rows || end <= start || end - start > max_queries ||
      starts[0] != 0 || starts[gridDim.x / (2 * tiles)] != query_rows) {
    asm volatile("trap;");
    return;
  }
  if (start + cluster_qbase / 6 >= end)
    return;
  int last = lengths[min(end - 1, start + (cluster_qbase + 255) / 6)];
  if (last < first || last > table_width * 784) {
    asm volatile("trap;");
    return;
  }
  int length = 0;
  if (tid < 256) {
    int row = start + (qbase + (warp % 4) * 32 + (warp / 4) * 16 + lane % 16) / 6;
    length = row < end ? lengths[row] : 0;
    if (length < 0 || length > last) {
      asm volatile("trap;");
      return;
    }
  }
  TMEM::Allocator2Sm allocator;
  if (warp == 13)
    allocator.allocate(512, tmem_base);
  if (tid == 0) {
    for (int index = 0; index < (Decouple ? 21 : 18); ++index)
      initialize_barrier(barriers[index], (Decouple && index == 7) || index == 10 || index == 11
                                              ? 256
                                          : index >= 8 && index < 14 ? 128
                                                                     : 1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  cute::cluster_sync();
  auto sQ = make_tensor(make_smem_ptr(query_storage), QLayout{});
  if (tid < 128) {
    for (int r = tid / 4; r < 128; r += 32) {
      int row = start + (qbase + r) / 6, head = kh * 6 + (qbase + r) % 6;
      int64_t offset = (int64_t(row) * 24 + head) * 256;
      float maximum = 0.f;
      for (int c = tid % 4; c < 256; c += 4)
        maximum = fmaxf(maximum, row < end ? fabsf(__bfloat162float(query[offset + c])) : 0.f);
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2));
      float scale = fmaxf(maximum, 1e-10f) / 448.f;
      float inverse = __frcp_rn(scale);
      if (tid % 4 == 0)
        query_scales[r] = scale;
      for (int c = tid % 4 * 4; c < 256; c += 16) {
        float value[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          value[j] = row < end
                         ? fa_scale_operand(__bfloat162float(query[offset + c + j]), scale, inverse)
                         : 0.f;
        *reinterpret_cast<unsigned *>(&sQ(make_coord(r, c % 32), 0, c / 32)) =
            fa_pack4(value[0], value[1], value[2], value[3]);
      }
    }
    cutlass::arch::fence_view_async_shared();
  } else if (tid >= 256 && tid < 384) {
    uint32_t address = *tmem_base + uint32_t((tid - 256) / 32 * 32) * 65536u + 256;
    float zeros[32] = {};
#pragma unroll
    for (int c = 0; c < 256; c += 32)
      fa_tmem_store32(address + c, zeros);
    cutlass::arch::fence_view_async_tmem_store();
    fa_prefill_before_sync();
  }
  __syncthreads();
  fa_prefill_after_sync();
  int trips = (last - first + 127) / 128;
  if (warp == 12 || (Decouple && warp == 14)) {
    for (int tile = 0, base = first; tile < trips; ++tile, base += 128) {
      int slot = tile & 1, phase = (tile / 2) & 1;
      auto *keys = key_storage + slot * cosize_v<KLayout>;
      auto *values = value_storage + slot * cosize_v<VLayout>;
#pragma unroll
      for (int operand = (Decouple && warp == 14); operand < (Decouple ? (warp == 14) + 1 : 2);
           ++operand) {
        bool is_value = operand != 0;
        int operand_base = base + (is_value ? 0 : rank * 64);
        int token_count = is_value ? 128 : 64;
        bool tma = operand_base % 16 == 0 && operand_base + token_count <= last &&
                   operand_base % 784 + token_count <= 784;
        auto pages = fa_prefill_cluster_page_span(tables, seq, table_width, operand_base, last);
        if (!fa_prefill_cluster_pages_valid(pages, cache_pages)) {
          asm volatile("trap;");
          return;
        }
        int page = pages.first_page;
        if (tile >= 2)
          wait_barrier(is_value   ? consumed[slot]
                       : Decouple ? key_consumed[slot]
                                  : score_ready[slot],
                       phase ^ 1);
        __syncwarp();
        auto *signal = is_value ? value_ready : key_ready;
        if (!is_value) {
          if (tma) {
            if (lane == 0) {
              cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(signal + slot, 16384);
              SM90_TMA_LOAD_5D::copy(&key_map, signal + slot,
                                     uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), keys, 0,
                                     operand_base % 784, 0, kh, page * 2);
            }
          } else {
            auto sK = make_tensor(make_smem_ptr(keys), KLayout{});
            for (int i = lane; i < 1024; i += 32) {
              int r = i / 16, c = (i % 16) * 16, pos = operand_base + r;
              int logical = min(pos, last - 1), p = fa_prefill_cluster_cached_page(pages, logical);
              auto *source =
                  encoded_cache + ((int64_t(p) * 1568 + logical % 784) * 4 + kh) * 256 + c;
              unsigned destination =
                  __cvta_generic_to_shared(&sK(make_coord(r, c % 32), 0, c / 32));
              int bytes = pos < last ? 16 : 0;
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(destination),
                           "l"(source), "r"(bytes)
                           : "memory");
            }
            asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
            cutlass::arch::fence_view_async_shared();
            __syncwarp();
            if (lane == 0)
              Barrier::arrive(signal + slot);
          }
        } else if (tma) {
          if (lane == 0) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(signal + slot, 32768);
            SM90_TMA_LOAD_5D::copy(&value_map, signal + slot,
                                   uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), values, 0, 0,
                                   rank * 2, kh, (page * 2 + 1) * 98 + (base % 784) / 8);
          }
        } else if (operand_base % 16 == 0 && operand_base + token_count <= last) {
          if (lane == 0) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(signal + slot, 32768);
            auto sV = make_tensor(make_smem_ptr(values), VLayout{});
            for (int r = 0; r < token_count; r += 16) {
              int logical = operand_base + r, p = fa_prefill_cluster_cached_page(pages, logical);
              auto *target = &sV(make_coord(0, r % 16), 0, r / 16);
              SM90_TMA_LOAD_5D::copy(&value16_map, signal + slot,
                                     uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), target, 0, 0,
                                     rank * 2, kh, (p * 2 + 1) * 98 + (logical % 784) / 8);
            }
          }
        } else {
          auto sV = make_tensor(make_smem_ptr(values), VLayout{});
          for (int i = lane; i < 2048; i += 32) {
            int r = i / 16, c = (i % 16) * 8, pos = operand_base + r;
            int logical = min(pos, last - 1), p = fa_prefill_cluster_cached_page(pages, logical);
            auto *source = cache + ((int64_t(p) * 1568 + logical % 784) * 4 + kh) * 256 + c +
                           int64_t(784) * 4 * 256 + rank * 128;
            unsigned destination = __cvta_generic_to_shared(&sV(make_coord(c, r % 16), 0, r / 16));
            int bytes = pos < last ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(destination),
                         "l"(source), "r"(bytes)
                         : "memory");
          }
          asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
          cutlass::arch::fence_view_async_shared();
          __syncwarp();
          if (lane == 0)
            Barrier::arrive(signal + slot);
        }
        if constexpr (Decouple) {
          if (!is_value && rank == 1) {
            wait_barrier(key_ready[slot], phase);
            fa_prefill_after_sync();
            if (lane == 0)
              Barrier::arrive(remote_key_ready + slot, 0, 1);
          }
        }
      }
    }
  } else if (warp == 13) {
    QKMma qk;
    PVMma pv;
    auto qfrag = qk.get_slice(rank).make_fragment_A(sQ);
    auto score = partition_fragment_C(qk, Shape<_256, _128>{});
    auto output = partition_fragment_C(pv, Shape<_256, _256>{});
    output.data() = *tmem_base + 256;
    auto issue_qk = [&](int tile) {
      int slot = tile & 1, phase = (tile / 2) & 1;
      wait_barrier(key_ready[slot], phase);
      wait_barrier(remote_key_ready[slot], phase);
      if constexpr (Decouple) {
        if (tile > 0) {
          wait_barrier(score_released[0], (tile - 1) & 1);
          wait_barrier(remote_score_released[0], (tile - 1) & 1);
        }
      } else if (tile >= 2) {
        wait_barrier(consumed[slot], phase ^ 1);
      }
      fa_prefill_after_sync();
      auto sK = make_tensor(make_smem_ptr(key_storage + slot * cosize_v<KLayout>), KLayout{});
      auto kfrag = qk.get_slice(rank).make_fragment_B(sK);
      score.data() = *tmem_base + (Decouple ? 0 : slot * 128);
      qk.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        gemm(qk, qfrag(_, _, k), kfrag(_, _, k), score);
        qk.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive_multicast_2x1SM(score_ready + (Decouple ? 0 : slot), 3);
    };
    if (rank == 0) {
      if (trips > 0)
        issue_qk(0);
      if constexpr (Decouple) {
        if (trips > 1)
          issue_qk(1);
      }
      for (int tile = 0; tile < trips; ++tile) {
        int slot = tile & 1, phase = (tile / 2) & 1;
        if (tile + (Decouple ? 2 : 1) < trips)
          issue_qk(tile + (Decouple ? 2 : 1));
        wait_barrier(value_ready[slot], phase);
        wait_barrier(remote_pv_ready[slot], phase);
        wait_barrier(probability_ready[slot], phase);
        wait_barrier(corrected[slot], phase);
        fa_prefill_after_sync();
        auto sP = make_tensor(make_smem_ptr(probability_storage), PLayout{});
        auto sV = make_tensor(make_smem_ptr(value_storage + slot * cosize_v<VLayout>), VLayout{});
        auto pfrag = pv.get_slice(rank).make_fragment_A(sP);
        pfrag.data() = *tmem_base + (Decouple ? 128 + slot * 64 : slot * 128 + 64);
        auto vfrag = pv.get_slice(rank).make_fragment_B(sV);
        pv.accumulate_ = UMMA::ScaleOut::One;
#pragma unroll
        for (int k = 0; k < 8; ++k)
          gemm(pv, pfrag(_, _, k), vfrag(_, _, k), output);
        cutlass::arch::umma_arrive_multicast_2x1SM(consumed + slot, 3);
      }
    } else {
      for (int tile = 0; tile < trips; ++tile) {
        int slot = tile & 1, phase = (tile / 2) & 1;
        if constexpr (!Decouple) {
          wait_barrier(key_ready[slot], phase);
          if (tile >= 2)
            wait_barrier(consumed[slot], phase ^ 1);
          if (lane == 0)
            Barrier::arrive(remote_key_ready + slot, 0, 1);
        }
        wait_barrier(value_ready[slot], phase);
        wait_barrier(probability_ready[slot], phase);
        wait_barrier(corrected[slot], phase);
        fa_prefill_after_sync();
        if (lane == 0)
          Barrier::arrive(remote_pv_ready + slot, 0, 1);
      }
    }
    if (trips > 0)
      wait_barrier(consumed[(trips - 1) & 1], ((trips - 1) / 2) & 1);
    fa_prefill_after_sync();
  } else if (tid < 256) {
    int owner = (warp % 4) * 32 + (warp / 4) * 16 + lane % 16, half = lane / 16;
    uint32_t lane_base = *tmem_base + uint32_t((warp / 4) * 16) * 65536u;
    int row = start + (qbase + owner) / 6;
    float maximum = -INFINITY, denominator = 0;
    int first_page = tables[seq * table_width + first / 784];
    if (first_page < 0 || first_page >= cache_pages) {
      asm volatile("trap;");
      return;
    }
    float score_scale = query_scales[owner] *
                        cache_scales[(int64_t(first_page) * 8 + kh) * 784 + first % 784] *
                        0.09016844005556021f;
    for (int tile = 0, base = first; tile < trips; ++tile, base += 128) {
      int slot = tile & 1, phase = (tile / 2) & 1;
      wait_barrier(score_ready[Decouple ? 0 : slot], Decouple ? tile & 1 : phase);
      fa_prefill_after_sync();
      float values[64], next_max = maximum;
      float maxima[4] = {maximum, maximum, maximum, maximum};
      fa_cluster_pair_load(lane_base + (Decouple ? 0 : slot * 128), values);
      cutlass::arch::fence_view_async_tmem_load();
      if constexpr (Decouple) {
        fa_prefill_before_sync();
        Barrier::arrive(score_released);
        if (rank == 1 && warp == 0) {
          wait_barrier(score_released[0], tile & 1);
          if (lane == 0)
            Barrier::arrive(remote_score_released, 0, 1);
        }
        if (warp == 0 && lane == 0)
          Barrier::arrive(key_consumed + slot);
      }
      bool complete = row < end && base + 128 <= length;
      if (complete) {
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          float2 scaled = __fmul2_rn(make_float2(values[2 * j], values[2 * j + 1]),
                                     make_float2(score_scale, score_scale));
          values[2 * j] = scaled.x;
          values[2 * j + 1] = scaled.y;
          maxima[(2 * j) % 4] = fmaxf(maxima[(2 * j) % 4], scaled.x);
          maxima[(2 * j + 1) % 4] = fmaxf(maxima[(2 * j + 1) % 4], scaled.y);
        }
      } else {
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          float2 scaled = __fmul2_rn(make_float2(values[2 * j], values[2 * j + 1]),
                                     make_float2(score_scale, score_scale));
          float first = row < end && base + half * 64 + 2 * j < length ? scaled.x : -INFINITY;
          float second = row < end && base + half * 64 + 2 * j + 1 < length ? scaled.y : -INFINITY;
          values[2 * j] = first;
          values[2 * j + 1] = second;
          maxima[(2 * j) % 4] = fmaxf(maxima[(2 * j) % 4], first);
          maxima[(2 * j + 1) % 4] = fmaxf(maxima[(2 * j + 1) % 4], second);
        }
      }
      next_max = fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
      next_max = fmaxf(next_max, __shfl_xor_sync(0xffffffff, next_max, 16));
      float safe = next_max == -INFINITY ? 0.f : next_max;
      float alpha = fa_exp2(maximum - safe), sum = 0;
      if constexpr (Decouple) {
        if (tile >= 2)
          wait_barrier(consumed[slot], phase ^ 1);
      }
      if (half == 0) {
        alpha_storage[slot * 128 + owner] = alpha;
        cutlass::arch::fence_view_async_shared();
        fa_prefill_before_sync();
        Barrier::arrive(score_free + slot);
      }
      float packed[32];
      float2 sums[4] = {};
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        float2 scaled =
            __fadd2_rn(make_float2(values[2 * j], values[2 * j + 1]), make_float2(-safe, -safe));
        float p0 = fa_exp2(scaled.x), p1 = fa_exp2(scaled.y);
        sums[j % 4] = __fadd2_rn(sums[j % 4], make_float2(p0, p1));
        uint32_t word;
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(word) : "f"(p0), "f"(p1));
        packed[j] = __uint_as_float(word);
      }
      float2 total = __fadd2_rn(__fadd2_rn(sums[0], sums[1]), __fadd2_rn(sums[2], sums[3]));
      sum = total.x + total.y;
      sum += __shfl_xor_sync(0xffffffff, sum, 16);
      fa_cluster_pair_store(lane_base + (Decouple ? 128 + slot * 64 : slot * 128 + 64), packed);
      cutlass::arch::fence_view_async_tmem_store();
      fa_prefill_before_sync();
      maximum = next_max;
      denominator = denominator * alpha + sum;
      Barrier::arrive(probability_ready + slot);
    }
    if (half == 0) {
      if (row < end) {
        int head = kh * 6 + (qbase + owner) % 6;
        lse[int64_t(row) * 24 + head] = maximum + log2f(denominator);
      }
      inverse_storage[owner] = denominator > 0 ? 1.f / denominator : 1.f;
    }
  } else {
    int owner = tid - 256;
    uint32_t address = *tmem_base + uint32_t(owner / 32 * 32) * 65536u + 256;
    for (int tile = 0; tile < trips; ++tile) {
      int slot = tile & 1, phase = (tile / 2) & 1;
      wait_barrier(score_free[slot], phase);
      if (tile > 0)
        wait_barrier(consumed[slot ^ 1], ((tile - 1) / 2) & 1);
      fa_prefill_after_sync();
      float alpha = alpha_storage[slot * 128 + owner];
      if (__any_sync(0xffffffff, alpha != 1.f)) {
#pragma unroll 1
        for (int c = 0; c < 256; c += 64) {
          float block[64];
          fa_cluster_tmem_load64(address + c, block);
          cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
          for (int j = 0; j < 32; ++j) {
            float2 scaled =
                __fmul2_rn(make_float2(block[2 * j], block[2 * j + 1]), make_float2(alpha, alpha));
            block[2 * j] = scaled.x;
            block[2 * j + 1] = scaled.y;
          }
          fa_cluster_tmem_store64(address + c, block);
        }
      }
      cutlass::arch::fence_view_async_tmem_store();
      fa_prefill_before_sync();
      Barrier::arrive(corrected + slot);
    }
  }
  __syncthreads();
  fa_prefill_after_sync();
  static_assert(128 * 257 * sizeof(float) <=
                (cosize_v<QLayout> + 2 * cosize_v<KLayout>)*sizeof(QKElement) +
                    2 * cosize_v<VLayout> * sizeof(Element) + 256 * sizeof(float));
  auto *staged = reinterpret_cast<float *>(storage);
  auto *packed_stage = reinterpret_cast<uint32_t *>(storage);
  if (tid >= 256 && tid < 384) {
    int owner = tid - 256;
    uint32_t address = *tmem_base + uint32_t(owner / 32 * 32) * 65536u + 256;
    float inverse = inverse_storage[owner];
    bool finite = true;
    if constexpr (ExportBFloat16) {
      int row = start + (qbase + owner) / 6, head = kh * 6 + (qbase + owner) % 6;
      finite = row < end && isfinite(lse[int64_t(row) * 24 + head]);
    }
#pragma unroll
    for (int c = 0; c < 256; c += 32) {
      float values[32];
      fa_tmem_load32(address + c, values);
      cutlass::arch::fence_view_async_tmem_load();
      if constexpr (ExportBFloat16) {
#pragma unroll
        for (int j = 0; j < 32; j += 2) {
          float x = finite ? 0.f + values[j] * inverse : 0.f;
          float y = finite ? 0.f + values[j + 1] * inverse : 0.f;
          __nv_bfloat162_raw pair = __floats2bfloat162_rn(x, y);
          packed_stage[owner * 129 + (c + j) / 2] = uint32_t(pair.x) | (uint32_t(pair.y) << 16);
        }
      } else {
#pragma unroll
        for (int j = 0; j < 32; ++j)
          staged[owner * 257 + c + j] = values[j] * inverse;
      }
    }
    fa_prefill_before_sync();
  }
  __syncthreads();
  if constexpr (ExportBFloat16) {
    for (int i = tid; i < 128 * 128; i += (Decouple ? 480 : 448)) {
      int r = i / 128, column = i % 128, target_row = start + (qbase + r) / 6;
      int head = kh * 6 + (qbase + r) % 6;
      if (target_row < end) {
        int64_t offset = (int64_t(target_row) * 24 + head) * 128 + column;
        reinterpret_cast<uint32_t *>(attended)[offset] = packed_stage[r * 129 + column];
      }
    }
  } else {
    for (int i = tid; i < 128 * 256; i += (Decouple ? 480 : 448)) {
      int r = i / 256, column = i % 256, target_row = start + (qbase + r) / 6;
      int head = kh * 6 + (qbase + r) % 6;
      if (target_row < end) {
        int64_t offset = (int64_t(target_row) * 24 + head) * 256 + column;
        partial[offset] = staged[r * 257 + column];
      }
    }
  }
  __syncthreads();
  cute::cluster_sync();
  if (warp == 13) {
    fa_prefill_after_sync();
    allocator.release_allocation_lock();
    allocator.free(*tmem_base, 512);
  }
}

static inline const FAPrefillPipelineTensorMaps &fa_prefill_cluster_tensor_maps(TensorView cache) {
  using CacheKey = std::tuple<int, uintptr_t, int64_t>;
  static thread_local std::map<CacheKey, FAPrefillPipelineTensorMaps> pipeline_maps;
  CacheKey key{cache.device().device_id, reinterpret_cast<uintptr_t>(cache.data_ptr()),
               cache.size(0)};
  auto found = pipeline_maps.find(key);
  if (found == pipeline_maps.end()) {
    FAPrefillPipelineTensorMaps maps{};
    uint64_t key_dimensions[5] = {64, 784, 4, 4, uint64_t(cache.size(0)) * 2};
    uint64_t key_strides[4] = {2048, 128, 512, 1605632};
    uint64_t value_dimensions[5] = {64, 8, 4, 4, uint64_t(cache.size(0)) * 196};
    uint64_t value_strides[4] = {2048, 128, 512, 16384};
    uint32_t key_box[5] = {64, 64, 4, 1, 1};
    uint32_t value_box[5] = {64, 8, 2, 1, 16};
    uint32_t key16_box[5] = {64, 16, 1, 1, 1};
    uint32_t value16_box[5] = {64, 8, 2, 1, 2};
    uint32_t elements[5] = {1, 1, 1, 1, 1};
    auto encode = [&](CUtensorMap *descriptor, const uint64_t *dimensions, const uint64_t *strides,
                      const uint32_t *box) {
      auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
          descriptor, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, cache.data_ptr(), dimensions, strides,
          box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    };
    encode(&maps.key, key_dimensions, key_strides, key_box);
    encode(&maps.value, value_dimensions, value_strides, value_box);
    encode(&maps.key16, key_dimensions, key_strides, key16_box);
    encode(&maps.value16, value_dimensions, value_strides, value16_box);
    found = pipeline_maps.emplace(key, maps).first;
  }
  return found->second;
}
