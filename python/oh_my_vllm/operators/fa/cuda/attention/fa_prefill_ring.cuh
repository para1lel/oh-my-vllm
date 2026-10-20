#pragma once

#include "fa_attention.cuh"
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cuda_host_adapter.hpp>
#include <map>
#include <tuple>
#include <utility>

template <size_t... Indices>
__device__ inline void fa_ring_tmem_load64_impl(uint32_t address, float (&values)[64],
                                                std::index_sequence<Indices...>) {
  uint32_t words[64];
  cute::SM100_TMEM_LOAD_32dp32b64x::copy(address, words[Indices]...);
#pragma unroll
  for (int i = 0; i < 64; ++i)
    values[i] = __uint_as_float(words[i]);
}

__device__ inline void fa_ring_tmem_load64(uint32_t address, float (&values)[64]) {
  fa_ring_tmem_load64_impl(address, values, std::make_index_sequence<64>{});
}

template <size_t... Indices>
__device__ inline void fa_ring_tmem_store64_impl(uint32_t address, const float (&values)[64],
                                                 std::index_sequence<Indices...>) {
  cute::SM100_TMEM_STORE_32dp32b64x::copy(__float_as_uint(values[Indices])..., address);
}

__device__ inline void fa_ring_tmem_store64(uint32_t address, const float (&values)[64]) {
  fa_ring_tmem_store64_impl(address, values, std::make_index_sequence<64>{});
}

__device__ inline void fa_ring_tmem_load32(uint32_t address, float (&values)[32]) {
  uint32_t words[32];
  cute::SM100_TMEM_LOAD_32dp32b32x::copy(
      address, words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7],
      words[8], words[9], words[10], words[11], words[12], words[13], words[14], words[15],
      words[16], words[17], words[18], words[19], words[20], words[21], words[22], words[23],
      words[24], words[25], words[26], words[27], words[28], words[29], words[30], words[31]);
#pragma unroll
  for (int i = 0; i < 32; ++i)
    values[i] = __uint_as_float(words[i]);
}

__device__ inline void fa_ring_tmem_store32(uint32_t address, const float (&values)[32]) {
  cute::SM100_TMEM_STORE_32dp32b32x::copy(
      __float_as_uint(values[0]), __float_as_uint(values[1]), __float_as_uint(values[2]),
      __float_as_uint(values[3]), __float_as_uint(values[4]), __float_as_uint(values[5]),
      __float_as_uint(values[6]), __float_as_uint(values[7]), __float_as_uint(values[8]),
      __float_as_uint(values[9]), __float_as_uint(values[10]), __float_as_uint(values[11]),
      __float_as_uint(values[12]), __float_as_uint(values[13]), __float_as_uint(values[14]),
      __float_as_uint(values[15]), __float_as_uint(values[16]), __float_as_uint(values[17]),
      __float_as_uint(values[18]), __float_as_uint(values[19]), __float_as_uint(values[20]),
      __float_as_uint(values[21]), __float_as_uint(values[22]), __float_as_uint(values[23]),
      __float_as_uint(values[24]), __float_as_uint(values[25]), __float_as_uint(values[26]),
      __float_as_uint(values[27]), __float_as_uint(values[28]), __float_as_uint(values[29]),
      __float_as_uint(values[30]), __float_as_uint(values[31]), address);
}

template <size_t... I>
__device__ inline void fa_ring_pair_load_impl(uint32_t address, float (&values)[64],
                                              std::index_sequence<I...>) {
  uint32_t words[64];
  cute::SM100_TMEM_LOAD_16dp32b64x::copy(address, words[I]...);
#pragma unroll
  for (int i = 0; i < 64; ++i)
    values[i] = __uint_as_float(words[i]);
}

__device__ inline void fa_ring_pair_load(uint32_t a, float (&v)[64]) {
  fa_ring_pair_load_impl(a, v, std::make_index_sequence<64>{});
}

template <size_t... I>
__device__ inline void fa_ring_pair_store_impl(uint32_t address, const float (&values)[32],
                                               std::index_sequence<I...>) {
  cute::SM100_TMEM_STORE_16dp32b32x::copy(__float_as_uint(values[I])..., address);
}

__device__ inline void fa_ring_pair_store(uint32_t a, const float (&v)[32]) {
  fa_ring_pair_store_impl(a, v, std::make_index_sequence<32>{});
}

#pragma once

inline constexpr int FARingSharedBytes = 199680;

__device__ inline void fa_ring_before_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ inline void fa_ring_after_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

struct FARingPageSpan {
  int first_logical_page;
  int first_page;
  int last_page;
};

__device__ inline FARingPageSpan fa_ring_page_span(const int32_t *tables, int seq, int table_width,
                                                   int base, int last) {
  int first_logical_page = min(base, last - 1) / 784;
  int last_logical_page = min(base + 127, last - 1) / 784;
  int first_page = tables[seq * table_width + first_logical_page];
  int last_page = first_page;
  if (last_logical_page != first_logical_page)
    last_page = tables[seq * table_width + last_logical_page];
  return {first_logical_page, first_page, last_page};
}

__device__ inline bool fa_ring_pages_valid(const FARingPageSpan &pages, int cache_pages) {
  return pages.first_page >= 0 && pages.first_page < cache_pages && pages.last_page >= 0 &&
         pages.last_page < cache_pages;
}

__device__ inline int fa_ring_cached_page(const FARingPageSpan &pages, int logical) {
  return logical / 784 == pages.first_logical_page ? pages.first_page : pages.last_page;
}

template <bool ExportBFloat16, bool RoundMax = false>
__global__ __launch_bounds__(480, 1) void fa_ring_kernel(
    const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ cache,
    const int32_t *__restrict__ tables, const int32_t *__restrict__ lengths,
    const int32_t *__restrict__ starts, float *__restrict__ partial, float *__restrict__ lse,
    int table_width, int tiles, int max_queries, int query_rows, int cache_pages, int first,
    const __grid_constant__ CUtensorMap key_map, const __grid_constant__ CUtensorMap value_map,
    int64_t l2_bytes, const __grid_constant__ CUtensorMap key16_map,
    const __grid_constant__ CUtensorMap value16_map, __nv_bfloat16 *attended) {
  using namespace cute;
  using Element = cutlass::bfloat16_t;
  using Barrier = cutlass::arch::ClusterBarrier;
  using QKMma = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_SS<Element, Element, float, 128, 128, UMMA::Major::K, UMMA::Major::K>{}));
  using PVMma = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_TS<Element, Element, float, 128, 128, UMMA::Major::K, UMMA::Major::MN>{}));
  using QLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_A(QKMma{}, Shape<_128, _256>{})));
  using KLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_B(QKMma{}, Shape<_128, _128>{})));
  using PLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_A(PVMma{}, Shape<_128, _128>{})));
  using VLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_MN_SW128_Atom<Element>{}, partition_shape_B(PVMma{}, Shape<_128, _128>{})));
  static_assert((cosize_v<QLayout> + 2 * cosize_v<KLayout> + 2 * cosize_v<VLayout> +
                 0 * cosize_v<PLayout>)*sizeof(Element) +
                    384 * sizeof(float) + 128 <=
                FARingSharedBytes);
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *query_storage = reinterpret_cast<Element *>(storage);
  auto *key_storage = query_storage + cosize_v<QLayout>;
  auto *value_storage = key_storage + 2 * cosize_v<KLayout>;
  auto *probability_storage = value_storage + 2 * cosize_v<VLayout>;
  auto *alpha_storage = reinterpret_cast<float *>(probability_storage);
  auto *inverse_storage = alpha_storage + 256;
  auto *barriers = reinterpret_cast<uint64_t *>(inverse_storage + 128);
  auto *key_ready = barriers;
  auto *value_ready = barriers + 2;
  auto *consumed = barriers + 4;
  auto *score_ready = barriers + 6;
  auto *score_free = barriers + 8;
  auto *probability_ready = barriers + 10;
  auto *corrected = barriers + 12;
  auto *tmem_base = reinterpret_cast<uint32_t *>(barriers + 22);
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  int rank = 0, cluster = blockIdx.x;
  int seq = cluster / tiles, cluster_qbase = (cluster % tiles) * 128;
  int qbase = cluster_qbase + rank * 128, kh = blockIdx.y;
  auto *key_consumed = barriers + 18;
  auto *value_consumed = barriers + 20;
  auto *remote_key_ready = barriers + 14;
  auto *remote_pv_ready = barriers + 16;
  int start = starts[seq], end = starts[seq + 1];
  if (start < 0 || end > query_rows || end <= start || end - start > max_queries ||
      starts[0] != 0 || starts[gridDim.x / tiles] != query_rows) {
    asm volatile("trap;");
    return;
  }
  if (start + cluster_qbase / 6 >= end)
    return;
  int last = lengths[min(end - 1, start + (cluster_qbase + 127) / 6)];
  if (last < first || last > table_width * 784) {
    asm volatile("trap;");
    return;
  }
  constexpr int64_t bytes_per_token = 2 * 4 * 256 * int64_t(sizeof(Element));
  int64_t kv_bytes = (int64_t(lengths[end - 1]) - first) * bytes_per_token;
  bool split_key_value = kv_bytes > l2_bytes;
  int length = 0;
  if (tid < 256) {
    int row = start + (qbase + (warp % 4) * 32 + (warp / 4) * 16 + lane % 16) / 6;
    length = row < end ? lengths[row] : 0;
    if (length < 0 || length > last) {
      asm volatile("trap;");
      return;
    }
  }
  TMEM::Allocator1Sm allocator;
  if (warp == 14)
    allocator.allocate(512, tmem_base);
  if (tid == 0) {
    for (int index = 0; index < 22; ++index)
      initialize_barrier(barriers[index], index == 10 || index == 11 ? 256
                                          : index >= 8 && index < 14 ? 128
                                                                     : 1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  auto sQ = make_tensor(make_smem_ptr(query_storage), QLayout{});
  if (tid < 128) {
    for (int i = tid; i < 128 * 32; i += 128) {
      int r = i / 32, c = (i % 32) * 8;
      int row = start + (qbase + r) / 6, head = kh * 6 + (qbase + r) % 6;
      uint4 value =
          row < end ? *reinterpret_cast<const uint4 *>(query + (int64_t(row) * 24 + head) * 256 + c)
                    : make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4 *>(&sQ(make_coord(r, c % 16), 0, c / 16)) = value;
    }
    cutlass::arch::fence_view_async_shared();
  } else if (tid >= 256 && tid < 384) {
    uint32_t address = *tmem_base + uint32_t((tid - 256) / 32 * 32) * 65536u + 256;
    float zeros[32] = {};
#pragma unroll
    for (int c = 0; c < 256; c += 32)
      fa_ring_tmem_store32(address + c, zeros);
    cutlass::arch::fence_view_async_tmem_store();
    fa_ring_before_sync();
  }
  __syncthreads();
  fa_ring_after_sync();
  int trips = (last - first + 127) / 128;
  if (warp == 12 || warp == 13) {
    bool value = warp == 13;
    for (int tile = 0, base = first; tile < trips; ++tile, base += 128) {
#pragma unroll
      for (int chunk = 0; chunk < 2; ++chunk) {
        int phase = tile & 1;
        if (tile > 0)
          wait_barrier(value ? value_consumed[chunk] : key_consumed[chunk], phase ^ 1);
        __syncwarp();
        auto pages = fa_ring_page_span(tables, seq, table_width, base, last);
        if (!fa_ring_pages_valid(pages, cache_pages)) {
          asm volatile("trap;");
          return;
        }
        bool tma = base % 16 == 0 && base + 128 <= last && base % 784 + 128 <= 784;
        auto *target = value ? value_storage + chunk * cosize_v<VLayout>
                             : key_storage + chunk * cosize_v<KLayout>;
        auto *signal = value ? value_ready : key_ready;
        if (tma) {
          if (lane == 0) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(signal + chunk, 32768);
            if (value)
              SM90_TMA_LOAD_5D::copy(
                  &value_map, signal + chunk, uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), target, 0,
                  0, chunk * 2, kh, (pages.first_page * 2 + 1) * 98 + (base % 784) / 8);
            else
              SM90_TMA_LOAD_5D::copy(&key_map, signal + chunk,
                                     uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), target, 0,
                                     base % 784, chunk * 2, kh, pages.first_page * 2);
          }
        } else if (base % 16 == 0 && base + 128 <= last) {
          if (lane == 0) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(signal + chunk, 32768);
            auto sK = make_tensor(make_smem_ptr(target), KLayout{});
            auto sV = make_tensor(make_smem_ptr(target), VLayout{});
            for (int r = 0; r < 128; r += 16) {
              int logical = base + r, page = fa_ring_cached_page(pages, logical);
              if (value) {
                auto *dest = &sV(make_coord(0, r % 16), 0, r / 16);
                SM90_TMA_LOAD_5D::copy(&value16_map, signal + chunk,
                                       uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), dest, 0, 0,
                                       chunk * 2, kh, (page * 2 + 1) * 98 + (logical % 784) / 8);
              } else
                for (int c = 0; c < 2; ++c) {
                  auto *dest = &sK(make_coord(r, 0), 0, c * 4);
                  SM90_TMA_LOAD_5D::copy(&key16_map, signal + chunk,
                                         uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), dest, 0,
                                         logical % 784, chunk * 2 + c, kh, page * 2);
                }
            }
          }
        } else {
          auto sK = make_tensor(make_smem_ptr(target), KLayout{});
          auto sV = make_tensor(make_smem_ptr(target), VLayout{});
          for (int i = lane; i < 2048; i += 32) {
            int row = i / 16, col = (i % 16) * 8, pos = base + row, logical = min(pos, last - 1);
            int page = fa_ring_cached_page(pages, logical);
            int64_t offset =
                ((int64_t(page) * 1568 + logical % 784) * 4 + kh) * 256 + chunk * 128 + col;
            unsigned dest =
                value ? __cvta_generic_to_shared(&sV(make_coord(col, row % 16), 0, row / 16))
                      : __cvta_generic_to_shared(&sK(make_coord(row, col % 16), 0, col / 16));
            auto *source = cache + offset + (value ? int64_t(784) * 4 * 256 : 0);
            int bytes = pos < last ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dest), "l"(source),
                         "r"(bytes)
                         : "memory");
          }
          asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
          __syncwarp();
          if (lane == 0)
            Barrier::arrive(signal + chunk);
        }
      }
    }
  } else if (warp == 14) {
    QKMma qk;
    PVMma pv;
    auto qfrag = qk.get_slice(0).make_fragment_A(sQ);
    auto score = partition_fragment_C(qk, Shape<_128, _128>{});
    auto output = partition_fragment_C(pv, Shape<_128, _128>{});
    auto qk_chunk = [&](int tile, int chunk) {
      int slot = tile & 1, phase = (tile / 2) & 1;
      if (chunk == 0 && tile >= 2)
        wait_barrier(consumed[slot], phase ^ 1);
      wait_barrier(key_ready[chunk], tile & 1);
      fa_ring_after_sync();
      auto sK = make_tensor(make_smem_ptr(key_storage + chunk * cosize_v<KLayout>), KLayout{});
      auto kfrag = qk.get_slice(0).make_fragment_B(sK);
      score.data() = *tmem_base + slot * 128;
      qk.accumulate_ = chunk == 0 ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        gemm(qk, qfrag(_, _, chunk * 8 + k), kfrag(_, _, k), score);
        qk.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(key_consumed + chunk);
      if (chunk == 1)
        cutlass::arch::umma_arrive(score_ready + slot);
    };
    auto pv_chunk = [&](int tile, int chunk) {
      int slot = tile & 1, phase = (tile / 2) & 1;
      wait_barrier(value_ready[chunk], tile & 1);
      if (chunk == 0) {
        wait_barrier(probability_ready[slot], phase);
        wait_barrier(corrected[slot], phase);
      }
      fa_ring_after_sync();
      auto sP = make_tensor(make_smem_ptr(probability_storage), PLayout{});
      auto sV = make_tensor(make_smem_ptr(value_storage + chunk * cosize_v<VLayout>), VLayout{});
      auto pfrag = pv.get_slice(0).make_fragment_A(sP);
      auto vfrag = pv.get_slice(0).make_fragment_B(sV);
      pfrag.data() = *tmem_base + slot * 128 + 64;
      output.data() = *tmem_base + 256 + chunk * 128;
      pv.accumulate_ = UMMA::ScaleOut::One;
#pragma unroll
      for (int k = 0; k < 8; ++k)
        gemm(pv, pfrag(_, _, k), vfrag(_, _, k), output);
      cutlass::arch::umma_arrive(value_consumed + chunk);
      if (chunk == 1)
        cutlass::arch::umma_arrive(consumed + slot);
    };
    if (trips > 0) {
      qk_chunk(0, 0);
      qk_chunk(0, 1);
    }
    for (int tile = 0; tile < trips; ++tile) {
      if (tile + 1 < trips)
        qk_chunk(tile + 1, 0);
      pv_chunk(tile, 0);
      if (tile + 1 < trips)
        qk_chunk(tile + 1, 1);
      pv_chunk(tile, 1);
    }
    if (trips > 0)
      wait_barrier(consumed[(trips - 1) & 1], ((trips - 1) / 2) & 1);
    fa_ring_after_sync();
  } else if (tid < 256) {
    int owner = (warp % 4) * 32 + (warp / 4) * 16 + lane % 16, half = lane / 16;
    uint32_t lane_base = *tmem_base + uint32_t((warp / 4) * 16) * 65536u;
    int row = start + (qbase + owner) / 6;
    float maximum = -INFINITY, denominator = 0;
    for (int tile = 0, base = first; tile < trips; ++tile, base += 128) {
      int slot = tile & 1, phase = (tile / 2) & 1;
      wait_barrier(score_ready[slot], phase);
      fa_ring_after_sync();
      float values[64], next_max = maximum;
      float maxima[4] = {maximum, maximum, maximum, maximum};
      fa_ring_pair_load(lane_base + slot * 128, values);
      cutlass::arch::fence_view_async_tmem_load();
      bool complete = row < end && base + 128 <= length;
      if (complete) {
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          float2 scaled = __fmul2_rn(make_float2(values[2 * j], values[2 * j + 1]),
                                     make_float2(0.09016844005556021f, 0.09016844005556021f));
          float v0 = scaled.x;
          float v1 = scaled.y;
          values[2 * j] = v0;
          values[2 * j + 1] = v1;
          maxima[(2 * j) % 4] = fmaxf(maxima[(2 * j) % 4], v0);
          maxima[(2 * j + 1) % 4] = fmaxf(maxima[(2 * j + 1) % 4], v1);
        }
      } else {
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          float2 scaled = __fmul2_rn(make_float2(values[2 * j], values[2 * j + 1]),
                                     make_float2(0.09016844005556021f, 0.09016844005556021f));
          float v0 = row < end && base + half * 64 + 2 * j < length ? scaled.x : -INFINITY;
          float v1 = row < end && base + half * 64 + 2 * j + 1 < length ? scaled.y : -INFINITY;
          values[2 * j] = v0;
          values[2 * j + 1] = v1;
          maxima[(2 * j) % 4] = fmaxf(maxima[(2 * j) % 4], v0);
          maxima[(2 * j + 1) % 4] = fmaxf(maxima[(2 * j + 1) % 4], v1);
        }
      }
      next_max = fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
      next_max = fmaxf(next_max, __shfl_xor_sync(0xffffffff, next_max, 16));
      if constexpr (RoundMax)
        next_max = ceilf(next_max);
      float safe = next_max == -INFINITY ? 0.f : next_max;
      float alpha = fa_exp2(maximum - safe), sum = 0;
      if (half == 0) {
        alpha_storage[slot * 128 + owner] = alpha;
        cutlass::arch::fence_view_async_shared();
        fa_ring_before_sync();
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
      fa_ring_pair_store(lane_base + slot * 128 + 64, packed);
      cutlass::arch::fence_view_async_tmem_store();
      fa_ring_before_sync();
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
      fa_ring_after_sync();
      float alpha = alpha_storage[slot * 128 + owner];
      if (__any_sync(0xffffffff, alpha != 1.f)) {
#pragma unroll 1
        for (int c = 0; c < 256; c += 64) {
          float block[64];
          fa_ring_tmem_load64(address + c, block);
          cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
          for (int j = 0; j < 32; ++j) {
            float2 v =
                __fmul2_rn(make_float2(block[2 * j], block[2 * j + 1]), make_float2(alpha, alpha));
            block[2 * j] = v.x;
            block[2 * j + 1] = v.y;
          }
          fa_ring_tmem_store64(address + c, block);
        }
      }
      cutlass::arch::fence_view_async_tmem_store();
      fa_ring_before_sync();
      Barrier::arrive(corrected + slot);
    }
  }
  __syncthreads();
  fa_ring_after_sync();
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
      fa_ring_tmem_load32(address + c, values);
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
    fa_ring_before_sync();
  }
  __syncthreads();
  if constexpr (ExportBFloat16) {
    for (int i = tid; i < 128 * 128; i += 480) {
      int r = i / 128, column = i % 128, target_row = start + (qbase + r) / 6;
      int head = kh * 6 + (qbase + r) % 6;
      if (target_row < end) {
        int64_t offset = (int64_t(target_row) * 24 + head) * 128 + column;
        reinterpret_cast<uint32_t *>(attended)[offset] = packed_stage[r * 129 + column];
      }
    }
  } else {
    for (int i = tid; i < 128 * 256; i += 480) {
      int r = i / 256, column = i % 256, target_row = start + (qbase + r) / 6;
      int head = kh * 6 + (qbase + r) % 6;
      if (target_row < end) {
        int64_t offset = (int64_t(target_row) * 24 + head) * 256 + column;
        partial[offset] = staged[r * 257 + column];
      }
    }
  }
  __syncthreads();
  if (warp == 14) {
    fa_ring_after_sync();
    allocator.release_allocation_lock();
    allocator.free(*tmem_base, 512);
  }
}

struct FARingPipelineTensorMaps {
  alignas(128) CUtensorMap key;
  alignas(128) CUtensorMap value;
  alignas(128) CUtensorMap key16;
  alignas(128) CUtensorMap value16;
};

static inline const FARingPipelineTensorMaps &fa_ring_pipeline_tensor_maps(TensorView cache) {
  using CacheKey = std::tuple<int, uintptr_t, int64_t>;
  static thread_local std::map<CacheKey, FARingPipelineTensorMaps> pipeline_maps;
  CacheKey key{cache.device().device_id, reinterpret_cast<uintptr_t>(cache.data_ptr()),
               cache.size(0)};
  auto found = pipeline_maps.find(key);
  if (found == pipeline_maps.end()) {
    FARingPipelineTensorMaps maps{};
    uint64_t key_dimensions[5] = {64, 784, 4, 4, uint64_t(cache.size(0)) * 2};
    uint64_t key_strides[4] = {2048, 128, 512, 1605632};
    uint64_t value_dimensions[5] = {64, 8, 4, 4, uint64_t(cache.size(0)) * 196};
    uint64_t value_strides[4] = {2048, 128, 512, 16384};
    uint32_t key_box[5] = {64, 128, 2, 1, 1};
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

static inline int64_t fa_ring_pipeline_l2_bytes(int device) {
  static thread_local std::map<int, int> capacities;
  auto found = capacities.find(device);
  if (found == capacities.end()) {
    int bytes = 0;
    check_cuda(cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, device));
    TVM_FFI_ICHECK(bytes > 0);
    found = capacities.emplace(device, bytes).first;
  }
  return found->second;
}
