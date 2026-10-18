#pragma once

inline constexpr int FAPrefillSharedBytes = 231040;

__device__ inline void fa_prefill_before_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ inline void fa_prefill_after_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

struct FAPrefillPageSpan {
  int first_logical_page;
  int first_page;
  int last_page;
};

__device__ inline FAPrefillPageSpan fa_prefill_page_span(const int32_t *tables, int seq,
                                                         int table_width, int base, int last) {
  int first_logical_page = base / 784;
  int last_logical_page = min(base + 63, last - 1) / 784;
  int first_page = tables[seq * table_width + first_logical_page];
  int last_page = first_page;
  if (last_logical_page != first_logical_page)
    last_page = tables[seq * table_width + last_logical_page];
  return {first_logical_page, first_page, last_page};
}

__device__ inline bool fa_prefill_pages_valid(const FAPrefillPageSpan &pages, int cache_pages) {
  return pages.first_page >= 0 && pages.first_page < cache_pages && pages.last_page >= 0 &&
         pages.last_page < cache_pages;
}

__device__ inline int fa_prefill_cached_page(const FAPrefillPageSpan &pages, int logical) {
  return logical / 784 == pages.first_logical_page ? pages.first_page : pages.last_page;
}

__global__ void fa_prefill_kernel(const __nv_bfloat16 *__restrict__ query,
                                  const __nv_bfloat16 *__restrict__ cache,
                                  const int32_t *__restrict__ tables,
                                  const int32_t *__restrict__ lengths,
                                  const int32_t *__restrict__ starts, float *__restrict__ partial,
                                  float *__restrict__ lse, int table_width, int tiles,
                                  int max_queries, int query_rows, int cache_pages, int first,
                                  const __grid_constant__ CUtensorMap key_map,
                                  const __grid_constant__ CUtensorMap value_map, int64_t l2_bytes) {
  using namespace cute;
  using Element = cutlass::bfloat16_t;
  using Barrier = cutlass::arch::ClusterBarrier;
  using QKMma = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_SS<Element, Element, float, 128, 64, UMMA::Major::K, UMMA::Major::K>{}));
  using PVMma = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_SS<Element, Element, float, 128, 256, UMMA::Major::K, UMMA::Major::MN>{}));
  using QLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_A(QKMma{}, Shape<_128, _256>{})));
  using KLayout = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<Element>{},
                                                   partition_shape_B(QKMma{}, Shape<_64, _256>{})));
  using PLayout = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<Element>{},
                                                   partition_shape_A(PVMma{}, Shape<_128, _64>{})));
  using VLayout = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_MN_SW128_Atom<Element>{},
                                                   partition_shape_B(PVMma{}, Shape<_256, _64>{})));
  static_assert((cosize_v<QLayout> + 2 * cosize_v<KLayout> + 2 * cosize_v<VLayout> +
                 2 * cosize_v<PLayout>)*sizeof(Element) +
                    384 * sizeof(float) + 128 <=
                FAPrefillSharedBytes);
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *query_storage = reinterpret_cast<Element *>(storage);
  auto *key_storage = query_storage + cosize_v<QLayout>;
  auto *value_storage = key_storage + 2 * cosize_v<KLayout>;
  auto *probability_storage = value_storage + 2 * cosize_v<VLayout>;
  auto *alpha_storage = reinterpret_cast<float *>(probability_storage + 2 * cosize_v<PLayout>);
  auto *inverse_storage = alpha_storage + 256;
  auto *barriers = reinterpret_cast<uint64_t *>(inverse_storage + 128);
  auto *key_ready = barriers;
  auto *value_ready = barriers + 2;
  auto *consumed = barriers + 4;
  auto *score_ready = barriers + 6;
  auto *score_free = barriers + 8;
  auto *probability_ready = barriers + 10;
  auto *corrected = barriers + 12;
  auto *tmem_base = reinterpret_cast<uint32_t *>(barriers + 14);
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  int seq = blockIdx.x / tiles, qbase = (blockIdx.x % tiles) * 128, kh = blockIdx.y;
  int start = starts[seq], end = starts[seq + 1];
  if (start < 0 || end > query_rows || end <= start || end - start > max_queries ||
      starts[0] != 0 || starts[gridDim.x / tiles] != query_rows) {
    asm volatile("trap;");
    return;
  }
  if (start + qbase / 6 >= end)
    return;
  int last = lengths[min(end - 1, start + (qbase + 127) / 6)];
  if (last < first || last > table_width * 784) {
    asm volatile("trap;");
    return;
  }
  constexpr int64_t bytes_per_token = 2 * 4 * 256 * int64_t(sizeof(Element));
  int64_t kv_bytes = (int64_t(lengths[end - 1]) - first) * bytes_per_token;
  bool split_key_value = kv_bytes > l2_bytes;
  int length = 0;
  if (tid < 128) {
    int row = start + (qbase + tid) / 6;
    length = row < end ? lengths[row] : 0;
    if (length < 0 || length > last) {
      asm volatile("trap;");
      return;
    }
  }
  TMEM::Allocator1Sm allocator;
  if (warp == 9)
    allocator.allocate(512, tmem_base);
  if (tid == 0) {
    for (int index = 0; index < 14; ++index)
      initialize_barrier(barriers[index], index >= 8 ? 128 : 1);
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
  } else if (tid < 256) {
    uint32_t address = *tmem_base + uint32_t((tid - 128) / 32 * 32) * 65536u + 128;
    float zeros[32] = {};
#pragma unroll
    for (int c = 0; c < 256; c += 32)
      fa_tmem_store32(address + c, zeros);
    cutlass::arch::fence_view_async_tmem_store();
    fa_prefill_before_sync();
  }
  __syncthreads();
  fa_prefill_after_sync();
  int trips = (last - first + 63) / 64;
  if (warp == 8) {
    for (int tile = 0, base = first; tile < trips; ++tile, base += 64) {
      int slot = tile & 1, phase = (tile / 2) & 1;
      auto *keys = key_storage + slot * cosize_v<KLayout>;
      auto *values = value_storage + slot * cosize_v<VLayout>;
      if (!split_key_value) {
        if (tile >= 2)
          wait_barrier(consumed[slot], phase ^ 1);
        __syncwarp();
        auto pages = fa_prefill_page_span(tables, seq, table_width, base, last);
        if (!fa_prefill_pages_valid(pages, cache_pages)) {
          asm volatile("trap;");
          return;
        }
        if (first % 16 == 0 && base + 64 <= last && base % 784 + 64 <= 784) {
          if (lane == 0) {
            int page = pages.first_page;
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(key_ready + slot, 65536);
            SM90_TMA_LOAD_5D::copy(&key_map, key_ready + slot,
                                   uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), keys, 0, base % 784,
                                   0, kh, page * 2);
            SM90_TMA_LOAD_5D::copy(&value_map, key_ready + slot,
                                   uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), values, 0, 0, 0, kh,
                                   (page * 2 + 1) * 98 + (base % 784) / 8);
          }
        } else {
          auto sK = make_tensor(make_smem_ptr(keys), KLayout{});
          auto sV = make_tensor(make_smem_ptr(values), VLayout{});
          for (int i = lane; i < 64 * 32; i += 32) {
            int r = i / 32, c = (i % 32) * 8, pos = base + r;
            int logical = min(pos, last - 1), page = fa_prefill_cached_page(pages, logical);
            int64_t offset = ((int64_t(page) * 1568 + logical % 784) * 4 + kh) * 256 + c;
            unsigned kd = __cvta_generic_to_shared(&sK(make_coord(r, c % 16), 0, c / 16));
            unsigned vd = __cvta_generic_to_shared(&sV(make_coord(c, r % 16), 0, r / 16));
            int bytes = pos < last ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(kd),
                         "l"(cache + offset), "r"(bytes)
                         : "memory");
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(vd),
                         "l"(cache + offset + int64_t(784) * 4 * 256), "r"(bytes)
                         : "memory");
          }
          asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
          __syncwarp();
          if (lane == 0)
            Barrier::arrive(key_ready + slot);
        }
        continue;
      }
      bool tma = first % 16 == 0 && base + 64 <= last && base % 784 + 64 <= 784;
      auto pages = fa_prefill_page_span(tables, seq, table_width, base, last);
      if (!fa_prefill_pages_valid(pages, cache_pages)) {
        asm volatile("trap;");
        return;
      }
      int page = pages.first_page;
#pragma unroll
      for (int operand = 0; operand < 2; ++operand) {
        bool is_value = operand != 0;
        if (tile >= 2)
          wait_barrier(is_value ? consumed[slot] : score_ready[slot], phase ^ 1);
        __syncwarp();
        auto *signal = is_value ? value_ready : key_ready;
        if (tma) {
          if (lane == 0) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(signal + slot, 32768);
            if (is_value)
              SM90_TMA_LOAD_5D::copy(&value_map, signal + slot,
                                     uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), values, 0, 0, 0,
                                     kh, (page * 2 + 1) * 98 + (base % 784) / 8);
            else
              SM90_TMA_LOAD_5D::copy(&key_map, signal + slot,
                                     uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), keys, 0,
                                     base % 784, 0, kh, page * 2);
          }
        } else {
          auto sK = make_tensor(make_smem_ptr(keys), KLayout{});
          auto sV = make_tensor(make_smem_ptr(values), VLayout{});
          for (int i = lane; i < 64 * 32; i += 32) {
            int r = i / 32, c = (i % 32) * 8, pos = base + r;
            int logical = min(pos, last - 1), page = fa_prefill_cached_page(pages, logical);
            int64_t offset = ((int64_t(page) * 1568 + logical % 784) * 4 + kh) * 256 + c;
            unsigned destination =
                is_value ? __cvta_generic_to_shared(&sV(make_coord(c, r % 16), 0, r / 16))
                         : __cvta_generic_to_shared(&sK(make_coord(r, c % 16), 0, c / 16));
            auto *source = cache + offset + (is_value ? int64_t(784) * 4 * 256 : 0);
            int bytes = pos < last ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(destination),
                         "l"(source), "r"(bytes)
                         : "memory");
          }
          asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
          __syncwarp();
          if (lane == 0)
            Barrier::arrive(signal + slot);
        }
      }
    }
  } else if (warp == 9) {
    QKMma qk;
    PVMma pv;
    auto qfrag = qk.get_slice(0).make_fragment_A(sQ);
    auto score = partition_fragment_C(qk, Shape<_128, _64>{});
    auto output = partition_fragment_C(pv, Shape<_128, _256>{});
    output.data() = *tmem_base + 128;
    auto issue_qk = [&](int tile) {
      int slot = tile & 1, phase = (tile / 2) & 1;
      wait_barrier(key_ready[slot], phase);
      if (tile >= 2)
        wait_barrier(score_free[slot], phase ^ 1);
      fa_prefill_after_sync();
      auto sK = make_tensor(make_smem_ptr(key_storage + slot * cosize_v<KLayout>), KLayout{});
      auto kfrag = qk.get_slice(0).make_fragment_B(sK);
      score.data() = *tmem_base + slot * 64;
      qk.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        gemm(qk, qfrag(_, _, k), kfrag(_, _, k), score);
        qk.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(score_ready + slot);
    };
    if (trips > 0)
      issue_qk(0);
    for (int tile = 0; tile < trips; ++tile) {
      int slot = tile & 1, phase = (tile / 2) & 1;
      if (tile + 1 < trips)
        issue_qk(tile + 1);
      if (split_key_value)
        wait_barrier(value_ready[slot], phase);
      wait_barrier(corrected[slot], phase);
      fa_prefill_after_sync();
      auto sP =
          make_tensor(make_smem_ptr(probability_storage + slot * cosize_v<PLayout>), PLayout{});
      auto sV = make_tensor(make_smem_ptr(value_storage + slot * cosize_v<VLayout>), VLayout{});
      auto pfrag = pv.get_slice(0).make_fragment_A(sP);
      auto vfrag = pv.get_slice(0).make_fragment_B(sV);
      pv.accumulate_ = UMMA::ScaleOut::One;
#pragma unroll
      for (int k = 0; k < 4; ++k)
        gemm(pv, pfrag(_, _, k), vfrag(_, _, k), output);
      cutlass::arch::umma_arrive(consumed + slot);
    }
    if (trips > 0)
      wait_barrier(consumed[(trips - 1) & 1], ((trips - 1) / 2) & 1);
    fa_prefill_after_sync();
  } else if (tid < 128) {
    uint32_t lane_base = *tmem_base + uint32_t(warp * 32) * 65536u;
    int row = start + (qbase + tid) / 6;
    float maximum = -INFINITY, denominator = 0;
    for (int tile = 0, base = first; tile < trips; ++tile, base += 64) {
      int slot = tile & 1, phase = (tile / 2) & 1;
      if (tile >= 2)
        wait_barrier(consumed[slot], phase ^ 1);
      wait_barrier(score_ready[slot], phase);
      fa_prefill_after_sync();
      float values[64], next_max = maximum;
#pragma unroll
      for (int c = 0; c < 64; c += 32) {
        float block[32];
        fa_tmem_load32(lane_base + slot * 64 + c, block);
        cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          float s =
              row < end && base + c + j < length ? block[j] * 0.09016844005556021f : -INFINITY;
          values[c + j] = s;
          next_max = fmaxf(next_max, s);
        }
      }
      fa_prefill_before_sync();
      Barrier::arrive(score_free + slot);
      float safe = next_max == -INFINITY ? 0.f : next_max;
      float alpha = fa_exp2(maximum - safe), sum = 0;
      auto sP =
          make_tensor(make_smem_ptr(probability_storage + slot * cosize_v<PLayout>), PLayout{});
#pragma unroll
      for (int c = 0; c < 64; c += 8) {
        uint32_t packed[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float p0 = fa_exp2(values[c + j * 2] - safe);
          sum += p0;
          float p1 = fa_exp2(values[c + j * 2 + 1] - safe);
          sum += p1;
          uint32_t word;
          asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(word) : "f"(p0), "f"(p1));
          packed[j] = word;
        }
        unsigned address = __cvta_generic_to_shared(&sP(make_coord(tid, c % 16), 0, c / 16));
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :
                     : "r"(address), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3])
                     : "memory");
      }
      maximum = next_max;
      denominator = denominator * alpha + sum;
      alpha_storage[slot * 128 + tid] = alpha;
      cutlass::arch::fence_view_async_shared();
      Barrier::arrive(probability_ready + slot);
    }
    if (row < end) {
      int head = kh * 6 + (qbase + tid) % 6;
      lse[int64_t(row) * 24 + head] = maximum + log2f(denominator);
    }
    inverse_storage[tid] = denominator > 0 ? 1.f / denominator : 1.f;
  } else {
    int owner = tid - 128;
    uint32_t address = *tmem_base + uint32_t(owner / 32 * 32) * 65536u + 128;
    for (int tile = 0; tile < trips; ++tile) {
      int slot = tile & 1, phase = (tile / 2) & 1;
      wait_barrier(probability_ready[slot], phase);
      if (tile > 0)
        wait_barrier(consumed[slot ^ 1], ((tile - 1) / 2) & 1);
      fa_prefill_after_sync();
      float alpha = alpha_storage[slot * 128 + owner];
      if (__any_sync(0xffffffff, alpha != 1.f)) {
#pragma unroll 1
        for (int c = 0; c < 256; c += 32) {
          float block[32];
          fa_tmem_load32(address + c, block);
          cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
          for (int j = 0; j < 32; ++j)
            block[j] *= alpha;
          fa_tmem_store32(address + c, block);
        }
      }
      cutlass::arch::fence_view_async_tmem_store();
      fa_prefill_before_sync();
      Barrier::arrive(corrected + slot);
    }
  }
  __syncthreads();
  fa_prefill_after_sync();
  if (tid >= 128 && tid < 256) {
    int owner = tid - 128;
    uint32_t address = *tmem_base + uint32_t(owner / 32 * 32) * 65536u + 128;
    float inverse = inverse_storage[owner];
    auto *staged = reinterpret_cast<float *>(key_storage);
#pragma unroll
    for (int c = 0; c < 256; c += 32) {
      float block[32];
      fa_tmem_load32(address + c, block);
      cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
      for (int j = 0; j < 32; ++j)
        staged[owner * 33 + j] = block[j] * inverse;
      asm volatile("barrier.sync 1, 128;" ::: "memory");
      for (int i = owner; i < 128 * 32; i += 128) {
        int r = i / 32, column = i % 32, target_row = start + (qbase + r) / 6;
        int head = kh * 6 + (qbase + r) % 6;
        if (target_row < end)
          partial[(int64_t(target_row) * 24 + head) * 256 + c + column] = staged[r * 33 + column];
      }
      asm volatile("barrier.sync 1, 128;" ::: "memory");
    }
    fa_prefill_before_sync();
  }
  __syncthreads();
  if (warp == 9) {
    fa_prefill_after_sync();
    allocator.release_allocation_lock();
    allocator.free(*tmem_base, 512);
  }
}
