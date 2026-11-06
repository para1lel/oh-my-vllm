#pragma once

namespace fa_prefill_stages {
__device__ __forceinline__ void load_combined(Shared &s, const Work &w, Element *keys,
                                              Element *values, int slot, int base,
                                              const FAPrefillPageSpan &pages) {
  if (w.first % 16 == 0 && base + 64 <= w.last && base % 784 + 64 <= 784) {
    if (w.lane == 0) {
      int page = pages.first_page;
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(s.key_ready + slot, 65536);
      SM90_TMA_LOAD_5D::copy(&w.key_map, s.key_ready + slot,
                             uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), keys, 0, base % 784, 0,
                             w.kh, page * 2);
      SM90_TMA_LOAD_5D::copy(&w.value_map, s.key_ready + slot,
                             uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), values, 0, 0, 0, w.kh,
                             (page * 2 + 1) * 98 + (base % 784) / 8);
    }
  } else {
    auto sK = make_tensor(make_smem_ptr(keys), KLayout{});
    auto sV = make_tensor(make_smem_ptr(values), VLayout{});
    for (int i = w.lane; i < 64 * 32; i += 32) {
      int r = i / 32, c = (i % 32) * 8, pos = base + r;
      int logical = min(pos, w.last - 1), page = fa_prefill_cached_page(pages, logical);
      int64_t offset = ((int64_t(page) * 1568 + logical % 784) * 4 + w.kh) * 256 + c;
      unsigned kd = __cvta_generic_to_shared(&sK(make_coord(r, c % 16), 0, c / 16));
      unsigned vd = __cvta_generic_to_shared(&sV(make_coord(c, r % 16), 0, r / 16));
      int bytes = pos < w.last ? 16 : 0;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(kd), "l"(w.cache + offset),
                   "r"(bytes)
                   : "memory");
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(vd),
                   "l"(w.cache + offset + int64_t(784) * 4 * 256), "r"(bytes)
                   : "memory");
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    __syncwarp();
    if (w.lane == 0)
      Barrier::arrive(s.key_ready + slot);
  }
}

__device__ __forceinline__ void load_split(Shared &s, const Work &w, Element *keys, Element *values,
                                           int slot, int tile, int phase, int base, bool tma,
                                           const FAPrefillPageSpan &pages) {
  int page = pages.first_page;
#pragma unroll
  for (int operand = 0; operand < 2; ++operand) {
    bool is_value = operand != 0;
    if (tile >= 2)
      wait_barrier(is_value ? s.consumed[slot] : s.score_ready[slot], phase ^ 1);
    __syncwarp();
    auto *signal = is_value ? s.value_ready : s.key_ready;
    if (tma) {
      if (w.lane == 0) {
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(signal + slot, 32768);
        if (is_value)
          SM90_TMA_LOAD_5D::copy(&w.value_map, signal + slot,
                                 uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), values, 0, 0, 0, w.kh,
                                 (page * 2 + 1) * 98 + (base % 784) / 8);
        else
          SM90_TMA_LOAD_5D::copy(&w.key_map, signal + slot,
                                 uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), keys, 0, base % 784, 0,
                                 w.kh, page * 2);
      }
    } else {
      auto sK = make_tensor(make_smem_ptr(keys), KLayout{});
      auto sV = make_tensor(make_smem_ptr(values), VLayout{});
      for (int i = w.lane; i < 64 * 32; i += 32) {
        int r = i / 32, c = (i % 32) * 8, pos = base + r;
        int logical = min(pos, w.last - 1), page = fa_prefill_cached_page(pages, logical);
        int64_t offset = ((int64_t(page) * 1568 + logical % 784) * 4 + w.kh) * 256 + c;
        unsigned destination =
            is_value ? __cvta_generic_to_shared(&sV(make_coord(c, r % 16), 0, r / 16))
                     : __cvta_generic_to_shared(&sK(make_coord(r, c % 16), 0, c / 16));
        auto *source = w.cache + offset + (is_value ? int64_t(784) * 4 * 256 : 0);
        int bytes = pos < w.last ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(destination),
                     "l"(source), "r"(bytes)
                     : "memory");
      }
      asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
      __syncwarp();
      if (w.lane == 0)
        Barrier::arrive(signal + slot);
    }
  }
}

__device__ __forceinline__ bool load(Shared &s, const Work &w) {
  for (int tile = 0, base = w.first; tile < w.trips; ++tile, base += 64) {
    int slot = tile & 1, phase = (tile / 2) & 1;
    auto *keys = s.key_storage + slot * cosize_v<KLayout>;
    auto *values = s.value_storage + slot * cosize_v<VLayout>;
    if (!w.split_key_value) {
      if (tile >= 2)
        wait_barrier(s.consumed[slot], phase ^ 1);
      __syncwarp();
      auto pages = fa_prefill_page_span(w.tables, w.seq, w.table_width, base, w.last);
      if (!fa_prefill_pages_valid(pages, w.cache_pages)) {
        asm volatile("trap;");
        return false;
      }
      load_combined(s, w, keys, values, slot, base, pages);
      continue;
    }
    bool tma = w.first % 16 == 0 && base + 64 <= w.last && base % 784 + 64 <= 784;
    auto pages = fa_prefill_page_span(w.tables, w.seq, w.table_width, base, w.last);
    if (!fa_prefill_pages_valid(pages, w.cache_pages)) {
      asm volatile("trap;");
      return false;
    }
    load_split(s, w, keys, values, slot, tile, phase, base, tma, pages);
  }
  return true;
}

} // namespace fa_prefill_stages
