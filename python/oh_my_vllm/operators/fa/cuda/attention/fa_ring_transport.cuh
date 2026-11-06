#pragma once

namespace fa_ring_stages {
__device__ __forceinline__ void copy_tile(Shared &s, const Work &w, int base, int chunk, bool value,
                                          const FARingPageSpan &pages) {
  bool tma = base % 16 == 0 && base + 128 <= w.last && base % 784 + 128 <= 784;
  auto *target = value ? s.value_storage + chunk * cosize_v<VLayout>
                       : s.key_storage + chunk * cosize_v<KLayout>;
  auto *signal = value ? s.value_ready : s.key_ready;
  if (tma) {
    if (w.lane == 0) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(signal + chunk, 32768);
      if (value)
        SM90_TMA_LOAD_5D::copy(&w.value_map, signal + chunk,
                               uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), target, 0, 0, chunk * 2,
                               w.kh, (pages.first_page * 2 + 1) * 98 + (base % 784) / 8);
      else
        SM90_TMA_LOAD_5D::copy(&w.key_map, signal + chunk,
                               uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), target, 0, base % 784,
                               chunk * 2, w.kh, pages.first_page * 2);
    }
  } else if (base % 16 == 0 && base + 128 <= w.last) {
    if (w.lane == 0) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(signal + chunk, 32768);
      auto sK = make_tensor(make_smem_ptr(target), KLayout{});
      auto sV = make_tensor(make_smem_ptr(target), VLayout{});
      for (int r = 0; r < 128; r += 16) {
        int logical = base + r, page = fa_ring_cached_page(pages, logical);
        if (value) {
          auto *dest = &sV(make_coord(0, r % 16), 0, r / 16);
          SM90_TMA_LOAD_5D::copy(&w.value16_map, signal + chunk,
                                 uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), dest, 0, 0, chunk * 2,
                                 w.kh, (page * 2 + 1) * 98 + (logical % 784) / 8);
        } else
          for (int c = 0; c < 2; ++c) {
            auto *dest = &sK(make_coord(r, 0), 0, c * 4);
            SM90_TMA_LOAD_5D::copy(&w.key16_map, signal + chunk,
                                   uint64_t(TMA::CacheHintSm90::EVICT_NORMAL), dest, 0,
                                   logical % 784, chunk * 2 + c, w.kh, page * 2);
          }
      }
    }
  } else {
    auto sK = make_tensor(make_smem_ptr(target), KLayout{});
    auto sV = make_tensor(make_smem_ptr(target), VLayout{});
    for (int i = w.lane; i < 2048; i += 32) {
      int row = i / 16, col = (i % 16) * 8, pos = base + row, logical = min(pos, w.last - 1);
      int page = fa_ring_cached_page(pages, logical);
      int64_t offset =
          ((int64_t(page) * 1568 + logical % 784) * 4 + w.kh) * 256 + chunk * 128 + col;
      unsigned dest = value ? __cvta_generic_to_shared(&sV(make_coord(col, row % 16), 0, row / 16))
                            : __cvta_generic_to_shared(&sK(make_coord(row, col % 16), 0, col / 16));
      auto *source = w.cache + offset + (value ? int64_t(784) * 4 * 256 : 0);
      int bytes = pos < w.last ? 16 : 0;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dest), "l"(source),
                   "r"(bytes)
                   : "memory");
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    __syncwarp();
    if (w.lane == 0)
      Barrier::arrive(signal + chunk);
  }
}

__device__ __forceinline__ bool load(Shared &s, const Work &w) {
  bool value = w.warp == 13;
  for (int tile = 0, base = w.first; tile < w.trips; ++tile, base += 128) {
#pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
      int phase = tile & 1;
      if (tile > 0)
        wait_barrier(value ? s.value_consumed[chunk] : s.key_consumed[chunk], phase ^ 1);
      __syncwarp();
      auto pages = fa_ring_page_span(w.tables, w.seq, w.table_width, base, w.last);
      if (!fa_ring_pages_valid(pages, w.cache_pages)) {
        asm volatile("trap;");
        return false;
      }
      copy_tile(s, w, base, chunk, value, pages);
    }
  }
  return true;
}

} // namespace fa_ring_stages
