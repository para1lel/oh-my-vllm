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

#include "fa_prefill_layout.cuh"
#include "fa_prefill_transport.cuh"
#include "fa_prefill_compute.cuh"
#include "fa_prefill_output.cuh"

template <bool ExportBFloat16>
__global__ void fa_prefill_kernel(
    const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ cache,
    const int32_t *__restrict__ tables, const int32_t *__restrict__ lengths,
    const int32_t *__restrict__ starts, float *__restrict__ partial, float *__restrict__ lse,
    int table_width, int tiles, int max_queries, int query_rows, int cache_pages, int first,
    const __grid_constant__ CUtensorMap key_map, const __grid_constant__ CUtensorMap value_map,
    int64_t l2_bytes, __nv_bfloat16 *attended) {
  using namespace fa_prefill_stages;
  extern __shared__ __align__(1024) unsigned char storage[];
  Shared shared(storage);
  Work work{query, cache,       tables,    lengths,     starts,     partial,
            lse,   table_width, tiles,     max_queries, query_rows, cache_pages,
            first, key_map,     value_map, l2_bytes,    attended};
  if (!prepare(work))
    return;
  cute::TMEM::Allocator1Sm allocator;
  if (work.warp == 9)
    allocator.allocate(512, shared.tmem_base);
  if (work.tid == 0) {
    for (int index = 0; index < 14; ++index)
      initialize_barrier(shared.barriers[index], index >= 8 ? 128 : 1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  initialize_query(shared, work);
  work.trips = (work.last - work.first + 63) / 64;
  if (work.warp == 8) {
    if (!load(shared, work))
      return;
  } else if (work.warp == 9) {
    multiply(shared, work);
  } else if (work.tid < 128) {
    softmax(shared, work);
  } else {
    correct(shared, work);
  }
  __syncthreads();
  fa_prefill_after_sync();
  export_output<ExportBFloat16>(shared, work);
  __syncthreads();
  if (work.warp == 9) {
    fa_prefill_after_sync();
    allocator.release_allocation_lock();
    allocator.free(*shared.tmem_base, 512);
  }
}
