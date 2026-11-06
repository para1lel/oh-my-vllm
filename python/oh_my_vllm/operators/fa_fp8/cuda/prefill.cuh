#pragma once

#include "format.cuh"
#include "../../fa/cuda/attention/fa_prefill.cuh"
#include "../../fa/cuda/attention/fa_prefill_ring.cuh"

namespace fa_fp8 {
__device__ inline __nv_bfloat16 element(const uint8_t *codes, const float *scales,
                                        const int32_t *tables, int seq, int width, int pages,
                                        int position, int last, int plane, int head, int dim) {
  if (position >= last)
    return __float2bfloat16_rn(0.f);
  int logical = position / PageRows;
  if (position < 0 || logical >= width) {
    asm volatile("trap;");
    return __float2bfloat16_rn(0.f);
  }
  int page = tables[int64_t(seq) * width + logical];
  if (page <= 0 || page >= pages) {
    asm volatile("trap;");
    return __float2bfloat16_rn(0.f);
  }
  int64_t index = scale_index(page, plane, position % PageRows, head);
  return decode(codes[index * Width + dim], scales[index]);
}

__device__ inline void prefill_load(fa_prefill_stages::Shared &s, const fa_prefill_stages::Work &w,
                                    const uint8_t *codes, const float *scales) {
  using namespace fa_prefill_stages;
  for (int tile = 0, base = w.first; tile < w.trips; ++tile, base += 64) {
    int slot = tile & 1, phase = tile / 2 & 1;
    if (tile >= 2)
      wait_barrier(s.consumed[slot], phase ^ 1);
    __syncwarp();
    auto keys = make_tensor(make_smem_ptr(s.key_storage + slot * cosize_v<KLayout>), KLayout{});
    auto values = make_tensor(make_smem_ptr(s.value_storage + slot * cosize_v<VLayout>), VLayout{});
    for (int i = w.lane; i < 64 * 256; i += 32) {
      int row = i / 256, dim = i % 256;
      auto key = element(codes, scales, w.tables, w.seq, w.table_width, w.cache_pages, base + row,
                         w.last, 0, w.kh, dim);
      auto value = element(codes, scales, w.tables, w.seq, w.table_width, w.cache_pages, base + row,
                           w.last, 1, w.kh, dim);
      *reinterpret_cast<__nv_bfloat16 *>(&keys(make_coord(row, dim % 16), 0, dim / 16)) = key;
      *reinterpret_cast<__nv_bfloat16 *>(&values(make_coord(dim, row % 16), 0, row / 16)) = value;
    }
    cutlass::arch::fence_view_async_shared();
    fa_prefill_before_sync();
    __syncwarp();
    if (w.lane == 0)
      Barrier::arrive(s.key_ready + slot);
  }
}

template <bool Export>
__global__ void prefill(const __nv_bfloat16 *query, const uint8_t *codes, const float *scales,
                        const int32_t *tables, const int32_t *lengths, const int32_t *starts,
                        float *partial, float *lse, int width, int tiles, int max_queries, int rows,
                        int pages, int first, const __grid_constant__ CUtensorMap unused,
                        __nv_bfloat16 *attended) {
  using namespace fa_prefill_stages;
  extern __shared__ __align__(1024) unsigned char storage[];
  Shared shared(storage);
  Work work{query,       nullptr, tables, lengths, starts, partial, lse,       width,   tiles,
            max_queries, rows,    pages,  first,   unused, unused,  INT64_MAX, attended};
  if (!prepare(work))
    return;
  if (work.last > width * PageRows) {
    asm volatile("trap;");
    return;
  }
  work.split_key_value = false;
  cute::TMEM::Allocator1Sm allocator;
  if (work.warp == 9)
    allocator.allocate(512, shared.tmem_base);
  if (work.tid == 0) {
    for (int i = 0; i < 14; ++i)
      cute::initialize_barrier(shared.barriers[i], i >= 8 ? 128 : 1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  initialize_query(shared, work);
  work.trips = (work.last - work.first + 63) / 64;
  if (work.warp == 8)
    prefill_load(shared, work, codes, scales);
  else if (work.warp == 9)
    multiply(shared, work);
  else if (work.tid < 128)
    softmax(shared, work);
  else
    correct(shared, work);
  __syncthreads();
  fa_prefill_after_sync();
  export_output<Export>(shared, work);
  __syncthreads();
  if (work.warp == 9) {
    fa_prefill_after_sync();
    allocator.release_allocation_lock();
    allocator.free(*shared.tmem_base, 512);
  }
}

__device__ inline void ring_load(fa_ring_stages::Shared &s, const fa_ring_stages::Work &w,
                                 const uint8_t *codes, const float *scales) {
  using namespace fa_ring_stages;
  bool value = w.warp == 13;
  for (int tile = 0, base = w.first; tile < w.trips; ++tile, base += 128) {
#pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
      if (tile > 0)
        wait_barrier(value ? s.value_consumed[chunk] : s.key_consumed[chunk], (tile & 1) ^ 1);
      __syncwarp();
      auto keys = make_tensor(make_smem_ptr(s.key_storage + chunk * cosize_v<KLayout>), KLayout{});
      auto values =
          make_tensor(make_smem_ptr(s.value_storage + chunk * cosize_v<VLayout>), VLayout{});
      for (int i = w.lane; i < 128 * 128; i += 32) {
        int row = i / 128, dim = i % 128;
        auto decoded = element(codes, scales, w.tables, w.seq, w.table_width, w.cache_pages,
                               base + row, w.last, int(value), w.kh, chunk * 128 + dim);
        if (value)
          *reinterpret_cast<__nv_bfloat16 *>(&values(make_coord(dim, row % 16), 0, row / 16)) =
              decoded;
        else
          *reinterpret_cast<__nv_bfloat16 *>(&keys(make_coord(row, dim % 16), 0, dim / 16)) =
              decoded;
      }
      cutlass::arch::fence_view_async_shared();
      fa_ring_before_sync();
      __syncwarp();
      if (w.lane == 0)
        Barrier::arrive((value ? s.value_ready : s.key_ready) + chunk);
    }
  }
}

template <bool RoundMax>
__global__ __launch_bounds__(480, 1) void ring(const __nv_bfloat16 *query, const uint8_t *codes,
                                               const float *scales, const int32_t *tables,
                                               const int32_t *lengths, const int32_t *starts,
                                               float *partial, float *lse, int width, int tiles,
                                               int max_queries, int rows, int pages, int first,
                                               const __grid_constant__ CUtensorMap unused,
                                               __nv_bfloat16 *attended) {
  using namespace fa_ring_stages;
  extern __shared__ __align__(1024) unsigned char storage[];
  Shared shared(storage);
  Work work{query,  nullptr,   tables,      lengths, starts,  partial, lse,
            width,  tiles,     max_queries, rows,    pages,   first,   unused,
            unused, INT64_MAX, unused,      unused,  attended};
  if (!prepare(work))
    return;
  if (work.last > width * PageRows) {
    asm volatile("trap;");
    return;
  }
  cute::TMEM::Allocator1Sm allocator;
  if (work.warp == 14)
    allocator.allocate(512, shared.tmem_base);
  if (work.tid == 0) {
    for (int i = 0; i < 22; ++i)
      cute::initialize_barrier(shared.barriers[i], i == 10 || i == 11 ? 256
                                                   : i >= 8 && i < 14 ? 128
                                                                      : 1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  initialize_query(shared, work);
  work.trips = (work.last - work.first + 127) / 128;
  if (work.warp == 12 || work.warp == 13)
    ring_load(shared, work, codes, scales);
  else if (work.warp == 14)
    multiply(shared, work);
  else if (work.tid < 256)
    softmax<RoundMax>(shared, work);
  else
    correct(shared, work);
  __syncthreads();
  fa_ring_after_sync();
  export_output<true>(shared, work, storage);
  __syncthreads();
  if (work.warp == 14) {
    fa_ring_after_sync();
    allocator.release_allocation_lock();
    allocator.free(*shared.tmem_base, 512);
  }
}

inline void prefill_attend(TensorView query, TensorView codes, TensorView scales, TensorView tables,
                           TensorView lengths, TensorView starts, TensorView out, TensorView lse,
                           int queries, int first, __nv_bfloat16 *attended, cudaStream_t stream) {
  CUtensorMap unused{};
  int tiles = (queries * 6 + 127) / 128;
  auto launch = [&](auto ring_mode, auto mode) {
    constexpr bool Ring = decltype(ring_mode)::value, Mode = decltype(mode)::value;
    auto kernel = [] {
      if constexpr (Ring)
        return ring<Mode>;
      else
        return prefill<Mode>;
    }();
    constexpr int Bytes = Ring ? FARingSharedBytes : FAPrefillSharedBytes;
    static std::once_flag configured;
    std::call_once(configured, [=] {
      check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes));
    });
    kernel<<<dim3(tables.size(0) * tiles, 4, lse.size(2)), Ring ? 480 : 320, Bytes, stream>>>(
        static_cast<const __nv_bfloat16 *>(query.data_ptr()),
        static_cast<const uint8_t *>(codes.data_ptr()),
        static_cast<const float *>(scales.data_ptr()),
        static_cast<const int32_t *>(tables.data_ptr()),
        static_cast<const int32_t *>(lengths.data_ptr()),
        static_cast<const int32_t *>(starts.data_ptr()), static_cast<float *>(out.data_ptr()),
        static_cast<float *>(lse.data_ptr()), tables.size(1), tiles, queries, query.size(0),
        codes.size(0), first, unused, attended);
  };
  if (queries >= 1024)
    launch(std::true_type{}, std::true_type{});
  else if (queries >= 512)
    launch(std::true_type{}, std::false_type{});
  else if (lse.size(2) == 1)
    launch(std::false_type{}, std::true_type{});
  else
    launch(std::false_type{}, std::false_type{});
  check_cuda(cudaGetLastError());
}
} // namespace fa_fp8
