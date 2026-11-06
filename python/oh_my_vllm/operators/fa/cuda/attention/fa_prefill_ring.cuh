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

#include "fa_ring_layout.cuh"
#include "fa_ring_transport.cuh"
#include "fa_ring_compute.cuh"
#include "fa_ring_output.cuh"

template <bool ExportBFloat16, bool RoundMax = false>
__global__ __launch_bounds__(480, 1) void fa_ring_kernel(
    const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ cache,
    const int32_t *__restrict__ tables, const int32_t *__restrict__ lengths,
    const int32_t *__restrict__ starts, float *__restrict__ partial, float *__restrict__ lse,
    int table_width, int tiles, int max_queries, int query_rows, int cache_pages, int first,
    const __grid_constant__ CUtensorMap key_map, const __grid_constant__ CUtensorMap value_map,
    int64_t l2_bytes, const __grid_constant__ CUtensorMap key16_map,
    const __grid_constant__ CUtensorMap value16_map, __nv_bfloat16 *attended) {
  using namespace fa_ring_stages;
  extern __shared__ __align__(1024) unsigned char storage[];
  Shared shared(storage);
  Work work{query,       cache,    tables,      lengths,     starts,      partial, lse,
            table_width, tiles,    max_queries, query_rows,  cache_pages, first,   key_map,
            value_map,   l2_bytes, key16_map,   value16_map, attended};
  if (!prepare(work))
    return;
  cute::TMEM::Allocator1Sm allocator;
  if (work.warp == 14)
    allocator.allocate(512, shared.tmem_base);
  if (work.tid == 0) {
    for (int index = 0; index < 22; ++index)
      initialize_barrier(shared.barriers[index], index == 10 || index == 11 ? 256
                                                 : index >= 8 && index < 14 ? 128
                                                                            : 1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  initialize_query(shared, work);
  work.trips = (work.last - work.first + 127) / 128;
  if (work.warp == 12 || work.warp == 13) {
    if (!load(shared, work))
      return;
  } else if (work.warp == 14) {
    multiply(shared, work);
  } else if (work.tid < 256) {
    softmax<RoundMax>(shared, work);
  } else {
    correct(shared, work);
  }
  __syncthreads();
  fa_ring_after_sync();
  export_output<ExportBFloat16>(shared, work, storage);
  __syncthreads();
  if (work.warp == 14) {
    fa_ring_after_sync();
    allocator.release_allocation_lock();
    allocator.free(*shared.tmem_base, 512);
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
