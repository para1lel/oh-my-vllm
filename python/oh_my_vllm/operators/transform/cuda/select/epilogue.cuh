/*
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Source: https://github.com/deepseek-ai/DeepSelect/tree/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18
CUDA BF16 selection core; local CUDA and tvmffi boundary.
*/

#pragma once

#include <cub/cub.cuh>
#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include "platform.cuh"

#include "types.cuh"
#include "utils.cuh"
#include "config.cuh"
#include "bits.cuh"

using transac_bar_t = select_platform::transac_bar_t;

namespace topk_select_common {

#ifdef SELECT_ENABLE_SM100
#define IS_LDG_STG_256_AVAILABLE 1
#else
#define IS_LDG_STG_256_AVAILABLE 0
#endif

// Choose the loading & storing instruction
#if IS_LDG_STG_256_AVAILABLE
// Use LDG256 / STG256 on sm100+
static constexpr uint32_t NUM_BYTES_PER_GMEM_LOAD = 256 / 8;
static constexpr uint32_t NUM_BYTES_PER_GMEM_STORE = 256 / 8;
#define LOAD_FROM_GMEM(src_ptr, dst_reg)                                                           \
  SELECT_LDG_256(src_ptr, dst_reg, ".nc", "no_allocate", "evict_last", "256B")
#define STORE_TO_GMEM(dst_ptr, src_reg)                                                            \
  SELECT_STG_256(dst_ptr, src_reg, "no_allocate", "evict_first")
#else
// Use LDG128 on sm90 and below
static constexpr uint32_t NUM_BYTES_PER_GMEM_LOAD = 128 / 8;
static constexpr uint32_t NUM_BYTES_PER_GMEM_STORE = 128 / 8;
#define LOAD_FROM_GMEM(src_ptr, dst_reg)                                                           \
  SELECT_LDG_128(src_ptr, dst_reg, ".nc", "no_allocate", "256B")
#define STORE_TO_GMEM(dst_ptr, src_reg) SELECT_STG_128(dst_ptr, src_reg, "no_allocate", "")
#endif

static constexpr uint32_t NUM_BYTES_PER_SMEM_LOAD = 16;
static constexpr uint32_t NUM_BYTES_PER_SMEM_STORE = 16;

template <uint32_t NUM_VALUES, typename ValueT>
__device__ __forceinline__ void ld_shared(ValueT res[NUM_VALUES], const ValueT *ptr) {
  constexpr uint32_t NUM_BYTES_TO_LOAD = NUM_VALUES * sizeof(ValueT);
  static_assert(NUM_BYTES_TO_LOAD == 16 || NUM_BYTES_TO_LOAD == 8 || NUM_BYTES_TO_LOAD == 4 ||
                NUM_BYTES_TO_LOAD == 2);
  uint32_t addr = cute::cast_smem_ptr_to_uint(ptr);
  if constexpr (NUM_BYTES_TO_LOAD == 16) {
    *(__int128_t *)res = select_platform::ld_shared(ptr);
  } else if constexpr (NUM_BYTES_TO_LOAD == 8) {
    asm volatile("ld.weak.shared::cta.b64 %0, [%1];\n"
                 : "=l"(*(int64_t *)res)
                 : "r"(addr)
                 : "memory");
  } else if constexpr (NUM_BYTES_TO_LOAD == 4) {
    asm volatile("ld.weak.shared::cta.b32 %0, [%1];\n"
                 : "=r"(*(int32_t *)res)
                 : "r"(addr)
                 : "memory");
  } else if constexpr (NUM_BYTES_TO_LOAD == 2) {
    asm volatile("ld.weak.shared::cta.b16 %0, [%1];\n"
                 : "=h"(*(int16_t *)res)
                 : "r"(addr)
                 : "memory");
  } else {
    __builtin_unreachable();
  }
}

template <uint32_t NUM_VALUES, typename ValueT>
__device__ __forceinline__ void ld_shared_with_loop(ValueT res[NUM_VALUES], const ValueT *ptr) {
  constexpr uint32_t NUM_BYTES_TO_LOAD = NUM_VALUES * sizeof(ValueT);
  if constexpr (NUM_BYTES_TO_LOAD <= 16) {
    ld_shared<NUM_VALUES>(res, ptr);
  } else {
    static constexpr uint32_t NUM_LOAD_VALUES_THIS_ROUND = 16 / sizeof(ValueT);
    static_assert(16 % sizeof(ValueT) == 0);
    ld_shared<NUM_LOAD_VALUES_THIS_ROUND>(res, ptr);
    ld_shared_with_loop<NUM_VALUES - NUM_LOAD_VALUES_THIS_ROUND>(res + NUM_LOAD_VALUES_THIS_ROUND,
                                                                 ptr + NUM_LOAD_VALUES_THIS_ROUND);
  }
}

template <uint32_t NUM_VALUES, typename ValueT>
__device__ __forceinline__ void st_global(ValueT *ptr, ValueT src[NUM_VALUES]) {
  constexpr uint32_t NUM_BYTES_TO_STORE = NUM_VALUES * sizeof(ValueT);
  static_assert((IS_LDG_STG_256_AVAILABLE && NUM_BYTES_TO_STORE == 32) ||
                NUM_BYTES_TO_STORE == 16 || NUM_BYTES_TO_STORE == 8 || NUM_BYTES_TO_STORE == 4 ||
                NUM_BYTES_TO_STORE == 2);
  if constexpr (IS_LDG_STG_256_AVAILABLE && NUM_BYTES_TO_STORE == 32) {
    SELECT_STG_256(ptr, src, "no_allocate", "evict_first");
  } else if constexpr (NUM_BYTES_TO_STORE == 16) {
    SELECT_STG_128(ptr, src, "no_allocate", "");
  } else if constexpr (NUM_BYTES_TO_STORE == 8) {
    *(uint64_t *)ptr = *(uint64_t *)src;
  } else if constexpr (NUM_BYTES_TO_STORE == 4) {
    *(uint32_t *)ptr = *(uint32_t *)src;
  } else if constexpr (NUM_BYTES_TO_STORE == 2) {
    *(uint16_t *)ptr = *(uint16_t *)src;
  } else {
    __builtin_unreachable();
  }
}

template <typename Config, uint32_t MAX_TOPK, uint32_t NUM_THREADS, uint32_t TARGET_OCCUPANCY>
struct EpilogueRunner {
  using ValueT = typename Config::ValueT;
  using OutIdxT = typename Config::OutIdxT;
  static_assert(cute::is_same_v<ValueT, float> || cute::is_same_v<ValueT, nv_bfloat16>);
  static_assert(cute::is_same_v<OutIdxT, int32_t> || cute::is_same_v<OutIdxT, int64_t>);
  using UIntValueT = cute::conditional_t<cute::is_same_v<ValueT, float>, uint32_t, uint16_t>;

  static constexpr uint32_t NUM_WARPS = NUM_THREADS / 32;
  static_assert(NUM_THREADS % 32 == 0);
  static_assert(NUM_WARPS % 2 == 0);

  static constexpr uint32_t NUM_VALUES_PER_LOAD_STORE = NUM_BYTES_PER_GMEM_STORE / sizeof(ValueT);
  static constexpr uint32_t NUM_VALUES_PER_THREAD_FOR_SORT = MAX_TOPK / NUM_THREADS;
  static_assert(MAX_TOPK % NUM_THREADS == 0);
  static constexpr uint32_t NUM_REGS_PER_THREAD =
      (65536 / (NUM_THREADS * TARGET_OCCUPANCY)) / 8 * 8;
  using BlockRadixSortT =
      cub::BlockRadixSort<UIntValueT, NUM_THREADS, NUM_VALUES_PER_THREAD_FOR_SORT, uint32_t, 4,
                          !(NUM_REGS_PER_THREAD <= 80 &&
                            NUM_VALUES_PER_THREAD_FOR_SORT >=
                                16) // MemoizeOuterScan. Setting this to `false` can reduce register
                                    // pressure. We use a simple heuristic here
                          >;
  // Gated on `sorted_value`: the sort only runs in the `sorted_value` branch, so this keeps the
  // temp storage out of the smem plan of `sorted_value == false` kernels.
  using BlockRadixSortTempStorageT =
      cute::conditional_t<Config::sorted_value, typename BlockRadixSortT::TempStorage, char>;

  template <bool IS_SHORTCUT>
  static __device__ __forceinline__ void
  store_shortcut_output(const TopkSelectArgs &args, uint32_t end_vocab_idx,
                        int32_t output_idx_offset, ValueT oob_fill_value, ValueT *input_values,
                        OutIdxT *result_indices, ValueT *result_values) {
    // Saves indices 0...end_vocab_idx (plus `output_idx_offset`) followed by
    // (args.topk-end_vocab_idx) `args.idx_oob_fill_value` into `result_indices`
    constexpr uint32_t NUM_OUTPUT_IDXS_PER_STORE = NUM_BYTES_PER_GMEM_STORE / sizeof(OutIdxT);
#pragma unroll 2
    for (uint32_t i = threadIdx.x * NUM_OUTPUT_IDXS_PER_STORE; i < args.topk;
         i += NUM_THREADS * NUM_OUTPUT_IDXS_PER_STORE) {
      OutIdxT out[NUM_OUTPUT_IDXS_PER_STORE];
      CUTE_UNROLL
      for (uint32_t j = 0; j < NUM_OUTPUT_IDXS_PER_STORE; ++j)
        out[j] =
            i + j < end_vocab_idx ? (OutIdxT)(i + j) + output_idx_offset : args.idx_oob_fill_value;
      STORE_TO_GMEM(result_indices + i, out);
    }

    // Save values into `result_values`, if `RETURN_VALUE` is `True`
    if constexpr (Config::return_value) {
#pragma unroll 2
      for (uint32_t i = threadIdx.x * NUM_VALUES_PER_LOAD_STORE; i < args.topk;
           i += NUM_THREADS * NUM_VALUES_PER_LOAD_STORE) {
        ValueT values[NUM_VALUES_PER_LOAD_STORE];
        if (i < end_vocab_idx) {
          LOAD_FROM_GMEM(input_values + i, values);
        }
        CUTE_UNROLL
        for (uint32_t j = 0; j < NUM_VALUES_PER_LOAD_STORE; ++j)
          values[j] = i + j < end_vocab_idx ? values[j] : oob_fill_value;
        STORE_TO_GMEM(result_values + i, values);
      }
    }
  }

  static __device__ __forceinline__ void
  store_selected_output(const TopkSelectArgs &args, int32_t output_idx_offset,
                        ValueT *smem_value_buf, uint32_t *smem_index_buf, OutIdxT *result_indices,
                        ValueT *result_values) {
    // Load & cast & save indices
    constexpr uint32_t NUM_OUTPUT_IDX_PER_ROUND = cute::min(
        NUM_BYTES_PER_SMEM_STORE / sizeof(uint32_t), NUM_BYTES_PER_GMEM_STORE / sizeof(OutIdxT));
#pragma unroll 2
    for (uint32_t i = threadIdx.x * NUM_OUTPUT_IDX_PER_ROUND; i < args.topk;
         i += NUM_THREADS * NUM_OUTPUT_IDX_PER_ROUND) {
      uint32_t indices_u32[NUM_OUTPUT_IDX_PER_ROUND];
      OutIdxT out[NUM_OUTPUT_IDX_PER_ROUND];
      ld_shared<NUM_OUTPUT_IDX_PER_ROUND>(indices_u32, smem_index_buf + i);
      CUTE_UNROLL
      for (uint32_t j = 0; j < NUM_OUTPUT_IDX_PER_ROUND; ++j)
        out[j] = (OutIdxT)indices_u32[j] + output_idx_offset;
      st_global<NUM_OUTPUT_IDX_PER_ROUND>(result_indices + i, out);
    }

    if constexpr (Config::return_value) {
      // Load & store values
      constexpr uint32_t NUM_VALUES_PER_ROUND =
          cute::min(NUM_BYTES_PER_SMEM_STORE, NUM_BYTES_PER_GMEM_STORE) / sizeof(ValueT);
#pragma unroll 2
      for (uint32_t i = threadIdx.x * NUM_VALUES_PER_ROUND; i < args.topk;
           i += NUM_THREADS * NUM_VALUES_PER_ROUND) {
        UIntValueT values[NUM_VALUES_PER_ROUND];
        ld_shared<NUM_VALUES_PER_ROUND>(values, (UIntValueT *)(smem_value_buf + i));
        st_global<NUM_VALUES_PER_ROUND>((UIntValueT *)(result_values + i), values);
      }
    }
  }

  template <bool IS_SHORTCUT>
  static __device__ __forceinline__ void
  sort_and_store_output(const TopkSelectArgs &args, uint32_t end_vocab_idx,
                        int32_t output_idx_offset, ValueT oob_fill_value, ValueT *smem_value_buf,
                        uint32_t *smem_index_buf, OutIdxT *result_indices, ValueT *result_values,
                        BlockRadixSortTempStorageT &radix_sort_temp_storage) {
    // Sort
    UIntValueT local_values[NUM_VALUES_PER_THREAD_FOR_SORT];
    uint32_t local_indices[NUM_VALUES_PER_THREAD_FOR_SORT];
    uint32_t thread_offset = threadIdx.x * NUM_VALUES_PER_THREAD_FOR_SORT;
    ld_shared_with_loop<NUM_VALUES_PER_THREAD_FOR_SORT>(
        local_values, (UIntValueT *)smem_value_buf + threadIdx.x * NUM_VALUES_PER_THREAD_FOR_SORT);
    ld_shared_with_loop<NUM_VALUES_PER_THREAD_FOR_SORT>(
        local_indices, smem_index_buf + threadIdx.x * NUM_VALUES_PER_THREAD_FOR_SORT);
    CUTE_UNROLL
    for (uint32_t i = 0; i < NUM_VALUES_PER_THREAD_FOR_SORT; ++i)
      local_values[i] = distort(local_values[i]);
    if (IS_SHORTCUT || args.topk < MAX_TOPK) {
      uint32_t limit = IS_SHORTCUT ? end_vocab_idx : args.topk;
      CUTE_UNROLL
      for (uint32_t i = 0; i < NUM_VALUES_PER_THREAD_FOR_SORT; ++i)
        local_values[i] = thread_offset + i < limit ? local_values[i] : 0;
    }
    __syncthreads(); // To allow `radix_sort_temp_storage` overlap with `smem_value_buf` and
                     // `smem_index_buf`
    BlockRadixSortT(radix_sort_temp_storage).SortDescending(local_values, local_indices);

    // Store directly to gmem (no smem buffering)
    // For common cases, our MAX_TOPK <= 1024 and NUM_THREADS >= 256, so we have
    // NUM_VALUES_PER_THREAD_FOR_SORT <= 8, which means each thread only need to call STG once. So
    // we don't use SMEM to buffer here
    if (thread_offset < args.topk) {
      CUTE_UNROLL
      for (uint32_t i = 0; i < NUM_VALUES_PER_THREAD_FOR_SORT; ++i)
        local_values[i] =
            !IS_SHORTCUT || threadIdx.x * NUM_VALUES_PER_THREAD_FOR_SORT + i < end_vocab_idx
                ? un_distort(local_values[i])
                : *(UIntValueT *)&oob_fill_value;
      OutIdxT local_indices_new[NUM_VALUES_PER_THREAD_FOR_SORT];
      CUTE_UNROLL
      for (uint32_t i = 0; i < NUM_VALUES_PER_THREAD_FOR_SORT; ++i)
        local_indices_new[i] = (OutIdxT)((int32_t)local_indices[i] + output_idx_offset);
      auto store = [&]<uint32_t NUM_VALUES, typename T>(T *dst, T src[NUM_VALUES]) {
        constexpr uint32_t NUM_BYTES_TO_STORE = NUM_VALUES * sizeof(T);
        if constexpr (NUM_BYTES_TO_STORE <= NUM_BYTES_PER_GMEM_STORE) {
          // Don't need to do OOB check because the output array is at least padded to the maximum
          // width of global store.
          st_global<NUM_VALUES>(dst, src);
        } else {
          static_assert(NUM_BYTES_TO_STORE % NUM_BYTES_PER_GMEM_STORE == 0);
          CUTE_UNROLL
          for (uint32_t i = 0; i < NUM_BYTES_TO_STORE; i += NUM_BYTES_PER_GMEM_STORE) {
            uint32_t elem_offset = i / sizeof(T);
            if (thread_offset + elem_offset < args.topk)
              STORE_TO_GMEM(dst + elem_offset, src + elem_offset);
          }
        }
      };
      store.template operator()<NUM_VALUES_PER_THREAD_FOR_SORT>(
          (UIntValueT *)result_values + thread_offset, local_values);
      store.template operator()<NUM_VALUES_PER_THREAD_FOR_SORT>(result_indices + thread_offset,
                                                                local_indices_new);
    }
  }

  template <bool IS_SHORTCUT> // "shortcut" means `end_vocab_idx` <= `topk`
  static __device__ __forceinline__ void
  topk_select_epilogue(typename Config::ValueT *smem_value_buf, // [MAX_TOPK]
                       uint32_t *smem_index_buf,                // [MAX_TOPK]
                       const TopkSelectArgs &args, uint32_t batch_idx, uint32_t end_vocab_idx,
                       uint32_t warp_idx, BlockRadixSortTempStorageT &radix_sort_temp_storage) {
    // General epilogue for topk select
    // This epilogue has two modes: shortcut mode and non-shortcut mode.
    // In shortcut mode, it
    //  - Generates indices
    //  - Load values from global memory, if `Config::return_value` is `true`
    //  - Perform sorting, if `Config::sorted_value` is `true`
    //  - Save indices & values to global memory
    // In non-shortcut mode, it
    //  - Assume end_vocab_idx > topk holds, i.e. there are enough values for topk selection
    //  - Assume topk indices locate at smem_index_buf[:topk] and topk values locate at
    //  smem_value_buf[:topk] (an extra `__syncthreads()` might be necessary)
    //  - Perform sorting, if `Config::sorted_value` is `true`
    //  - Save indices & values to global memory (values are un-distorted if `UNDISTORT_VALUES` is
    //  `true`)

    ValueT *input_values =
        Config::return_value ? (ValueT *)args.input + (uint64_t)batch_idx * args.stride_input_batch
                             : nullptr;
    OutIdxT *result_indices =
        (OutIdxT *)args.output_index + (uint64_t)batch_idx * args.stride_output_index_batch;
    ValueT *result_values =
        Config::return_value
            ? (ValueT *)args.output_value + (uint64_t)batch_idx * args.stride_output_value_batch
            : nullptr;
    int32_t output_idx_offset =
        args.output_idx_offset != nullptr ? __ldg(args.output_idx_offset + batch_idx) : 0;
    ValueT oob_fill_value =
        Config::return_value && IS_SHORTCUT ? (ValueT)args.value_oob_fill_value : (ValueT)0.0f;

    if constexpr (!Config::sorted_value) {
      if constexpr (IS_SHORTCUT) {
        store_shortcut_output<IS_SHORTCUT>(args, end_vocab_idx, output_idx_offset, oob_fill_value,
                                           input_values, result_indices, result_values);
      } else {
        store_selected_output(args, output_idx_offset, smem_value_buf, smem_index_buf,
                              result_indices, result_values);
      }
    } else {
      // Load indices & values into shared memory
      if constexpr (IS_SHORTCUT) {
        if (warp_idx < NUM_WARPS / 2) {
#pragma unroll 8
          for (uint32_t i = threadIdx.x; i < args.topk; i += NUM_THREADS / 2) {
            smem_index_buf[i] = i < end_vocab_idx ? i : args.idx_oob_fill_value - output_idx_offset;
          }
        } else {
          constexpr uint32_t NUM_VALUES_PER_LDG128 = NUM_BYTES_PER_SMEM_LOAD / sizeof(ValueT);
#pragma unroll 2
          for (uint32_t i = (threadIdx.x - NUM_THREADS / 2) * NUM_VALUES_PER_LDG128;
               i < end_vocab_idx; i += (NUM_THREADS / 2) * NUM_VALUES_PER_LDG128) {
            ValueT values[NUM_VALUES_PER_LDG128];
            SELECT_LDG_128(input_values + i, values, ".nc", "no_allocate", "256B");
            select_platform::st_shared(smem_value_buf + i, *(__int128_t *)values);
          }
        }
        __syncthreads();
      }

      sort_and_store_output<IS_SHORTCUT>(args, end_vocab_idx, output_idx_offset, oob_fill_value,
                                         smem_value_buf, smem_index_buf, result_indices,
                                         result_values, radix_sort_temp_storage);
    }
  }
};

} // namespace topk_select_common
