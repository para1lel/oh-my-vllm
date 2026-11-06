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
#include "epilogue.cuh"

namespace topk_select_common {
template <typename Config> class TopkSelectKernelBase {
public:
  using ValueT = typename Config::ValueT;
  using OutIdxT = typename Config::OutIdxT;
  static_assert(std::is_same_v<ValueT, nv_bfloat16> || std::is_same_v<ValueT, float>);

  static constexpr uint32_t PLACEHOLDER_U32 =
      std::is_same_v<ValueT, nv_bfloat16> ? 0xff80 : 0xff800000; // -INF
  static constexpr uint64_t PLACEHOLDER_B64 =
      std::is_same_v<ValueT, nv_bfloat16> ? 0xff80ff80ff80ff80 : 0xff800000ff800000;
  static constexpr uint64_t PLACEHOLDER_PAIR = (uint64_t)PLACEHOLDER_U32 << 32;

  static constexpr uint32_t TARGET_OCCUPANCY = Config::target_occupancy;
  static constexpr uint32_t NUM_THREADS = Config::num_threads;
  static constexpr uint32_t NUM_WARPS = NUM_THREADS / 32;
  static constexpr uint32_t MAX_TOPK = Config::max_topk;
  static_assert(NUM_THREADS == 256 || NUM_THREADS == 512);

  static constexpr uint32_t NUM_ELEMS_PER_128b = 16 / sizeof(ValueT);
  static constexpr uint32_t NUM_UINT32_PER_128b = 4;
  static constexpr uint32_t NUM_ELEMS_PER_ROUND = Config::elements_per_round;
  static constexpr uint32_t NUM_ELEMS_PER_THREAD_PER_ROUND =
      NUM_ELEMS_PER_ROUND / NUM_THREADS; // TODO Unify "values", "elements", and "elems" to "elems"
  static constexpr uint32_t NUM_128b_PER_THREAD_PER_ROUND =
      NUM_ELEMS_PER_THREAD_PER_ROUND / NUM_ELEMS_PER_128b;
  static_assert(NUM_ELEMS_PER_ROUND % (NUM_ELEMS_PER_128b * NUM_THREADS) == 0);
  static constexpr uint32_t ELEMS_PER_THREAD_PER_ROUND_MASK = NUM_ELEMS_PER_THREAD_PER_ROUND - 1;
  // NUM_128b_PER_THREAD_PER_ROUND is the canonical name; no separate NUM_UNITS alias is kept.

  static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND == 16);

  // Each segment/window contains this many elements. It is the granularity used by both the
  // pseudo-random visit order and the sorted-index epilogue.
  static constexpr uint32_t NUM_ELEMS_PER_SEG = Config::elements_per_segment; // shuffle rule
  static_assert(NUM_ELEMS_PER_SEG == 512);
  // The last NUM_TAIL_ELEMS elements of a row are always visited in their original order.
  // 4096 keeps the tail small enough to fit alongside the init-window perm segments.
  static constexpr uint32_t NUM_TAIL_ELEMS = 4096;
  static constexpr uint32_t NUM_TAIL_SEGS = NUM_TAIL_ELEMS / NUM_ELEMS_PER_SEG;
  // The init window is 32 KiB of input data
  static constexpr uint32_t NUM_ELEMS_IN_INIT_WINDOW = 32768 / sizeof(ValueT);
  static constexpr uint32_t NUM_INIT_ROUNDS_MAX = NUM_ELEMS_IN_INIT_WINDOW / NUM_ELEMS_PER_ROUND;
  static_assert(NUM_ELEMS_IN_INIT_WINDOW % NUM_ELEMS_PER_ROUND == 0);
  static_assert(NUM_ELEMS_IN_INIT_WINDOW % NUM_ELEMS_PER_SEG == 0);

  static constexpr uint32_t NUM_SEGS_PER_ROUND = NUM_ELEMS_PER_ROUND / NUM_ELEMS_PER_SEG;
  static_assert(NUM_ELEMS_PER_ROUND % NUM_ELEMS_PER_SEG == 0);
  static_assert(NUM_ELEMS_PER_SEG % NUM_ELEMS_PER_128b == 0);
  static constexpr uint32_t NUM_128b_PER_SEG = NUM_ELEMS_PER_SEG / NUM_ELEMS_PER_128b;
  static_assert(NUM_THREADS % NUM_128b_PER_SEG == 0);
  static_assert(NUM_128b_PER_SEG % 32 == 0); // every lane in a warp should in the same seg

  static constexpr uint32_t NUM_ISSUE_WARPS = NUM_SEGS_PER_ROUND < 4 ? NUM_SEGS_PER_ROUND : 4;
  static constexpr uint32_t NUM_SEGS_PER_ISSUE_WARP = NUM_SEGS_PER_ROUND / NUM_ISSUE_WARPS;
  static_assert(NUM_SEGS_PER_ROUND % NUM_ISSUE_WARPS == 0);
  static_assert(NUM_TAIL_SEGS % NUM_ISSUE_WARPS == 0);
  static_assert(NUM_WARPS >= NUM_ISSUE_WARPS);

  static constexpr uint32_t NUM_TMA_LOAD_BUFS = Config::tma_buffer_depth;
  static constexpr uint32_t TMA_PREFETCH_DEPTH = NUM_TMA_LOAD_BUFS - 1;
  static_assert(NUM_TMA_LOAD_BUFS >= 2);

  static constexpr uint32_t RECONSTRUCT_THRESHOLD = Config::reconstruct_threshold;
  static_assert(RECONSTRUCT_THRESHOLD >= MAX_TOPK);

  // Parameters of the pseudo-random permutation over permuted segments:
  //   permuted_id = (linear_id * (PERM_MUL_PRIME % perm_len) + PERM_ADD_BASE) % perm_len.
  static constexpr uint32_t PERM_ADD_BASE = 0x22262226u;
  static constexpr uint32_t PERM_MUL_PRIME = 0xB559EB75u;

  static constexpr uint32_t NUM_SEGS_IN_INIT_WINDOW = NUM_ELEMS_IN_INIT_WINDOW / NUM_ELEMS_PER_SEG;
  // Number of permuted segments included in the init window after reserving room for the tail.
  static constexpr uint32_t NUM_PERM_SEGS_IN_INIT_WINDOW =
      (NUM_ELEMS_IN_INIT_WINDOW - NUM_TAIL_ELEMS) / NUM_ELEMS_PER_SEG;
  static_assert(NUM_PERM_SEGS_IN_INIT_WINDOW % NUM_TAIL_SEGS == 0);
  static_assert(NUM_PERM_SEGS_IN_INIT_WINDOW > 0);
  static constexpr uint32_t NUM_UINT32_IN_INIT_WINDOW_PER_THREAD =
      NUM_ELEMS_IN_INIT_WINDOW * sizeof(ValueT) / sizeof(uint32_t) / NUM_THREADS;
  static constexpr uint32_t NUM_128b_INIT_PER_THREAD =
      NUM_UINT32_IN_INIT_WINDOW_PER_THREAD / NUM_UINT32_PER_128b;
  static_assert(NUM_ELEMS_IN_INIT_WINDOW % (sizeof(uint32_t) / sizeof(ValueT) * NUM_THREADS) == 0);
  static_assert(NUM_UINT32_IN_INIT_WINDOW_PER_THREAD % (16 * sizeof(ValueT) / sizeof(uint32_t)) ==
                0);
  static_assert(2 * NUM_UINT32_IN_INIT_WINDOW_PER_THREAD <= 256);

  static constexpr uint32_t NUM_EXTRA_SLOTS = RECONSTRUCT_THRESHOLD + NUM_ELEMS_PER_ROUND;

  static constexpr uint32_t NUM_RECONSTRUCT_RADIX_BITS = 8;
  static constexpr uint32_t NUM_RECONSTRUCT_BUCKETS = 1u << NUM_RECONSTRUCT_RADIX_BITS;
  // Each row also holds one sink slot (index NUM_RECONSTRUCT_BUCKETS) that the LSB histogram sends
  // the out-of-bucket elements to; the +4 rounds a row up to a multiple of 16 bytes so that both
  // rows stay 16B-aligned, since the reader relies on for its 128-bit shared-memory loads.
  static constexpr uint32_t NUM_RECONSTRUCT_BUCKET_SLOTS = NUM_RECONSTRUCT_BUCKETS + 4;
  static_assert(sizeof(uint32_t) * NUM_RECONSTRUCT_BUCKET_SLOTS % 16 == 0);
  static_assert(MAX_TOPK + NUM_EXTRA_SLOTS <= 0xFFFF);
  static_assert(NUM_EXTRA_SLOTS % 2 == 0);

  static constexpr uint32_t NUM_RECONSTRUCT_UNITS_MAX = (MAX_TOPK + NUM_EXTRA_SLOTS) / 2;

  using EpilogueT = EpilogueRunner<Config, MAX_TOPK, NUM_THREADS, TARGET_OCCUPANCY>;
  static_assert(sizeof(typename EpilogueT::BlockRadixSortTempStorageT) <=
                NUM_EXTRA_SLOTS * sizeof(uint64_t));

  static_assert(NUM_EXTRA_SLOTS * sizeof(uint64_t) >= NUM_ELEMS_IN_INIT_WINDOW * sizeof(ValueT),
                "the init window must fit in the extra pairs region");
  static_assert(NUM_EXTRA_SLOTS * sizeof(uint64_t) / sizeof(uint32_t) >=
                    MAX_VOCAB_SIZE / NUM_ELEMS_PER_SEG,
                "the index sort's window delta table must fit in the extra pairs region");
  // array `incoming_topk_pairs` will be used as
  //     1. tma buffer during init phase
  //     2. extra array during main phase
  //     3. windows_delta counter during epilogue phase
  static_assert(2 * MAX_TOPK * sizeof(uint64_t) % 1024 == 0);

  static constexpr uint32_t SWIZZLE_SHIFT = 3;
  static constexpr uint32_t SWIZZLE_MASK = 7 * (16 / sizeof(ValueT));

  static constexpr uint32_t NUM_BYTES_PER_TMA_ROW =
      128; // Limit the innermost box dim of TMA to 1) prevent OOB 2) swizzling has limitations on
           // the innermost box dim
  static_assert(NUM_BYTES_PER_TMA_ROW <= INPUT_STRIDE_ALIGNMENT_REQUIREMENT); // To prevent OOB
  static constexpr uint32_t NUM_ELEMS_PER_TMA_ROW = NUM_BYTES_PER_TMA_ROW / sizeof(ValueT);
  static_assert(NUM_ELEMS_PER_SEG % NUM_ELEMS_PER_TMA_ROW == 0);
  static constexpr uint32_t NUM_TMA_ROWS_PER_SEG = NUM_ELEMS_PER_SEG / NUM_ELEMS_PER_TMA_ROW;
  static_assert(SWIZZLE_MASK == ((1u << SWIZZLE_SHIFT) - 1) * NUM_ELEMS_PER_128b);
  static_assert(NUM_ELEMS_PER_TMA_ROW == (1u << SWIZZLE_SHIFT) * NUM_ELEMS_PER_128b);
  static_assert(NUM_TMA_ROWS_PER_SEG % (1u << SWIZZLE_SHIFT) == 0);

  static_assert(NUM_SEGS_PER_ROUND == NUM_WARPS);
  static_assert(NUM_SEGS_PER_ROUND >= NUM_TAIL_SEGS);
  static_assert(NUM_ELEMS_PER_SEG % (Config::elements_per_round / NUM_WARPS) == 0);

  struct TmaParams {
    CUtensorMap tensor_map;
  };

  static CUtensorMap make_topk_tensor_map(const TopkSelectArgs &args) {
    static_assert(INPUT_STRIDE_ALIGNMENT_REQUIREMENT >= NUM_ELEMS_PER_TMA_ROW * sizeof(ValueT));
    constexpr CUtensorMapDataType dtype = std::is_same_v<ValueT, nv_bfloat16>
                                              ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
                                              : CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    return select_platform::make_tensor_map(
        // Split args.vocab_size into two dims because
        //   - TMA requires innermost box dim <= swizzle size
        //   - To avoid OOB as we have `INPUT_STRIDE_ALIGNMENT_REQUIREMENT`
        {NUM_ELEMS_PER_TMA_ROW,
         select_platform::ceil_div((uint64_t)args.vocab_size, (uint64_t)NUM_ELEMS_PER_TMA_ROW),
         args.batch_size},
        select_platform::make_stride_helper<uint64_t>(
            {NUM_ELEMS_PER_TMA_ROW, args.stride_input_batch}, sizeof(ValueT)),
        {NUM_ELEMS_PER_TMA_ROW, NUM_TMA_ROWS_PER_SEG, 1}, args.input, dtype,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
  }

  struct SharedMemoryPlanBase {
    // bf16: pair(64b) = unused(16b) | value(16b) | index(32b)
    // fp32: pair(64b) = value(32b) | index(32b)
    // candidate[0]/candidate[1] are the A/B buffers swapped on each reconstruct;
    // incoming holds the pairs scanned since the last reconstruct.
    CUTE_ALIGNAS(1024) uint64_t surviving_topk_pairs[2][MAX_TOPK];
    CUTE_ALIGNAS(1024) uint64_t incoming_topk_pairs[NUM_EXTRA_SLOTS];
    CUTE_ALIGNAS(1024) ValueT tma_load_buf[NUM_TMA_LOAD_BUFS][NUM_ELEMS_PER_ROUND];
    transac_bar_t tma_load_full_bar[NUM_TMA_LOAD_BUFS];
    transac_bar_t init_full_bar[NUM_INIT_ROUNDS_MAX];
    uint32_t warp_cnt[NUM_WARPS];
    uint32_t reconstruct_pivot_bucket;
    uint32_t reconstruct_num_should_select;
    CUTE_ALIGNAS(16) uint32_t reconstruct_bucket_counter[2][NUM_RECONSTRUCT_BUCKET_SLOTS];
  };

  // Map a linear visit index to the actual permuted segment id.
  static __device__ __forceinline__ uint32_t get_permuted_seg_idx(uint32_t linear_id,
                                                                  uint32_t perm_len,
                                                                  uint32_t perm_mul) {
    return ((linear_id % perm_len) * perm_mul + PERM_ADD_BASE) % perm_len;
  }

  static __device__ __forceinline__ void advance_perm_state(uint32_t &state, uint32_t stride,
                                                            uint32_t perm_len) {
    state += stride;
    if (state >= perm_len)
      state -= perm_len;
  }

  static __device__ __forceinline__ uint32_t sw_msk(uint32_t e) {
    return (e >> SWIZZLE_SHIFT) & SWIZZLE_MASK;
  }

  static __device__ __forceinline__ uint32_t sw_elem(uint32_t e) { return e ^ sw_msk(e); }

  static __device__ __forceinline__ uint32_t sw_b128(uint32_t u) { return u ^ ((u >> 3) & (7)); }

  struct EqGtPrefix {
    uint32_t start_pos_in_collector;
    uint32_t eq_quota;
  };

  // Given local > pivot and == pivot counts, compute this thread's output prefix (start position in
  // the final output buffer) and how many equal elements it may still append (eq_quota). Should be
  // called by all threads cnt_gt and cnt_eq must be < 65536.
  static __device__ __forceinline__ EqGtPrefix
  compute_equal_quota_and_prefix(uint32_t cnt_gt, uint32_t cnt_eq, uint32_t topk, uint32_t warp_idx,
                                 uint32_t lane_idx, uint32_t *warp_cnt) {
    static_assert(NUM_WARPS <= 32);
    uint32_t cnt_packed = (cnt_gt << 16) | cnt_eq;
    uint32_t warp_total_packed = __reduce_add_sync(0xFFFFFFFF, cnt_packed);
    if (lane_idx == 0) {
      warp_cnt[warp_idx] = warp_total_packed;
    }
    __syncthreads();

    uint32_t off_lane_packed = warp_level_exclusive_prefix_sum(cnt_packed, lane_idx);
    uint32_t stored_warp_packed = lane_idx < NUM_WARPS ? warp_cnt[lane_idx] : 0u;
    uint32_t off_warp_packed =
        __reduce_add_sync(0xFFFFFFFF, lane_idx < warp_idx ? stored_warp_packed : 0u);
    uint32_t total_packed = __reduce_add_sync(0xFFFFFFFF, stored_warp_packed);
    uint32_t num_total_ge = total_packed >> 16;
    uint32_t num_total_eq_quota = topk - num_total_ge;
    uint32_t off_packed = off_warp_packed + off_lane_packed;
    uint32_t num_gt_before = off_packed >> 16;
    uint32_t num_eq_before = off_packed & 0xFFFFu;
    uint32_t eq_quota =
        num_total_eq_quota > num_eq_before ? num_total_eq_quota - num_eq_before : 0u;
    uint32_t num_elems_should_select_before =
        num_gt_before + min(num_eq_before, num_total_eq_quota);
    return {num_elems_should_select_before, eq_quota};
  }

  // Load this thread's slice from the swizzled smem round buffer into registers.
  template <uint32_t NUM_128b>
  static __device__ __forceinline__ void load_swizzled_slice(
      ValueT dst[NUM_128b * NUM_ELEMS_PER_128b], const ValueT *smem_buf,
      uint32_t
          smem_slice_elem_offset, // element offset into smem_buf where this thread's slice starts
      uint32_t unit_swizzle_mask, // swizzle mask applied to each 128-bit unit inside the slice
      uint32_t first_128b_unit_in_slice =
          0 // which 128-bit unit of this slice to start loading from
  ) {
    CUTE_UNROLL
    for (uint32_t u = 0; u < NUM_128b; ++u) {
      ld_shared<NUM_ELEMS_PER_128b>(
          dst + u * NUM_ELEMS_PER_128b,
          smem_buf + smem_slice_elem_offset +
              (((first_128b_unit_in_slice + u) * NUM_ELEMS_PER_128b) ^ unit_swizzle_mask));
    }
  }

  // Find the bucket id that the topk-th element resides in, and get how many elements we should
  // select from that bucket Save result to `smem.reconstruct_pivot_bucket` and
  // `smem.reconstruct_num_should_select` If `CHECK_IF_SHOULD_SELECT_WHOLE_BUCKET`, return whether
  // we should select all elements in that bucket. Should be called by a whole warp
  template <bool CHECK_IF_SHOULD_SELECT_WHOLE_BUCKET>
  static __device__ __forceinline__ bool find_pivot_in_histogram(SharedMemoryPlanBase &smem,
                                                                 const uint32_t *bucket_counter,
                                                                 uint32_t topk, uint32_t lane_idx) {
    bool should_select_whole_bucket = false;
    // Each lane loads 8 counters
    static_assert(NUM_RECONSTRUCT_BUCKETS == 32 * 8);
    uint32_t counts[8];
    const uint32_t *bucket_ptr = bucket_counter + lane_idx * 8;
    ld_shared<4>(counts, bucket_ptr);
    ld_shared<4>(counts + 4, bucket_ptr + 4);

    uint32_t local_sum = 0;
    CUTE_UNROLL
    for (uint32_t j = 0; j < 8; ++j)
      local_sum += counts[j];

    // Get the inclusive suffix sum of the histogram
    uint32_t suffix_count[9];
    suffix_count[8] = warp_level_exclusive_suffix_sum(local_sum, lane_idx);
    CUTE_UNROLL
    for (int32_t j = 7; j >= 0; --j)
      suffix_count[j] = suffix_count[j + 1] + counts[j];

    if (suffix_count[8] < topk && topk <= suffix_count[0]) {
      uint32_t j = 0;
      CUTE_UNROLL
      for (uint32_t k = 1; k < 8; ++k)
        j += (uint32_t)(suffix_count[k] >= topk);

      // A depth-3 SEL tree picking suffix_count[j+1] by the bits of j
      // j is the bucket that contains the top-k element
      uint32_t b0 = (j & 1) ? suffix_count[2] : suffix_count[1],
               b1 = (j & 1) ? suffix_count[4] : suffix_count[3];
      uint32_t b2 = (j & 1) ? suffix_count[6] : suffix_count[5],
               b3 = (j & 1) ? suffix_count[8] : suffix_count[7];
      b0 = (j & 2) ? b1 : b0;
      b2 = (j & 2) ? b3 : b2;
      uint32_t suffix_count_j_plus_1 = (j & 4) ? b2 : b0; // = s[j+1]
      smem.reconstruct_pivot_bucket = lane_idx * 8 + j;
      smem.reconstruct_num_should_select = topk - suffix_count_j_plus_1;

      if constexpr (CHECK_IF_SHOULD_SELECT_WHOLE_BUCKET) {
        // topk == suffix_count[j] means the whole pivot bucket is selected
        uint32_t d0 = (j & 1) ? suffix_count[1] : suffix_count[0],
                 d1 = (j & 1) ? suffix_count[3] : suffix_count[2];
        uint32_t d2 = (j & 1) ? suffix_count[5] : suffix_count[4],
                 d3 = (j & 1) ? suffix_count[7] : suffix_count[6];
        d0 = (j & 2) ? d1 : d0;
        d2 = (j & 2) ? d3 : d2;
        uint32_t suffix_count_j = (j & 4) ? d2 : d0; // = suffix_count[j]
        should_select_whole_bucket = (topk == suffix_count_j);
      }
    }

    if constexpr (CHECK_IF_SHOULD_SELECT_WHOLE_BUCKET) {
      // The owning lane holds the result; reduce so all lanes in the warp see the same answer.
      return __reduce_or_sync(0xFFFFFFFFu, should_select_whole_bucket);
    }
    return false;
  }

  // Clean every bucket counter using 16-byte shared-memory writes.
  // Need to be called by all threads.
  static __device__ __forceinline__ void clear_reconstruct_histograms(SharedMemoryPlanBase &smem,
                                                                      uint32_t thread_idx) {
    static_assert((2 * NUM_RECONSTRUCT_BUCKET_SLOTS) % 4 == 0);
    __int128_t zero = (__int128_t)0ull;
    for (uint32_t i = thread_idx; i < (2 * NUM_RECONSTRUCT_BUCKET_SLOTS) / 4; i += NUM_THREADS) {
      reinterpret_cast<__int128_t *>(smem.reconstruct_bucket_counter)[i] = zero;
    }
  }

  template <typename ExtraBarInitF>
  static __device__ __forceinline__ void init_shared_memory(SharedMemoryPlanBase &smem,
                                                            uint32_t warp_idx,
                                                            ExtraBarInitF &&init_extra_bar_func) {
    // Pad unused elements in `surviving_topk_pairs` as `-INF` to avoid being selected
    static_assert((2 * MAX_TOPK) % NUM_THREADS == 0);
    CUTE_UNROLL
    for (uint32_t i = 0; i < MAX_TOPK / NUM_THREADS; ++i) {
      smem.surviving_topk_pairs[0][i * NUM_THREADS + threadIdx.x] = PLACEHOLDER_PAIR;
      smem.surviving_topk_pairs[1][i * NUM_THREADS + threadIdx.x] = PLACEHOLDER_PAIR;
    }

    clear_reconstruct_histograms(smem, threadIdx.x);

    if (warp_idx == 0 && cute::elect_one_sync()) {
      CUTE_UNROLL
      for (uint32_t i = 0; i < NUM_TMA_LOAD_BUFS; ++i) {
        smem.tma_load_full_bar[i].init(NUM_ISSUE_WARPS);
      }
      CUTE_UNROLL
      for (uint32_t i = 0; i < NUM_INIT_ROUNDS_MAX; ++i) {
        smem.init_full_bar[i].init(NUM_ISSUE_WARPS);
      }
      init_extra_bar_func();
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
  }

  static __device__ __forceinline__ void init_shared_memory(SharedMemoryPlanBase &smem,
                                                            uint32_t warp_idx) {
    init_shared_memory(smem, warp_idx, [] {});
  }

#include "base_transport.cuh"

#include "base_output.cuh"

  // Take the appropriate action when we see a NaN (i.e. `trap()` or write 0x3f3f3f3f)
  static __device__ __forceinline__ void take_action_when_have_nan(const TopkSelectArgs &args,
                                                                   uint32_t batch_idx) {
    if (args.abort_when_nan_found) {
      if (threadIdx.x == 0) {
        printf("[topk_select] NaN detected. Calling `trap;` which will result in \"unspecified "
               "launch failure\"\n");
      }
      __syncthreads();
      select_platform::trap();
    } else {
      if (threadIdx.x == 0) {
        *((OutIdxT *)args.output_index + batch_idx * args.stride_output_index_batch) = 0x3F3F3F3F;
      }
    }
  }
};

} // namespace topk_select_common
