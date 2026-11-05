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
#include "base.cuh"

namespace topk_select_common {
template <typename Config> class TopkSelectKernelBF16Base : public TopkSelectKernelBase<Config> {
  using Base = TopkSelectKernelBase<Config>;

public:
  using ValueT = typename Base::ValueT;
  using OutIdxT = typename Base::OutIdxT;
  using TmaParams = typename Base::TmaParams;
  using SharedMemoryPlanBase = typename Base::SharedMemoryPlanBase;
  using EpilogueT = typename Base::EpilogueT;
  using Base::ELEMS_PER_THREAD_PER_ROUND_MASK;
  using Base::MAX_TOPK;
  using Base::NUM_128b_INIT_PER_THREAD;
  using Base::NUM_128b_PER_THREAD_PER_ROUND;
  using Base::NUM_ELEMS_IN_INIT_WINDOW;
  using Base::NUM_ELEMS_PER_128b;
  using Base::NUM_ELEMS_PER_ROUND;
  using Base::NUM_ELEMS_PER_SEG;
  using Base::NUM_ELEMS_PER_THREAD_PER_ROUND;
  using Base::NUM_EXTRA_SLOTS;
  using Base::NUM_INIT_ROUNDS_MAX;
  using Base::NUM_ISSUE_WARPS;
  using Base::NUM_RECONSTRUCT_BUCKETS;
  using Base::NUM_RECONSTRUCT_UNITS_MAX;
  using Base::NUM_SEGS_IN_INIT_WINDOW;
  using Base::NUM_SEGS_PER_ISSUE_WARP;
  using Base::NUM_SEGS_PER_ROUND;
  using Base::NUM_TAIL_ELEMS;
  using Base::NUM_TAIL_SEGS;
  using Base::NUM_THREADS;
  using Base::NUM_TMA_LOAD_BUFS;
  using Base::NUM_UINT32_IN_INIT_WINDOW_PER_THREAD;
  using Base::NUM_UINT32_PER_128b;
  using Base::NUM_WARPS;
  using Base::PERM_ADD_BASE;
  using Base::PERM_MUL_PRIME;
  using Base::PLACEHOLDER_PAIR;
  using Base::RECONSTRUCT_THRESHOLD;
  using Base::TMA_PREFETCH_DEPTH;

  static_assert(std::is_same_v<ValueT, nv_bfloat16>);
  static_assert(!Config::sorted_value);
  static constexpr uint32_t NUM_UINT32_RECONSTRUCT_PER_THREAD =
      ((NUM_RECONSTRUCT_UNITS_MAX + NUM_THREADS - 1) / NUM_THREADS) | 1u; // padding
  static_assert(2 * NUM_UINT32_RECONSTRUCT_PER_THREAD <= 256);
  static constexpr uint16_t PLACEHOLDER = 0xff80; // -INF
  static constexpr uint32_t NEG_INF_X2_BITS = 0xFF80FF80u;

  // Histogram pass 1 (MSB radix): for each distorted bf16x2 value, retrieve the upper 8 bits and
  // performs atomicAdd on the bucket counter `num_packed_values` should be guaranteed to be aligned
  // by `NUM_PACKED_VALUES_ALIGNMENT`
  template <uint32_t N, uint32_t NUM_PACKED_VALUES_ALIGNMENT = 1>
  static __device__ __forceinline__ void histogram_radix_msb(uint32_t *bucket_counter,
                                                             const nv_bfloat162 (&packed_values)[N],
                                                             uint32_t num_packed_values) {
    uint32_t bucket_base = cute::cast_smem_ptr_to_uint(bucket_counter);
    CUTE_UNROLL
    for (uint32_t i = 0; i < N; i++) {
      if (i % NUM_PACKED_VALUES_ALIGNMENT == 0 && i == num_packed_values)
        break;
      uint32_t raw = bf16x2_to_u32(packed_values[i]);
      uint32_t distorted;
      distort_x2<uint16_t>((uint16_t *)&distorted, (const uint16_t *)&raw);
      // bfe + mad form each bucket address in two instructions, then two shared-memory adds.
      asm volatile("{\n"
                   ".reg .b32 i0, i1, a0, a1;\n"
                   "bfe.u32 i0, %0, 8, 8;\n"
                   "bfe.u32 i1, %0, 24, 8;\n"
                   "mad.lo.u32 a0, i0, 4, %1;\n"
                   "mad.lo.u32 a1, i1, 4, %1;\n"
                   "red.shared.add.u32 [a0], 1;\n"
                   "red.shared.add.u32 [a1], 1;\n"
                   "}\n"
                   :
                   : "r"(distorted), "r"(bucket_base)
                   : "memory");
    }
  }

  // Histogram pass 2 (LSB radix refinement): among values whose distorted MSB equals pivot_hi8,
  // retrieve the lower 8 bits and performs atomicAdd on the bucket counter `num_packed_values`
  // should be guaranteed to be aligned by `NUM_PACKED_VALUES_ALIGNMENT` `pivot_hi8` is the highest
  // 8 bit of the DISTORTED pivot
  template <bool USE_CLUSTER_ADDRESSING, uint32_t N, uint32_t NUM_PACKED_VALUES_ALIGNMENT = 1>
  static __device__ __forceinline__ void
  histogram_radix_lsb_for_pivot_msb(uint32_t *bucket_counter, uint32_t pivot_hi8,
                                    const nv_bfloat162 (&packed_values)[N],
                                    uint32_t num_packed_values) {
    uint32_t bucket1_base = cute::cast_smem_ptr_to_uint(bucket_counter);
    // Every elements in the pivot bucket must have the same sign, so we can use this information to
    // optimize instead of
    bool bucket_negative = pivot_hi8 < 0x80;
    uint32_t orig_pivot_hi8 = bucket_negative ? 0xFF - pivot_hi8 : pivot_hi8 - 0x80; // Un-distort
    uint32_t raw_hi8_x2 = (orig_pivot_hi8 << 16) | orig_pivot_hi8;
    uint32_t low_byte_xor = bucket_negative ? 0xFFFFFFFFu : 0u;
    // Elements outside the bucket are counted in this row's sink slot (index
    // NUM_RECONSTRUCT_BUCKETS) and never read.
    uint32_t scratch_bucket_idx_x2 = 0x01000100; // 2x 256
    CUTE_UNROLL
    for (uint32_t i = 0; i < N; i++) {
      if (i % NUM_PACKED_VALUES_ALIGNMENT == 0 && i == num_packed_values)
        break;
      asm volatile("{\n"
                   ".reg .b32 hi, lo, sel, t, t0, t1, a0, a1;\n"
                   "prmt.b32 hi, %0, 0, 0x5341;\n" // bytes {1, zero, 3, zero}: both raw high bytes
                                                   // as 0x00hh halves
                   "set.eq.s32.bf16x2 sel, hi, %1;\n" // 0xFFFF per half whose raw high byte equals
                                                      // the pivot's
                   "lop3.b32 lo, %0, %4, 0x00ff00ff, 0x28;\n" // lo = (%0 ^ %4) & 0x00ff00ff
                   "lop3.b32 t, sel, lo, %3, 0xca;\n"         // t = sel ? lo : %3, per-bit
                   "and.b32 t0, t, 0xFFFF;\n"
                   "shr.u32 t1, t, 16;\n"
                   "mad.lo.u32 a0, t0, 4, %2;\n" // base + bucket * 4
                   "mad.lo.u32 a1, t1, 4, %2;\n"
                   "red.shared.add.u32 [a0], 1;\n"
                   "red.shared.add.u32 [a1], 1;\n"
                   "}\n"
                   :
                   : "r"(bf16x2_to_u32(packed_values[i])), "r"(raw_hi8_x2), "r"(bucket1_base),
                     "r"(scratch_bucket_idx_x2), "r"(low_byte_xor)
                   : "memory");
    }
  }

  // Count the number of elements that are 1) > pivot 2) = pivot 3) NaN in the given array
  struct CensusCounts {
    uint32_t cnt_gt;
    uint32_t cnt_eq;
    uint32_t cnt_nan;
  };

  // `num_packed_values` should be guaranteed to be aligned by `NUM_PACKED_VALUES_ALIGNMENT`
  template <uint32_t N, uint32_t NUM_PACKED_VALUES_ALIGNMENT = 1>
  static __device__ __forceinline__ CensusCounts get_census_counts(const nv_bfloat162 (&values)[N],
                                                                   uint32_t num_packed_values,
                                                                   uint32_t pivot_value_x2_bits) {
    static_assert(2 * N <= 256); // Since we're going to use bf16 for accumulation
    nv_bfloat162 gt_accum = {0.0f, 0.0f};
    nv_bfloat162 eq_accum = {0.0f, 0.0f};
    nv_bfloat162 nan_accum = {0.0f, 0.0f};
    CUTE_UNROLL
    for (uint32_t i = 0; i < N; i++) {
      if (i % NUM_PACKED_VALUES_ALIGNMENT == 0 && i == num_packed_values)
        break;
      uint32_t raw = bf16x2_to_u32(values[i]);
      asm volatile("{\n"
                   ".reg .b32 gt_result, eq_result;\n"
                   "set.gt.bf16x2.bf16x2 gt_result, %3, %4;\n"
                   "add.rn.bf16x2 %0, gt_result, %0;\n"
                   "set.eq.bf16x2.bf16x2 eq_result, %3, %4;\n"
                   "add.rn.bf16x2 %1, eq_result, %1;\n"
                   "min.NaN.bf16x2 %2, %2, %3;\n" // We use `min.NaN` for NaN detection
                   "}\n"
                   : "+r"(*(uint32_t *)&gt_accum), "+r"(*(uint32_t *)&eq_accum),
                     "+r"(*(uint32_t *)&nan_accum)
                   : "r"(raw), "r"(pivot_value_x2_bits));
    }
    uint32_t cnt_gt = (uint32_t)(float)(gt_accum.x + gt_accum.y);
    uint32_t cnt_eq = (uint32_t)(float)(eq_accum.x + eq_accum.y);
    uint32_t nan_flag;
    asm("set.nan.bf16x2.bf16x2 %0, %1, %1;" : "=r"(nan_flag) : "r"(*(uint32_t *)&nan_accum));
    uint32_t cnt_nan = nan_flag != 0 ? 1u : 0u;
    return {cnt_gt, cnt_eq, cnt_nan};
  }

  // Decide whether or not to accept an element in the old survivor buffer
  // If taken, update relevant status (`dst_ptr` and `eq_quota`)
  template <bool USE_CLUSTER_ADDRESSING>
  static __device__ __forceinline__ void copy_selected_pairs_to_survivor(
      uint32_t &dst_ptr,  // The pointer to the new survivor buffer. May increase. Marked as uint32
                          // since it's obtained from cast_smem_ptr_to_uint
      uint32_t &eq_quota, // The EQ quota of this thread. May decrease
      nv_bfloat162 packed_values, uint32_t pivot_value_x2_bits,
      uint32_t
          src_ptr // The pointer to the location of src. `src` contains 8B pairs in {index, value}
  ) {
    uint32_t values_raw = bf16x2_to_u32(packed_values);
#define MOVE_SELECTED_PAIR2(ADD)                                                                   \
  asm volatile("{\n"                                                                               \
               ".reg .pred g0, g1, e0, e1, q, t0, t1, s0, s1;\n"                                   \
               ".reg .b32 pair0_lo, pair0_hi, pair1_lo, pair1_hi;\n"                               \
               "ld.shared.v4.b32 {pair0_lo, pair0_hi, pair1_lo, pair1_hi}, [%4];\n"                \
               "setp.gt.bf16x2 g0|g1, %2, %3;\n"                                                   \
               "setp.eq.bf16x2 e0|e1, %2, %3;\n" /* element 0 (low half) */                        \
               "setp.ne.and.u32 t0, %1, 0, e0;\n"                                                  \
               "@t0 sub.u32 %1, %1, 1;\n"                                                          \
               "or.pred s0, g0, t0;\n"                                                             \
               "@s0 st.shared.v2.b32 [%0], {pair0_lo, pair0_hi};\n"                                \
               "@s0 " ADD                                                                          \
               "\n" /* element 1 (high half; quota is re-tested after element 0's decrement) */    \
               "setp.ne.and.u32 t1, %1, 0, e1;\n"                                                  \
               "@t1 sub.u32 %1, %1, 1;\n"                                                          \
               "or.pred s1, g1, t1;\n"                                                             \
               "@s1 st.shared.v2.b32 [%0], {pair1_lo, pair1_hi};\n"                                \
               "@s1 " ADD "\n"                                                                     \
               "}\n"                                                                               \
               : "+r"(dst_ptr), "+r"(eq_quota)                                                     \
               : "r"(values_raw), "r"(pivot_value_x2_bits), "r"(src_ptr)                           \
               : "memory")
    if constexpr (USE_CLUSTER_ADDRESSING) {
      MOVE_SELECTED_PAIR2("add.u32 %0, %0, 8;");
    } else {
      MOVE_SELECTED_PAIR2("add.f32 %0, %0, 0f00000008;");
    }
#undef MOVE_SELECTED_PAIR2
  }

  // The src-in-register version of `copy_selected_pairs_to_survivor`
  template <bool USE_CLUSTER_ADDRESSING, bool CHECK_PADDING = false>
  static __device__ __forceinline__ void append_selected_pairs_from_registers(
      uint32_t &dst_ptr, uint32_t &eq_quota, nv_bfloat162 packed_values,
      uint32_t pivot_value_x2_bits, uint32_t index0, uint32_t index1, uint32_t end_vocab_idx = 0) {
    if constexpr (CHECK_PADDING) {
      if (index0 >= end_vocab_idx)
        return;
      if (index1 >= end_vocab_idx) {
        float value = __bfloat162float(packed_values.x);
        float pivot = __bfloat162float(u32_to_bf16x2(pivot_value_x2_bits).x);
        if (value > pivot || (value == pivot && eq_quota != 0)) {
          if (value == pivot)
            --eq_quota;
          asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::"r"(dst_ptr), "r"(index0),
                       "r"(uint32_t(__bfloat16_as_ushort(packed_values.x)))
                       : "memory");
          dst_ptr += sizeof(uint64_t);
        }
        return;
      }
    }
    uint32_t values_raw = bf16x2_to_u32(packed_values);
    uint32_t val_word1 = values_raw >> 16; // element 1 low, clean high
#define APPEND_SELECTED_PAIR2(ADD)                                                                 \
  asm volatile(                                                                                    \
      "{\n"                                                                                        \
      ".reg .pred g0, g1, e0, e1, q, t0, t1, s0, s1;\n"                                            \
      "setp.gt.bf16x2 g0|g1, %2, %3;\n"                                                            \
      "setp.eq.bf16x2 e0|e1, %2, %3;\n" /* element 0 (low half) */                                 \
      "setp.ne.and.u32 t0, %1, 0, e0;\n"                                                           \
      "@t0 sub.u32 %1, %1, 1;\n"                                                                   \
      "or.pred s0, g0, t0;\n"                                                                      \
      "@s0 st.shared.v2.u32 [%0], {%4, %2};\n"                                                     \
      "@s0 " ADD "\n" /* element 1 (high half; quota is re-tested after element 0's decrement) */  \
      "setp.ne.and.u32 t1, %1, 0, e1;\n"                                                           \
      "@t1 sub.u32 %1, %1, 1;\n"                                                                   \
      "or.pred s1, g1, t1;\n"                                                                      \
      "@s1 st.shared.v2.u32 [%0], {%5, %6};\n"                                                     \
      "@s1 " ADD "\n"                                                                              \
      "}\n"                                                                                        \
      : "+r"(dst_ptr), "+r"(eq_quota)                                                              \
      : "r"(values_raw), "r"(pivot_value_x2_bits), "r"(index0), "r"(index1), "r"(val_word1)        \
      : "memory")
    if constexpr (USE_CLUSTER_ADDRESSING) {
      APPEND_SELECTED_PAIR2("add.u32 %0, %0, 8;");
    } else {
      APPEND_SELECTED_PAIR2("add.f32 %0, %0, 0f00000008;");
    }
#undef APPEND_SELECTED_PAIR2
  }

  // Get the pivot via a two-stage histogram and get PivotAndQuota
  // The MSB histogram should be ready and all threads should be able to see it (e.g. via a
  // `__syncthreads()`) before we call the function We don't absorb MSB histogram's logic inside
  // since sometimes we'd like to overlap TMA load and MSB histogram's modification
  struct PivotAndQuota {
    uint32_t pivot_value_x2_bits;
    uint32_t start_pos_in_collector;
    uint32_t eq_quota;
    uint32_t cnt_nan;
  };

  // `num_my` should be guaranteed to be aligned by `NUM_PACKED_VALUES_ALIGNMENT`
  template <bool USE_CLUSTER_ADDRESSING, uint32_t N, uint32_t NUM_PACKED_VALUES_ALIGNMENT = 1>
  static __device__ __forceinline__ PivotAndQuota compute_pivot_and_quota(
      uint32_t topk, const nv_bfloat162 (&values)[N], uint32_t num_values, uint32_t warp_idx,
      uint32_t lane_idx, SharedMemoryPlanBase &smem, uint32_t padding_elems = 0) {
    if (warp_idx == 0) {
      Base::template find_pivot_in_histogram<false>(smem, smem.reconstruct_bucket_counter[0], topk,
                                                    lane_idx);
    }
    __syncthreads();
    uint32_t pivot_hi8 = smem.reconstruct_pivot_bucket;
    uint32_t num_elem_should_select_in_pivot_bucket = smem.reconstruct_num_should_select;

    histogram_radix_lsb_for_pivot_msb<USE_CLUSTER_ADDRESSING, N, NUM_PACKED_VALUES_ALIGNMENT>(
        smem.reconstruct_bucket_counter[1], pivot_hi8, values, num_values);
    __syncthreads();

    if (warp_idx == 0) {
      Base::template find_pivot_in_histogram<false>(smem, smem.reconstruct_bucket_counter[1],
                                                    num_elem_should_select_in_pivot_bucket,
                                                    lane_idx);
    }
    __syncthreads();
    uint32_t pivot_lo8 = smem.reconstruct_pivot_bucket;
    uint32_t pivot_distorted = (pivot_hi8 << 8) | pivot_lo8;
    uint16_t pivot_value = un_distort((uint16_t)pivot_distorted);
    uint32_t pivot_value_x2_bits = ((uint32_t)pivot_value << 16) | pivot_value;

    auto census =
        get_census_counts<N, NUM_PACKED_VALUES_ALIGNMENT>(values, num_values, pivot_value_x2_bits);
    // Padding is -INF too, but it must not consume the equality quota.
    if (pivot_value == PLACEHOLDER)
      census.cnt_eq -= padding_elems;

    static_assert(NUM_WARPS <= NUM_RECONSTRUCT_BUCKETS);
    auto eqgt = Base::compute_equal_quota_and_prefix(census.cnt_gt, census.cnt_eq, topk, warp_idx,
                                                     lane_idx, smem.reconstruct_bucket_counter[0]);
    return {pivot_value_x2_bits, eqgt.start_pos_in_collector, eqgt.eq_quota, census.cnt_nan};
  }

#include "scan.cuh"
};
} // namespace topk_select_common
