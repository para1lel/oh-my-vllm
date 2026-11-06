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

static __device__ __forceinline__ void
find_scan_hits(SharedMemoryPlanBase &smem, const ValueT *buf, uint32_t &hit_mask,
               uint32_t smem_read_offset, uint32_t chunk_swizzle_mask, uint32_t threshold_x2_bits,
               uint32_t tma_buf_idx, uint32_t tma_buf_phase, bool is_warp_active) {
  if (is_warp_active) {
    smem.tma_load_full_bar[tma_buf_idx].wait(tma_buf_phase);
    ValueT values[NUM_ELEMS_PER_THREAD_PER_ROUND];
    Base::template load_swizzled_slice<NUM_128b_PER_THREAD_PER_ROUND>(values, buf, smem_read_offset,
                                                                      chunk_swizzle_mask);
    CUTE_UNROLL
    for (int32_t j = NUM_ELEMS_PER_THREAD_PER_ROUND - 4; j >= 0; j -= 4) {
      asm volatile(
          "{\n"
          ".reg .b32 r0, r1, ind, nib;\n"
          "set.gtu.s32.bf16x2 r0, %1, %3;\n" // .gtu: NaN always counts as a hit, so every NaN
                                             // reaches the incoming buffer and the census below
                                             // reports it.
          "set.gtu.s32.bf16x2 r1, %2, %3;\n" // `set` with .s32 fills the corresponding 16bit to
                                             // 0xFFFF (-1) when the condition holds. We use
                                             // `prmt` to extract one bit from each
          "prmt.b32 ind, r0, r1, 0x7531;\n"
          "dp4a.s32.s32 nib, ind, 0xF8FCFEFF, 0;\n" // ind's bytes are 0x00/0xFF, so with signed
                                                    // byte weights each hit contributes 1/2/4/8
          "mad.lo.u32 %0, %0, 16, nib;\n"
          "}\n"
          : "+r"(hit_mask)
          : "r"(*(const uint32_t *)(values + j)), "r"(*(const uint32_t *)(values + j + 2)),
            "r"(threshold_x2_bits));
    }
  }
}

static __device__ __forceinline__ void
append_scan_hits(SharedMemoryPlanBase &smem, uint32_t hit_mask, uint32_t dst_slot,
                 uint32_t seg_elem_base, uint32_t first_hit_e, uint32_t first_hit_val,
                 uint32_t elem_addr_base, uint32_t chunk_swizzle_mask, bool is_warp_active,
                 uint32_t warp_total_hits) {
  if (is_warp_active && warp_total_hits != 0) {
    uint32_t dst_ptr = cute::cast_smem_ptr_to_uint(smem.incoming_topk_pairs) +
                       dst_slot * (uint32_t)sizeof(uint64_t);
    if (hit_mask != 0) {
      asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::"r"(dst_ptr),
                   "r"(seg_elem_base + first_hit_e), "r"(first_hit_val)
                   : "memory");
      dst_ptr += (uint32_t)sizeof(uint64_t);
    }
    uint32_t mask = hit_mask & (hit_mask - 1u);
    while (mask != 0) {
      uint32_t e = __ffs(mask) - 1u;
      mask &= mask - 1u;
      uint32_t val_word;
      asm volatile("ld.shared.u16 %0, [%1];"
                   : "=r"(val_word)
                   : "r"(elem_addr_base + ((e ^ chunk_swizzle_mask) << 1)));
      asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::"r"(dst_ptr), "r"(seg_elem_base + e),
                   "r"(val_word)
                   : "memory");
      dst_ptr += (uint32_t)sizeof(uint64_t);
    }
  }
}

static __device__ __forceinline__ uint32_t collect_scan_round(
    SharedMemoryPlanBase &smem, bool is_warp_active, uint32_t tma_buf_idx, uint32_t tma_buf_phase,
    uint32_t threshold_x2_bits, uint32_t smem_read_offset, uint32_t chunk_swizzle_mask,
    uint32_t current_permuted_segment, uint32_t offset_in_segment, uint32_t num_incomers,
    uint32_t warp_idx, uint32_t lane_idx) {
  const ValueT *buf = smem.tma_load_buf[tma_buf_idx];
  // One bit per element of this thread's slice, in element order.
  // Hits are rare, so appending through the set bits of this mask is faster than testing +
  // storing every element again.
  static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND <= 32); // hit_mask is a uint32_t
  static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND % 4 ==
                0); // the loop below packs 4 elements per nibble
  uint32_t hit_mask = 0;
  find_scan_hits(smem, buf, hit_mask, smem_read_offset, chunk_swizzle_mask, threshold_x2_bits,
                 tma_buf_idx, tma_buf_phase, is_warp_active);
  uint32_t num_new_incomers = __popc(hit_mask);

  uint32_t warp_total_hits = __reduce_add_sync(0xFFFFFFFF, num_new_incomers);
  if (lane_idx == 0) {
    smem.warp_cnt[warp_idx] = warp_total_hits;
  }
  __syncthreads();

  static_assert(NUM_WARPS <= 32);
  uint32_t stored_warp_hits = lane_idx < NUM_WARPS ? smem.warp_cnt[lane_idx] : 0u;
  uint32_t num_total_hits_in_this_round = __reduce_add_sync(0xFFFFFFFF, stored_warp_hits);

  uint32_t seg_elem_base = current_permuted_segment * NUM_ELEMS_PER_SEG + offset_in_segment;

  // Element e of this thread's slice lives at smem_read_offset + (e ^ chunk_swizzle_mask)
  uint32_t elem_addr_base = cute::cast_smem_ptr_to_uint(buf + smem_read_offset);

  // Start the first hit's smem load before the prefix scan / count exchange below, so that its
  // latency (and the barrier wait) overlaps with them instead of delaying the first store.
  uint32_t first_hit_e = hit_mask != 0 ? __ffs(hit_mask) - 1u : 0u;
  uint32_t first_hit_val = 0;
  if (is_warp_active && warp_total_hits != 0) {
    asm volatile("ld.shared.u16 %0, [%1];"
                 : "=r"(first_hit_val)
                 : "r"(elem_addr_base + ((first_hit_e ^ chunk_swizzle_mask) << 1)));
  }

  // Lane prefix via one ballot per bit of the (<= 16) hit count: the ballots are independent,
  // so this is much shorter on the critical path than a 5-step shuffle scan.
  static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND < 32); // one ballot per bit of num_new_incomers
  uint32_t lane_prefix = 0;
  CUTE_UNROLL
  for (uint32_t k = 0; k < 5; ++k) {
    uint32_t bit =
        __ballot_sync(0xFFFFFFFF, (num_new_incomers >> k) & 1u) & ((1u << lane_idx) - 1u);
    lane_prefix += (uint32_t)__popc(bit) << k;
  }

  uint32_t dst_slot = num_incomers +
                      __reduce_add_sync(0xFFFFFFFF, lane_idx < warp_idx ? stored_warp_hits : 0u) +
                      lane_prefix;

  append_scan_hits(smem, hit_mask, dst_slot, seg_elem_base, first_hit_e, first_hit_val,
                   elem_addr_base, chunk_swizzle_mask, is_warp_active, warp_total_hits);
  return num_total_hits_in_this_round;
}

template <class IsWarpActiveF, class IssueMain, class Reconstruct>
static __device__ __forceinline__ void
scan_remaining_rounds(SharedMemoryPlanBase &smem, uint32_t num_main_rounds,
                      uint32_t &threshold_x2_bits, uint32_t &num_incomers,
                      uint32_t &current_permuted_segment,
                      uint32_t permuted_segment_stride_per_round, uint32_t perm_len,
                      uint32_t warp_idx, uint32_t lane_idx, IsWarpActiveF &is_warp_active_f,
                      IssueMain &issue_tma_copy_for_main_rounds, Reconstruct &reconstruct) {
  uint32_t logical_elem_offset = threadIdx.x * NUM_ELEMS_PER_THREAD_PER_ROUND;
  uint32_t offset_in_segment = logical_elem_offset % NUM_ELEMS_PER_SEG;

  uint32_t swizzle_mask = Base::sw_msk(logical_elem_offset);
  uint32_t chunk_swizzle_mask = swizzle_mask & ELEMS_PER_THREAD_PER_ROUND_MASK;
  uint32_t smem_read_offset =
      logical_elem_offset ^ (swizzle_mask & ~ELEMS_PER_THREAD_PER_ROUND_MASK);

  // Track the TMA buffer index and its phase incrementally instead of dividing every round.
  // On long rows the init window is a tiny fraction of the row, so its pivot stays far above the
  // row's true one and the append rate (hence the store path) stays high for the whole scan:
  // reconstruct early to tighten the pivot. Short rows would pay more for the reconstruct than the
  // extra appends cost, so they keep the plain threshold.
  uint32_t reconstruct_trigger =
      num_main_rounds > 16 ? RECONSTRUCT_THRESHOLD / 4 : RECONSTRUCT_THRESHOLD;

  uint32_t tma_buf_idx = 0;
  uint32_t tma_buf_phase = 0;

  for (uint32_t main_round_idx = 0; main_round_idx < num_main_rounds; ++main_round_idx) {
    if (main_round_idx + TMA_PREFETCH_DEPTH < num_main_rounds && cute::elect_one_sync()) {
      issue_tma_copy_for_main_rounds(main_round_idx + TMA_PREFETCH_DEPTH);
    }

    uint32_t num_total_hits_in_this_round = collect_scan_round(
        smem, is_warp_active_f(main_round_idx), tma_buf_idx, tma_buf_phase, threshold_x2_bits,
        smem_read_offset, chunk_swizzle_mask, current_permuted_segment, offset_in_segment,
        num_incomers, warp_idx, lane_idx);
    Base::advance_perm_state(current_permuted_segment, permuted_segment_stride_per_round, perm_len);

    num_incomers += num_total_hits_in_this_round;
    // No warp may run ahead into the next round: the TMA copy issued there reuses the smem
    // buffer this round is reading, and warp_cnt must not be overwritten while others read it.
    __syncthreads();

    if (++tma_buf_idx == NUM_TMA_LOAD_BUFS) {
      tma_buf_idx = 0;
      tma_buf_phase ^= 1u;
    }

    // On long rows the init window is a tiny fraction of the row, so its pivot stays far above the
    // row's true one and the append rate (hence the store path) stays high for the whole scan:
    // reconstruct early to tighten the pivot. Short rows would pay more for the reconstruct than
    // the extra appends cost, so they keep the plain threshold.
    if (num_incomers >= reconstruct_trigger && main_round_idx + 1 < num_main_rounds) {
      threshold_x2_bits = reconstruct(num_incomers);
      num_incomers = 0;
    }
  }

  if (num_incomers > 0) {
    reconstruct(num_incomers);
    num_incomers = 0;
  }
}
