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

template <class CopySegment>
static __device__ __forceinline__ void
issue_linear_segments(uint32_t end_vocab_idx, uint32_t round_idx, uint32_t warp_idx,
                      select_platform::transac_bar_t &bar, const CopySegment &copy_one_seg) {
  constexpr uint32_t NUM_BYTES_PER_SEG = NUM_ELEMS_PER_SEG * (uint32_t)sizeof(ValueT);
  uint32_t num_global_segs = select_platform::ceil_div(end_vocab_idx, (uint32_t)NUM_ELEMS_PER_SEG);
  uint32_t num_segs_issued = 0;
  CUTE_UNROLL
  for (uint32_t k = 0; k < NUM_SEGS_PER_ISSUE_WARP; k++) {
    uint32_t local_seg_idx = warp_idx + k * NUM_ISSUE_WARPS;
    uint32_t global_seg_idx = round_idx * NUM_SEGS_PER_ROUND + local_seg_idx;
    if (global_seg_idx < num_global_segs) {
      copy_one_seg(local_seg_idx, global_seg_idx);
      num_segs_issued += 1;
    }
  }
  bar.arrive_and_expect_tx(num_segs_issued * NUM_BYTES_PER_SEG);
}

// Issue TMA loads for one round
// Use IS_INIT to control whether is the init round.
// For init round: copy the "tail" [num_perm_segs * NUM_ELEMS_PER_SEG, end_vocab_idx), if
// HAVE_TAIL is true, and also the first NUM_SEGS_PER_ROUND - NUM_TAIL_SEGS segments according to
// the perm generator For non-init round: copy the next NUM_SEGS_PER_ROUND segs according to the
// perm generator
template <bool IS_INIT, bool HAVE_TAIL>
static __device__ __forceinline__ void
issue_tma_loads_for_round(SharedMemoryPlanBase &smem, const CUtensorMap &tensor_map,
                          uint32_t batch_idx, uint32_t end_vocab_idx, uint32_t num_perm_segs,
                          uint32_t round_idx, uint32_t &next_tma_permuted_segment,
                          uint32_t tma_permuted_segment_stride, uint32_t warp_idx) {
  uint32_t perm_len = max(num_perm_segs, 1u);
  uint32_t num_perm_elems = num_perm_segs * NUM_ELEMS_PER_SEG;

  static_assert(IS_INIT || !HAVE_TAIL);
  if (warp_idx >= NUM_ISSUE_WARPS)
    return;

  ValueT *dst_base;
  select_platform::transac_bar_t *bar_ptr;
  if constexpr (IS_INIT) {
    // Use `smem.incoming_topk_pairs` as a buffer for elements in the "init" rounds
    // We've asserted the size of `incoming_topk_pairs` is large enough
    dst_base = (ValueT *)smem.incoming_topk_pairs + round_idx * NUM_ELEMS_PER_ROUND;
    bar_ptr = &smem.init_full_bar[round_idx];
  } else {
    dst_base = smem.tma_load_buf[round_idx % NUM_TMA_LOAD_BUFS];
    bar_ptr = &smem.tma_load_full_bar[round_idx % NUM_TMA_LOAD_BUFS];
  }
  select_platform::transac_bar_t &bar = *bar_ptr;

  constexpr uint32_t NUM_BYTES_PER_SEG = NUM_ELEMS_PER_SEG * (uint32_t)sizeof(ValueT);
  auto copy_one_seg = [&](uint32_t dst_seg_idx, uint32_t global_seg_idx) {
    cute::SM90_TMA_LOAD_3D::copy(&tensor_map, reinterpret_cast<uint64_t *>(&bar),
                                 (uint64_t)cute::TMA::CacheHintSm90::EVICT_FIRST,
                                 dst_base + dst_seg_idx * NUM_ELEMS_PER_SEG, 0,
                                 global_seg_idx * NUM_TMA_ROWS_PER_SEG, batch_idx);
  };

  if (IS_INIT && num_perm_segs == 0) {
    // Do not permute if `num_perm_seg == 0`, just copy in all the elements in-a-row
    issue_linear_segments(end_vocab_idx, round_idx, warp_idx, bar, copy_one_seg);
  } else {
    static_assert(NUM_TAIL_SEGS % NUM_ISSUE_WARPS == 0);
    constexpr uint32_t NUM_TAIL_SEGS_PER_ISSUE_WARP =
        HAVE_TAIL ? NUM_TAIL_SEGS / NUM_ISSUE_WARPS : 0;
    if constexpr (HAVE_TAIL) {
      // Copy tail segs
      // To save memory BW, only copy segs that have at least one valid element. The other segs
      // are handled by `fill_padded_tail_segments`
      uint32_t num_tail_tma_segs =
          select_platform::ceil_div(end_vocab_idx - num_perm_elems, (uint32_t)NUM_ELEMS_PER_SEG);
      uint32_t num_real_tail_segs_issued = 0;
      CUTE_UNROLL
      for (uint32_t i = 0; i < NUM_TAIL_SEGS_PER_ISSUE_WARP; i++) {
        uint32_t cur_seg_idx = warp_idx + i * NUM_ISSUE_WARPS;
        if (cur_seg_idx < num_tail_tma_segs) {
          copy_one_seg(cur_seg_idx, (num_perm_segs + cur_seg_idx));
          num_real_tail_segs_issued += 1;
        }
      }
      bar.arrive_and_expect_tx(
          (num_real_tail_segs_issued + NUM_SEGS_PER_ISSUE_WARP - NUM_TAIL_SEGS_PER_ISSUE_WARP) *
          NUM_BYTES_PER_SEG);
    } else {
      bar.arrive_and_expect_tx(NUM_SEGS_PER_ISSUE_WARP * NUM_BYTES_PER_SEG);
    }
    CUTE_UNROLL
    for (uint32_t i = NUM_TAIL_SEGS_PER_ISSUE_WARP; i < NUM_SEGS_PER_ISSUE_WARP; i++) {
      uint32_t cur_global_seg_idx = next_tma_permuted_segment;
      uint32_t local_seg_idx = warp_idx + i * NUM_ISSUE_WARPS;
      copy_one_seg(local_seg_idx, cur_global_seg_idx);
      advance_perm_state(next_tma_permuted_segment, tma_permuted_segment_stride, perm_len);
    }
  }
}

// Fill the invalid (padded) tail segments with PLACEHOLDER (-INF).
// TMA only loads the real tail elements; the unused padded tail region must still be initialized
// as -INF
static __device__ __forceinline__ void fill_padded_tail_segments(ValueT *init_buf,
                                                                 uint32_t num_tail_elems) {
  uint32_t tail_covered_by_tma = select_platform::ceil(num_tail_elems, (uint32_t)NUM_ELEMS_PER_SEG);
  constexpr __int128_t FILLER_128b = ((__int128_t)PLACEHOLDER_B64 << 64) | PLACEHOLDER_B64;
  for (uint32_t i = tail_covered_by_tma + threadIdx.x * NUM_ELEMS_PER_128b; i < NUM_TAIL_ELEMS;
       i += NUM_THREADS * NUM_ELEMS_PER_128b) {
    select_platform::st_shared(init_buf + i, FILLER_128b);
  }
}
