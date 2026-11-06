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
prepare_window_scatter(SharedMemoryPlanBase &smem, const TopkSelectArgs &args,
                       uint32_t end_vocab_idx, uint32_t survivor_buf_idx, uint32_t warp_idx,
                       uint32_t lane_idx, uint32_t *window_scatter_delta) {
  uint32_t num_windows = select_platform::ceil_div(end_vocab_idx, (uint32_t)NUM_ELEMS_PER_SEG);
  auto get_window_id = [&](uint32_t offset) -> uint32_t {
    return (uint32_t)smem.surviving_topk_pairs[survivor_buf_idx][offset] / NUM_ELEMS_PER_SEG;
  };

  // Reset all window counters.
  for (uint32_t i = threadIdx.x; i < num_windows; i += NUM_THREADS) {
    window_scatter_delta[i] = 0;
  }
  __syncthreads();

  // Count how many selected candidates fall into each window.
  for (uint32_t i = threadIdx.x; i < args.topk; i += NUM_THREADS) {
    atomicAdd_block(window_scatter_delta + get_window_id(i), 1u);
  }
  __syncthreads();

  // Convert the per-window counts into exclusive prefix sums (starting offsets).
  uint32_t windows_per_thread = select_platform::ceil_div(num_windows, (uint32_t)NUM_THREADS);
  uint32_t w_lo = min(threadIdx.x * windows_per_thread, num_windows);
  uint32_t w_hi = min(w_lo + windows_per_thread, num_windows);
  uint32_t my_cnt = 0;
  for (uint32_t i = w_lo; i < w_hi; ++i) {
    // TODO Performance can be optimized by using wider reads (pay attn to alignment issues!)
    my_cnt += window_scatter_delta[i];
  }
  uint32_t warp_total = __reduce_add_sync(0xFFFFFFFF, my_cnt);
  if (lane_idx == 0) {
    smem.warp_cnt[warp_idx] = warp_total;
  }
  __syncthreads();

  static_assert(NUM_WARPS <= 32);
  uint32_t cur_prefix_sum =
      __reduce_add_sync(0xFFFFFFFF, lane_idx < warp_idx ? smem.warp_cnt[lane_idx] : 0u) +
      warp_level_exclusive_prefix_sum(my_cnt, lane_idx);
  for (uint32_t i = w_lo; i < w_hi; ++i) {
    uint32_t cnt = window_scatter_delta[i];
    window_scatter_delta[i] = cur_prefix_sum;
    cur_prefix_sum += cnt;
  }
  __syncthreads();

  // Subtract window_delta[i] by the index of the first element in this window around all
  // select elements So that later we can use "index + window_delta[i]" to choose the index
  for (uint32_t i = threadIdx.x + 1; i < args.topk;
       i += NUM_THREADS) { // +1 to skip the first element
    uint32_t idx = (uint32_t)smem.surviving_topk_pairs[survivor_buf_idx][i];
    uint32_t window_idx = idx / NUM_ELEMS_PER_SEG;
    uint32_t prev_elem_window_id = get_window_id(i - 1u);
    if (window_idx != prev_elem_window_id) {
      // i is the first element in its window
      window_scatter_delta[window_idx] -= i;
    }
  }

  __syncthreads();
}

// Stage the final top-k candidates from the survivor buffer into a contiguous smem layout
// and run the epilogue.
//
// When SORT_BY_INDEX is true, each 512-element window is visited as a whole in the permuted
// order, so the survivors are already sorted inside each window. Only the window groups need
// to be reordered by their window id
//
// Input cannot contain NaN
template <bool SORT_BY_INDEX>
static __device__ __forceinline__ void
stage_output_and_epilogue(SharedMemoryPlanBase &smem,
                          typename EpilogueT::BlockRadixSortTempStorageT &sort_temp_storage,
                          const TopkSelectArgs &args, uint32_t batch_idx, uint32_t end_vocab_idx,
                          uint32_t survivor_buf_idx, uint32_t warp_idx, uint32_t lane_idx) {

  uint32_t staging_buf_idx = survivor_buf_idx ^ 1;
  bool is_already_sorted_in_index = end_vocab_idx <= NUM_ELEMS_IN_INIT_WINDOW;
  uint32_t *window_scatter_delta = (uint32_t *)smem.incoming_topk_pairs;
  ValueT *smem_value_buf = (ValueT *)smem.surviving_topk_pairs[staging_buf_idx];
  uint32_t *smem_index_buf = (uint32_t *)(smem.surviving_topk_pairs[staging_buf_idx] +
                                          MAX_TOPK * (uint32_t)sizeof(ValueT) / 8);

  // Index-sort shuffle: each window's selected candidates are already sorted and contiguous;
  // only the whole window groups are in visit order, so reorder them by their window id.
  if constexpr (SORT_BY_INDEX)
    if (!is_already_sorted_in_index) {
      prepare_window_scatter(smem, args, end_vocab_idx, survivor_buf_idx, warp_idx, lane_idx,
                             window_scatter_delta);
    }

  CUTE_UNROLL
  for (uint32_t i = threadIdx.x; i < MAX_TOPK; i += NUM_THREADS) {
    uint64_t pair = smem.surviving_topk_pairs[survivor_buf_idx][i];
    uint32_t idx = (uint32_t)pair;
    ValueT val;
    if constexpr (sizeof(ValueT) == 2) {
      val = __ushort_as_bfloat16((uint16_t)(pair >> 32));
    } else {
      val = __uint_as_float((uint32_t)(pair >> 32));
    }
    if (SORT_BY_INDEX && !is_already_sorted_in_index) {
      if (i < args.topk) {
        uint32_t dst_pos = (i + window_scatter_delta[idx / NUM_ELEMS_PER_SEG]) & (MAX_TOPK - 1);
        smem_value_buf[dst_pos] = val;
        smem_index_buf[dst_pos] = idx;
        continue;
      }
    }
    smem_value_buf[i] = val;
    smem_index_buf[i] = idx;
  }

  __syncthreads();

  EpilogueT::template topk_select_epilogue<false>(smem_value_buf, smem_index_buf, args, batch_idx,
                                                  end_vocab_idx, warp_idx, sort_temp_storage);
}
