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

template <class IssueCopy, class IssueMain>
static __device__ __forceinline__ void
prefetch_initial_scan(SharedMemoryPlanBase &smem, uint32_t num_local_tail_elems_padded,
                      uint32_t num_perm_segs, uint32_t num_local_tail_elems,
                      uint32_t num_init_rounds, uint32_t num_main_rounds, IssueCopy &issue_tma_copy,
                      IssueMain &issue_tma_copy_for_main_rounds) {
  // Issue copy: Copy the tail and the first few segments, result in init_buf[0:
  // NUM_ELEMS_IN_INIT_WINDOW] (or init_buf[0: ])
  ValueT *init_buf = (ValueT *)smem.incoming_topk_pairs;
  if (cute::elect_one_sync()) {
    if (num_local_tail_elems_padded != 0) {
      issue_tma_copy.template operator()<true, true>(0);
    } else {
      issue_tma_copy.template operator()<true, false>(0);
    }

    CUTE_UNROLL
    for (uint32_t i = 1; i < NUM_INIT_ROUNDS_MAX; i++) {
      if (i < num_init_rounds) {
        issue_tma_copy.template operator()<true, false>(i);
      }
    }

    // Pre-issue TMA copies for the main part
    CUTE_UNROLL
    for (uint32_t i = 0; i < TMA_PREFETCH_DEPTH; i++) {
      if (i < num_main_rounds) {
        issue_tma_copy_for_main_rounds(i);
      }
    }
  }

  if (num_perm_segs != 0 && num_local_tail_elems_padded != 0) {
    Base::fill_padded_tail_segments(init_buf, num_local_tail_elems);
    __syncthreads();
  }
}

static __device__ __forceinline__ void
histogram_initial_scan(SharedMemoryPlanBase &smem, ValueT *init_buf, uint32_t num_init_rounds,
                       uint32_t num_local_tail_elems, uint32_t num_128b_in_init_window_padded) {
  CUTE_UNROLL
  for (uint32_t i = 0; i < NUM_INIT_ROUNDS_MAX; i++) {
    if (i == num_init_rounds)
      break;
    smem.init_full_bar[i].wait(0);

    // Fill the incomplete segment with PLACEHOLDER (-INF)
    if (num_local_tail_elems % NUM_ELEMS_PER_SEG != 0 &&
        i == num_local_tail_elems / NUM_ELEMS_PER_ROUND) {
      uint32_t box_end =
          select_platform::ceil_div(num_local_tail_elems, (uint32_t)NUM_ELEMS_PER_SEG) *
          NUM_ELEMS_PER_SEG;
      for (uint32_t e = num_local_tail_elems + threadIdx.x; e < box_end; e += NUM_THREADS) {
        init_buf[Base::sw_elem(e)] = __ushort_as_bfloat16(PLACEHOLDER);
      }
      __syncthreads();
    }
    nv_bfloat162 my_values[4 * NUM_128b_PER_THREAD_PER_ROUND];
    uint32_t num_my_values = 0;
    CUTE_UNROLL
    for (uint32_t k = 0; k < NUM_128b_PER_THREAD_PER_ROUND; ++k) {
      uint32_t pos = threadIdx.x + (i * NUM_128b_PER_THREAD_PER_ROUND + k) * NUM_THREADS;
      if (pos < num_128b_in_init_window_padded) {
        ld_shared<4>(my_values + k * 4, reinterpret_cast<const nv_bfloat162 *>(
                                            init_buf + Base::sw_b128(pos) * NUM_ELEMS_PER_128b));
        num_my_values = 4 * (k + 1);
      }
    }
    histogram_radix_msb<4 * NUM_128b_PER_THREAD_PER_ROUND, 4>(smem.reconstruct_bucket_counter[0],
                                                              my_values, num_my_values);
  }
}

template <bool USE_CLUSTER_ADDRESSING>
static __device__ __forceinline__ void
append_initial_scan(SharedMemoryPlanBase &smem,
                    const nv_bfloat162 (&init_values)[NUM_UINT32_IN_INIT_WINDOW_PER_THREAD],
                    uint32_t start_pos_in_collector, uint32_t &eq_quota,
                    uint32_t pivot_value_x2_bits, uint32_t local_start_seg_idx,
                    uint32_t my_elem_start_idx, uint32_t num_local_tail_elems_padded,
                    uint32_t perm_len, uint32_t perm_mul, uint32_t num_my_elems,
                    uint32_t num_perm_segs, uint32_t my_padding_elems, uint32_t end_vocab_idx) {
  {
    uint32_t smem_base = cute::cast_smem_ptr_to_uint(smem.surviving_topk_pairs[0]);
    uint32_t out_ptr = smem_base + start_pos_in_collector * (uint32_t)sizeof(uint64_t);
    uint32_t s0 = local_start_seg_idx + (max(my_elem_start_idx, num_local_tail_elems_padded) -
                                         num_local_tail_elems_padded) /
                                            NUM_ELEMS_PER_SEG;
    uint32_t perm_state = Base::get_permuted_seg_idx(s0, perm_len, perm_mul);
    float unit_base_f = 0.0f;
    CUTE_UNROLL
    for (uint32_t i = 0; i < NUM_UINT32_IN_INIT_WINDOW_PER_THREAD; i++) {
      // num_my_elems is a multiple of NUM_ELEMS_PER_128b (= 8), so i*2 == num_my_elems
      // only at i % 4 == 0; the extra condition lets the compiler unroll by 4.
      if (i % 4 == 0 && i * 2 == num_my_elems)
        break;
      if (i % 4 == 0) {
        uint32_t g_u = my_elem_start_idx + i / 4 * NUM_ELEMS_PER_128b;
        if (g_u < num_local_tail_elems_padded) {
          unit_base_f = __uint_as_float(num_perm_segs * NUM_ELEMS_PER_SEG + g_u);
        } else {
          unit_base_f = __uint_as_float(perm_state * NUM_ELEMS_PER_SEG + g_u % NUM_ELEMS_PER_SEG);
          if (g_u % NUM_ELEMS_PER_SEG == NUM_ELEMS_PER_SEG - NUM_ELEMS_PER_128b) {
            Base::advance_perm_state(perm_state, perm_mul, perm_len);
          }
        }
      }
      // fp32-subnormal FP-pipe int-add: index = base + element offset
      uint32_t index0 = __float_as_uint(unit_base_f + __uint_as_float(i % 4 * 2));
      uint32_t index1 = __float_as_uint(unit_base_f + __uint_as_float(i % 4 * 2 + 1));
      nv_bfloat162 cur_value = init_values[i]; // two bf16 payloads packed as bf16x2
      if (my_padding_elems == 0) {
        append_selected_pairs_from_registers<USE_CLUSTER_ADDRESSING>(
            out_ptr, eq_quota, cur_value, pivot_value_x2_bits, index0, index1);
      } else {
        append_selected_pairs_from_registers<USE_CLUSTER_ADDRESSING, true>(
            out_ptr, eq_quota, cur_value, pivot_value_x2_bits, index0, index1, end_vocab_idx);
      }
    }
  }
}

template <bool USE_CLUSTER_ADDRESSING, class IssueCopy, class IssueMain>
static __device__ __forceinline__ uint32_t initialize_scan(
    SharedMemoryPlanBase &smem, uint32_t topk, uint32_t warp_idx, uint32_t lane_idx,
    uint32_t num_perm_segs, uint32_t local_start_seg_idx, uint32_t num_local_perm_segs,
    uint32_t num_local_tail_elems_padded, uint32_t num_local_tail_elems,
    uint32_t num_local_tail_segs_padded, uint32_t num_init_rounds, uint32_t num_main_rounds,
    uint32_t perm_len, uint32_t perm_mul, uint32_t end_vocab_idx, uint32_t &num_survivors,
    bool &have_nan, IssueCopy &issue_tma_copy, IssueMain &issue_tma_copy_for_main_rounds) {
  ValueT *init_buf = (ValueT *)smem.incoming_topk_pairs;
  prefetch_initial_scan(smem, num_local_tail_elems_padded, num_perm_segs, num_local_tail_elems,
                        num_init_rounds, num_main_rounds, issue_tma_copy,
                        issue_tma_copy_for_main_rounds);

  // Decide how many 128-bit elements each thread processes.
  uint32_t num_elems_in_init_window_padded =
      num_local_tail_elems_padded +
      min(num_local_perm_segs, (uint32_t)NUM_SEGS_IN_INIT_WINDOW - num_local_tail_segs_padded) *
          NUM_ELEMS_PER_SEG; // Init window size, padded to NUM_ELEMS_PER_SEG
  uint32_t num_128b_in_init_window_padded = num_elems_in_init_window_padded / NUM_ELEMS_PER_128b;
  uint32_t cnt_floor = num_128b_in_init_window_padded / NUM_THREADS;
  uint32_t cnt_rem = num_128b_in_init_window_padded % NUM_THREADS;
  uint32_t my_elem_start_idx =
      (threadIdx.x * cnt_floor + min(threadIdx.x, cnt_rem)) * NUM_ELEMS_PER_128b;
  uint32_t num_my_elems =
      NUM_ELEMS_PER_128b * cnt_floor + (threadIdx.x < cnt_rem ? NUM_ELEMS_PER_128b : 0u);
  uint32_t tail_padding_elems = num_local_tail_elems_padded - num_local_tail_elems;
  uint32_t num_real_init_elems = num_elems_in_init_window_padded - tail_padding_elems;
  num_survivors = min(topk, num_real_init_elems);

  histogram_initial_scan(smem, init_buf, num_init_rounds, num_local_tail_elems,
                         num_128b_in_init_window_padded);

  nv_bfloat162 init_values[NUM_UINT32_IN_INIT_WINDOW_PER_THREAD];
  CUTE_UNROLL
  for (uint32_t j = 0; j < NUM_128b_INIT_PER_THREAD; ++j) {
    if (j * NUM_ELEMS_PER_128b == num_my_elems)
      break;
    uint32_t u0 = my_elem_start_idx / NUM_ELEMS_PER_128b + j;
    ld_shared<4>(
        init_values + j * NUM_UINT32_PER_128b,
        reinterpret_cast<const nv_bfloat162 *>(init_buf + Base::sw_b128(u0) * NUM_ELEMS_PER_128b));
  }

  __syncthreads(); // publish the round-1 histogram

  static_assert(NUM_ELEMS_IN_INIT_WINDOW <= 0xFFFF);
  // The CTA's init-window slice may contain fewer real elements than K (e.g. a cluster
  // rank whose visit range mostly overlaps the padded tail). In that case the pivot
  // selection must degrade to "select all real elements": cap the K used in the pivot /
  // quota computation at the real count
  uint32_t effective_topk = min(topk, num_real_init_elems);
  uint32_t my_padding_start = max(my_elem_start_idx, num_local_tail_elems);
  uint32_t my_padding_end = min(my_elem_start_idx + num_my_elems, num_local_tail_elems_padded);
  uint32_t my_padding_elems =
      my_padding_end > my_padding_start ? my_padding_end - my_padding_start : 0;
  auto [pivot_value_x2_bits, start_pos_in_collector, eq_quota, cnt_nan] =
      compute_pivot_and_quota<USE_CLUSTER_ADDRESSING, NUM_UINT32_IN_INIT_WINDOW_PER_THREAD, 4>(
          effective_topk, init_values, num_my_elems / 2, warp_idx, lane_idx, smem,
          my_padding_elems);
  // The init census walks the thread's whole slice of the window, i.e. every element of the init
  // window exactly once, so NaNs inside the window are caught here.
  have_nan |= cnt_nan != 0;

  append_initial_scan<USE_CLUSTER_ADDRESSING>(
      smem, init_values, start_pos_in_collector, eq_quota, pivot_value_x2_bits, local_start_seg_idx,
      my_elem_start_idx, num_local_tail_elems_padded, perm_len, perm_mul, num_my_elems,
      num_perm_segs, my_padding_elems, end_vocab_idx);

  return pivot_value_x2_bits;
}
