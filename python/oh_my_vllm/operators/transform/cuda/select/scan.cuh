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

// Main part of the normal kernel and phase A of the cluster kernel.
// The input is divided into:
//  - a "perm" prefix [0, num_perm_segs * NUM_ELEMS_PER_SEG), visited in pseudo-random order;
//  - a "tail" suffix [num_perm_segs * NUM_ELEMS_PER_SEG, end_vocab_idx), visited sequentially.
// The init phase loads the tail plus the first perm segments into one init window, selects
// the initial top-k from it, then the main loop continues over the remaining perm segments.
//
// Returns the number of real selected pairs in the final survivor buffer; this is normally
// `topk`, but can be smaller in a cluster CTA whose local range has fewer real elements.
template <bool USE_CLUSTER_ADDRESSING, typename IsWarpActiveF>
static __device__ __forceinline__ uint32_t scan_segs(
    const TmaParams &tma_params, SharedMemoryPlanBase &smem, uint32_t batch_idx,
    uint32_t end_vocab_idx, uint32_t topk, uint32_t warp_idx, uint32_t lane_idx,
    uint32_t num_perm_segs,       // The number of segments to be permuted, globally
    uint32_t local_start_seg_idx, // The index of the first "local" segment ("local" means "belong
                                  // to this CTA"). 0 if not the cluster-based implementation
    uint32_t num_local_perm_segs, // The number of permuted segments to be processed, locally
    uint32_t
        num_local_tail_elems_padded, // num_local_tail_elems, padded to NUM_TAIL_ELEMS if
                                     // num_perm_segs > 0, otherwise padded to NUM_ELEMS_PER_SEG
    uint32_t num_local_tail_elems,   // The number of tail elems locally
    uint32_t &survivor_buf_idx, bool &have_nan, IsWarpActiveF &&is_warp_active_f) {
  uint32_t num_local_elems_padded =
      num_local_tail_elems_padded + num_local_perm_segs * NUM_ELEMS_PER_SEG;
  uint32_t num_local_rounds =
      select_platform::ceil_div(num_local_elems_padded, (uint32_t)NUM_ELEMS_PER_ROUND);
  uint32_t num_init_rounds = min(num_local_rounds, (uint32_t)NUM_INIT_ROUNDS_MAX);
  uint32_t num_local_tail_segs_padded = num_local_tail_elems_padded / NUM_ELEMS_PER_SEG;
  uint32_t num_main_rounds = num_local_rounds - num_init_rounds;

  // Permutation generation arguments
  uint32_t perm_len = max(num_perm_segs, 1u);
  uint32_t perm_mul = PERM_MUL_PRIME % perm_len;

  // Permutation state
  // Only valid for the lane elected by `elect_one_sync`
  uint32_t tma_permuted_segment_stride = NUM_ISSUE_WARPS * perm_mul % perm_len;
  uint32_t linear_tma_segment_start = local_start_seg_idx + warp_idx;
  uint32_t next_tma_permuted_segment =
      Base::get_permuted_seg_idx(linear_tma_segment_start, perm_len, perm_mul);

  auto issue_tma_copy = [&]<bool IS_INIT, bool HAVE_TAIL>(uint32_t round_idx) {
    Base::template issue_tma_loads_for_round<IS_INIT, HAVE_TAIL>(
        smem, tma_params.tensor_map, batch_idx, end_vocab_idx, num_perm_segs, round_idx,
        next_tma_permuted_segment, tma_permuted_segment_stride, warp_idx);
  };
  auto issue_tma_copy_for_main_rounds = [&](uint32_t round_idx) {
    issue_tma_copy.template operator()<false, false>(round_idx);
  };

  uint32_t threshold_x2_bits = NEG_INF_X2_BITS;
  uint32_t num_incomers = 0;
  uint32_t num_survivors = 0;

  // Init phase
  if (num_local_elems_padded != 0) {
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

    nv_bfloat162 init_values[NUM_UINT32_IN_INIT_WINDOW_PER_THREAD];
    CUTE_UNROLL
    for (uint32_t j = 0; j < NUM_128b_INIT_PER_THREAD; ++j) {
      if (j * NUM_ELEMS_PER_128b == num_my_elems)
        break;
      uint32_t u0 = my_elem_start_idx / NUM_ELEMS_PER_128b + j;
      ld_shared<4>(init_values + j * NUM_UINT32_PER_128b,
                   reinterpret_cast<const nv_bfloat162 *>(init_buf +
                                                          Base::sw_b128(u0) * NUM_ELEMS_PER_128b));
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

    threshold_x2_bits = pivot_value_x2_bits;
  }
  __syncthreads();

  uint32_t linear_segment_start = local_start_seg_idx + num_init_rounds * NUM_SEGS_PER_ROUND -
                                  num_local_tail_segs_padded + warp_idx;
  uint32_t current_permuted_segment =
      Base::get_permuted_seg_idx(linear_segment_start, perm_len, perm_mul);
  uint32_t permuted_segment_stride_per_round = NUM_SEGS_PER_ROUND % perm_len * perm_mul % perm_len;

  // Re-select the top-k elements from the
  // smem.surviving_topk_pairs[survivor_buf_idx][:num_survivors] +
  // smem.incoming_topk_pairs[:num_incomers], Write the selected pairs into the other survivor
  // buffer, and return the new pivot
  auto reconstruct = [&](uint32_t cur_extra_len) -> uint32_t {
    uint32_t tid = threadIdx.x;
    Base::clear_reconstruct_histograms(smem, tid);

    uint32_t padded_extra_len =
        select_platform::ceil(cur_extra_len, 2u); // Since we process elements in pairs (uint32_t)
    if (warp_idx == 0 && (cur_extra_len & 1u)) {
      smem.incoming_topk_pairs[cur_extra_len] = PLACEHOLDER_PAIR;
    }
    __syncthreads();

    // units [0, MAX_TOPK/2): the current survivor buffer
    // The rest: incomers
    uint32_t num_uint32 = (MAX_TOPK + padded_extra_len) / 2;
    uint32_t num_uint32_per_thread = select_platform::ceil_div(num_uint32, (uint32_t)NUM_THREADS) |
                                     1u; // force odd => bank-conflict-free smem gathers
    uint32_t unit_base = tid * num_uint32_per_thread;
    uint32_t num_my_units =
        unit_base < num_uint32 ? min(num_uint32_per_thread, num_uint32 - unit_base) : 0u;

    auto unit_to_pair_addr = [&](uint32_t offset) -> uint32_t {
      return offset < MAX_TOPK / 2
                 ? cute::cast_smem_ptr_to_uint(smem.surviving_topk_pairs[survivor_buf_idx] +
                                               2 * offset)
                 : cute::cast_smem_ptr_to_uint(smem.incoming_topk_pairs + (2 * offset - MAX_TOPK));
    };

    nv_bfloat162 values[NUM_UINT32_RECONSTRUCT_PER_THREAD];
    CUTE_UNROLL
    for (uint32_t m = 0; m < NUM_UINT32_RECONSTRUCT_PER_THREAD; m++) {
      if (m == num_my_units)
        break;
      uint32_t pair2[4];
      ld_shared<4>(pair2, (const uint32_t *)unit_to_pair_addr(unit_base + m));
      // One b128 unit = 2 pairs {index0, value0, index1, value1}; pick the low halves of the
      // two value words (the bf16 payloads) and pack them into one bf16x2 word.
      values[m] = u32_to_bf16x2(__byte_perm(pair2[1], pair2[3], 0x5410));
    }

    histogram_radix_msb(smem.reconstruct_bucket_counter[0], values, num_my_units);
    __syncthreads();

    auto [pivot_value_x2_bits, out_prefix, eq_quota, cnt_nan] =
        compute_pivot_and_quota<USE_CLUSTER_ADDRESSING>(topk, values, num_my_units, warp_idx,
                                                        lane_idx, smem);
    // Every NaN that the main loop collected is part of the buffer this census just walked (the hit
    // test is `.gtu`, so NaN always becomes an incomer).
    have_nan |= cnt_nan != 0;

    { // write back
      uint32_t out_ptr =
          cute::cast_smem_ptr_to_uint(smem.surviving_topk_pairs[survivor_buf_idx ^ 1]) +
          out_prefix * (uint32_t)sizeof(uint64_t);
      CUTE_UNROLL
      for (uint32_t m = 0; m < NUM_UINT32_RECONSTRUCT_PER_THREAD; m++) {
        if (m == num_my_units)
          break;
        copy_selected_pairs_to_survivor<USE_CLUSTER_ADDRESSING>(
            out_ptr, eq_quota, values[m], pivot_value_x2_bits, unit_to_pair_addr(unit_base + m));
      }
    }

    survivor_buf_idx ^= 1;
    return pivot_value_x2_bits;
  };

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

    bool is_warp_active = is_warp_active_f(main_round_idx);

    const ValueT *buf = smem.tma_load_buf[tma_buf_idx];
    // One bit per element of this thread's slice, in element order.
    // Hits are rare, so appending through the set bits of this mask is faster than testing +
    // storing every element again.
    static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND <= 32); // hit_mask is a uint32_t
    static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND % 4 ==
                  0); // the loop below packs 4 elements per nibble
    uint32_t hit_mask = 0;
    if (is_warp_active) {
      smem.tma_load_full_bar[tma_buf_idx].wait(tma_buf_phase);
      ValueT values[NUM_ELEMS_PER_THREAD_PER_ROUND];
      Base::template load_swizzled_slice<NUM_128b_PER_THREAD_PER_ROUND>(
          values, buf, smem_read_offset, chunk_swizzle_mask);
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

  return num_survivors;
}
