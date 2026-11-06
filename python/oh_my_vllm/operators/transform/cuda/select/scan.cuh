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

#include "scan_initial.cuh"
#include "scan_reconstruct.cuh"
#include "scan_collect.cuh"

struct ScanRounds {
  uint32_t num_local_elems_padded, num_local_rounds, num_init_rounds;
  uint32_t num_local_tail_segs_padded, num_main_rounds, perm_len, perm_mul;
  uint32_t tma_permuted_segment_stride, linear_tma_segment_start, next_tma_permuted_segment;

  __device__ ScanRounds(uint32_t num_local_tail_elems_padded, uint32_t num_local_perm_segs,
                        uint32_t num_perm_segs, uint32_t local_start_seg_idx, uint32_t warp_idx) {
    num_local_elems_padded = num_local_tail_elems_padded + num_local_perm_segs * NUM_ELEMS_PER_SEG;
    num_local_rounds =
        select_platform::ceil_div(num_local_elems_padded, (uint32_t)NUM_ELEMS_PER_ROUND);
    num_init_rounds = min(num_local_rounds, (uint32_t)NUM_INIT_ROUNDS_MAX);
    num_local_tail_segs_padded = num_local_tail_elems_padded / NUM_ELEMS_PER_SEG;
    num_main_rounds = num_local_rounds - num_init_rounds;

    // Permutation generation arguments
    perm_len = max(num_perm_segs, 1u);
    perm_mul = PERM_MUL_PRIME % perm_len;

    // Permutation state
    // Only valid for the lane elected by `elect_one_sync`
    tma_permuted_segment_stride = NUM_ISSUE_WARPS * perm_mul % perm_len;
    linear_tma_segment_start = local_start_seg_idx + warp_idx;
    next_tma_permuted_segment =
        Base::get_permuted_seg_idx(linear_tma_segment_start, perm_len, perm_mul);
  }
};

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
  ScanRounds rounds(num_local_tail_elems_padded, num_local_perm_segs, num_perm_segs,
                    local_start_seg_idx, warp_idx);
  auto issue_tma_copy = [&]<bool IS_INIT, bool HAVE_TAIL>(uint32_t round_idx) {
    Base::template issue_tma_loads_for_round<IS_INIT, HAVE_TAIL>(
        smem, tma_params.tensor_map, batch_idx, end_vocab_idx, num_perm_segs, round_idx,
        rounds.next_tma_permuted_segment, rounds.tma_permuted_segment_stride, warp_idx);
  };
  auto issue_tma_copy_for_main_rounds = [&](uint32_t round_idx) {
    issue_tma_copy.template operator()<false, false>(round_idx);
  };

  uint32_t threshold_x2_bits = NEG_INF_X2_BITS;
  uint32_t num_incomers = 0;
  uint32_t num_survivors = 0;

  // Init phase
  if (rounds.num_local_elems_padded != 0) {
    threshold_x2_bits = initialize_scan<USE_CLUSTER_ADDRESSING>(
        smem, topk, warp_idx, lane_idx, num_perm_segs, local_start_seg_idx, num_local_perm_segs,
        num_local_tail_elems_padded, num_local_tail_elems, rounds.num_local_tail_segs_padded,
        rounds.num_init_rounds, rounds.num_main_rounds, rounds.perm_len, rounds.perm_mul,
        end_vocab_idx, num_survivors, have_nan, issue_tma_copy, issue_tma_copy_for_main_rounds);
  }
  __syncthreads();

  uint32_t linear_segment_start = local_start_seg_idx +
                                  rounds.num_init_rounds * NUM_SEGS_PER_ROUND -
                                  rounds.num_local_tail_segs_padded + warp_idx;
  uint32_t current_permuted_segment =
      Base::get_permuted_seg_idx(linear_segment_start, rounds.perm_len, rounds.perm_mul);
  uint32_t permuted_segment_stride_per_round =
      NUM_SEGS_PER_ROUND % rounds.perm_len * rounds.perm_mul % rounds.perm_len;

  auto reconstruct = [&](uint32_t count) {
    return reconstruct_scan<USE_CLUSTER_ADDRESSING>(smem, topk, warp_idx, lane_idx,
                                                    survivor_buf_idx, have_nan, count);
  };

  scan_remaining_rounds(smem, rounds.num_main_rounds, threshold_x2_bits, num_incomers,
                        current_permuted_segment, permuted_segment_stride_per_round,
                        rounds.perm_len, warp_idx, lane_idx, is_warp_active_f,
                        issue_tma_copy_for_main_rounds, reconstruct);

  return num_survivors;
}
