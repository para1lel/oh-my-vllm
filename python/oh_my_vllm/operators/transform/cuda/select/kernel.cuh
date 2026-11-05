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
#include "bfloat.cuh"

namespace topk_select_bf16_normal {

template <typename Config>
class TopkSelectKernelBF16Normal : public topk_select_common::TopkSelectKernelBF16Base<Config> {
  using BF16Base = topk_select_common::TopkSelectKernelBF16Base<Config>;

public:
  using ValueT = typename BF16Base::ValueT;
  using OutIdxT = typename BF16Base::OutIdxT;
  using TmaParams = typename BF16Base::TmaParams;
  using SharedMemoryPlanBase = typename BF16Base::SharedMemoryPlanBase;
  using EpilogueT = typename BF16Base::EpilogueT;
  using BF16Base::MAX_TOPK;
  using BF16Base::NUM_ELEMS_PER_ROUND;
  using BF16Base::NUM_ELEMS_PER_SEG;
  using BF16Base::NUM_INIT_ROUNDS_MAX;
  using BF16Base::NUM_SEGS_IN_INIT_WINDOW;
  using BF16Base::NUM_SEGS_PER_ROUND;
  using BF16Base::NUM_TAIL_ELEMS;
  using BF16Base::NUM_TAIL_SEGS;
  using BF16Base::NUM_THREADS;

  static_assert(Config::target_occupancy >= 1 && Config::target_occupancy <= 8);
  static_assert(MAX_TOPK == 512 || MAX_TOPK == 1024 || MAX_TOPK == 4096);
  static constexpr bool HAS_PARTIAL_ROUNDS = NUM_SEGS_PER_ROUND > NUM_TAIL_SEGS;

  struct SharedMemoryPlan : SharedMemoryPlanBase {};

  static __device__ __forceinline__ void topk_select_kernel_devfunc(const TopkSelectArgs &args,
                                                                    const TmaParams &tma_params) {
    uint32_t batch_idx = blockIdx.x;
    uint32_t end_vocab_idx =
        args.end_ptr == nullptr ? args.vocab_size : __ldg(args.end_ptr + batch_idx);

    extern __shared__ CUTE_ALIGNAS(1024) char wksp_buf[];
    SharedMemoryPlan &smem = *reinterpret_cast<SharedMemoryPlan *>(wksp_buf);

    uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    uint32_t lane_idx = threadIdx.x % 32;

    if (end_vocab_idx <= args.topk) {
      EpilogueT::template topk_select_epilogue<true>(
          (ValueT *)smem.surviving_topk_pairs[0], (uint32_t *)(smem.surviving_topk_pairs[1]), args,
          batch_idx, end_vocab_idx, warp_idx,
          *reinterpret_cast<typename EpilogueT::BlockRadixSortTempStorageT *>(
              smem.incoming_topk_pairs));
      return;
    }

    uint32_t survivor_buf_idx = 0; // Current candidate buffer (A/B, swapped by reconstruct)
    bool have_nan = false;

    BF16Base::init_shared_memory(smem, warp_idx);

    uint32_t num_input_segs = select_platform::ceil_div(end_vocab_idx, (uint32_t)NUM_ELEMS_PER_SEG);
    uint32_t num_perm_segs =
        num_input_segs <= NUM_SEGS_IN_INIT_WINDOW
            ? 0u
            : (num_input_segs - 1) / NUM_TAIL_SEGS *
                  NUM_TAIL_SEGS; // Skip permutation if the total number of segs is small enough
    uint32_t num_perm_elems = num_perm_segs * NUM_ELEMS_PER_SEG;
    uint32_t num_tail_elems_padded =
        num_perm_segs != 0 ? (uint32_t)NUM_TAIL_ELEMS : num_input_segs * NUM_ELEMS_PER_SEG;
    uint32_t num_rounds = select_platform::ceil_div(num_tail_elems_padded + num_perm_elems,
                                                    (uint32_t)NUM_ELEMS_PER_ROUND);
    uint32_t num_init_rounds = min(num_rounds, (uint32_t)NUM_INIT_ROUNDS_MAX);

    BF16Base::template scan_segs<false>(
        tma_params, smem, batch_idx, end_vocab_idx, args.topk, warp_idx, lane_idx, num_perm_segs, 0,
        num_perm_segs, num_tail_elems_padded, end_vocab_idx - num_perm_elems, survivor_buf_idx,
        have_nan, [&](uint32_t m) {
          if constexpr (HAS_PARTIAL_ROUNDS) {
            return (num_init_rounds + m) * NUM_SEGS_PER_ROUND + warp_idx <
                   NUM_TAIL_SEGS + num_perm_segs;
          } else {
            return true;
          }
        });

    // `__syncthreads_or` is also the barrier that makes the last `reconstruct`'s writes visible
    // before `stage_output_and_epilogue` reads the survivor buffer.
    have_nan = __syncthreads_or(have_nan) != 0;
    if (have_nan) {
      BF16Base::take_action_when_have_nan(args, batch_idx);
      return;
    }

    BF16Base::template stage_output_and_epilogue<Config::sorted_index>(
        smem,
        *reinterpret_cast<typename EpilogueT::BlockRadixSortTempStorageT *>(
            smem.incoming_topk_pairs),
        args, batch_idx, end_vocab_idx, survivor_buf_idx, warp_idx, lane_idx);
  }
};

template <typename Kernel>
__launch_bounds__(Kernel::NUM_THREADS, Kernel::TARGET_OCCUPANCY, 1) __global__
    void topk_kernel(__grid_constant__ const TopkSelectArgs args,
                     __grid_constant__ const typename Kernel::TmaParams tma_params) {
  Kernel::topk_select_kernel_devfunc(args, tma_params);
}

} // namespace topk_select_bf16_normal
