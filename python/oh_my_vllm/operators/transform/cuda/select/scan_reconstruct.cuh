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

// Re-select the top-k elements from the
// smem.surviving_topk_pairs[survivor_buf_idx][:num_survivors] +
// smem.incoming_topk_pairs[:num_incomers], Write the selected pairs into the other survivor
// buffer, and return the new pivot
template <bool USE_CLUSTER_ADDRESSING>
static __device__ __forceinline__ uint32_t
reconstruct_scan(SharedMemoryPlanBase &smem, uint32_t topk, uint32_t warp_idx, uint32_t lane_idx,
                 uint32_t &survivor_buf_idx, bool &have_nan, uint32_t cur_extra_len) {
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
}
