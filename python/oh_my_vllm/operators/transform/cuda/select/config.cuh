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

#include <cstdint>

template <typename ValueT_, typename OutIdxT_, bool sorted_value_, bool sorted_index_,
          bool return_value_, uint32_t max_topk_, uint32_t num_threads_, uint32_t target_occupancy_,
          uint32_t elements_per_round_, uint32_t reconstruct_threshold_, uint32_t tma_buffer_depth_,
          uint32_t elements_per_segment_ = 512, uint32_t cluster_size_ = 1>
struct TopkSelectConfig {
  using ValueT = ValueT_;
  using OutIdxT = OutIdxT_;
  static constexpr bool sorted_value = sorted_value_;
  static constexpr bool sorted_index = sorted_index_;
  static constexpr bool return_value = return_value_;
  static constexpr uint32_t max_topk = max_topk_;
  static constexpr uint32_t num_threads = num_threads_;
  static constexpr uint32_t target_occupancy = target_occupancy_; // Number of CTAs per SM
  static constexpr uint32_t elements_per_round = elements_per_round_;
  static constexpr uint32_t reconstruct_threshold = reconstruct_threshold_;
  static constexpr uint32_t tma_buffer_depth = tma_buffer_depth_;
  static constexpr uint32_t elements_per_segment = elements_per_segment_;
  static constexpr uint32_t cluster_size = cluster_size_; // (1 = not a cluster variant)
};
