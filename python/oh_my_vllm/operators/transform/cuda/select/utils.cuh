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

template <typename T>
__device__ __forceinline__ T warp_level_inclusive_prefix_sum(T x, uint32_t lane_idx) {
  static_assert(sizeof(T) == 4);
#pragma unroll
  for (uint32_t i = 1; i <= 16; i <<= 1) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 ".reg .b32 t;\n"
                 "shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff;\n"
                 "@p add.u32 %0, %0, t;\n"
                 "}\n"
                 : "+r"(x)
                 : "r"(i));
  }
  return x;
}

template <typename T>
__device__ __forceinline__ T warp_level_exclusive_prefix_sum(T x, uint32_t lane_idx) {
  T inclusive_prefix_sum = warp_level_inclusive_prefix_sum(x, lane_idx);
  return inclusive_prefix_sum - x;
}

template <typename T>
__device__ __forceinline__ T warp_level_inclusive_suffix_sum(T x, uint32_t lane_idx) {
  static_assert(sizeof(T) == 4);
#pragma unroll
  for (uint32_t i = 1; i <= 16; i <<= 1) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 ".reg .b32 t;\n"
                 "shfl.sync.down.b32 t|p, %0, %1, 31, 0xffffffff;\n"
                 "@p add.u32 %0, %0, t;\n"
                 "}\n"
                 : "+r"(x)
                 : "r"(i));
  }
  return x;
}

template <typename T>
__device__ __forceinline__ T warp_level_exclusive_suffix_sum(T x, uint32_t lane_idx) {
  T inclusive_suffix_sum = warp_level_inclusive_suffix_sum(x, lane_idx);
  return inclusive_suffix_sum - x;
}
