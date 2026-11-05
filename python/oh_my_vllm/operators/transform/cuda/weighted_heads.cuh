#pragma once

#include <cuda_bf16.h>
#include <limits>
#include "../../runtime/cuda/contracts.cuh"

namespace {

union alignas(16) BfloatVector {
  uint4 packed;
  __nv_bfloat16 values[8];
};

__global__ void weighted_heads_kernel(const __nv_bfloat16 *__restrict__ values,
                                      const __nv_bfloat16 *__restrict__ weights,
                                      float *__restrict__ output, int rows, int width) {
  int64_t vectors = int64_t(rows) * (width / 8);
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < vectors;
       index += int64_t(gridDim.x) * blockDim.x) {
    int row = index / (width / 8), column = (index % (width / 8)) * 8;
    // Keep the four independent accumulators of the BF16 head reduction.
    float partial[4][8] = {};
#pragma unroll
    for (int head = 0; head < 8; ++head) {
      BfloatVector inputs;
      inputs.packed =
          *reinterpret_cast<const uint4 *>(values + (int64_t(row) * 8 + head) * width + column);
      float weight = __bfloat162float(weights[int64_t(row) * 8 + head]);
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        float value = __bfloat162float(inputs.values[element]);
        float product =
            __bfloat162float(__float2bfloat16_rn(__fmul_rn(value < 0.f ? 0.f : value, weight)));
        partial[head % 4][element] = __fadd_rn(partial[head % 4][element], product);
      }
    }
    float result[8];
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      float sum = __fadd_rn(partial[0][element], partial[1][element]);
      sum = __fadd_rn(sum, partial[2][element]);
      sum = __fadd_rn(sum, partial[3][element]);
      result[element] = __bfloat162float(__float2bfloat16_rn(sum));
    }
    auto *destination = output + int64_t(row) * width + column;
    *reinterpret_cast<float4 *>(destination) =
        make_float4(result[0], result[1], result[2], result[3]);
    *reinterpret_cast<float4 *>(destination + 4) =
        make_float4(result[4], result[5], result[6], result[7]);
  }
}

template <bool AlignedPairs>
__global__ void weighted_heads_pairs_kernel(const __nv_bfloat16 *__restrict__ values,
                                            const __nv_bfloat16 *__restrict__ weights,
                                            float *__restrict__ output, int rows, int width) {
  int pairs = width / 2 + width % 2;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < int64_t(rows) * pairs; index += int64_t(gridDim.x) * blockDim.x) {
    int row = index / pairs, column = (index % pairs) * 2;
    float2 partial[4] = {};
    auto zero = __float2bfloat162_rn(0.f);
#pragma unroll
    for (int head = 0; head < 8; ++head) {
      const auto *source = values + (int64_t(row) * 8 + head) * width + column;
      __nv_bfloat162 value;
      if constexpr (AlignedPairs)
        value = *reinterpret_cast<const __nv_bfloat162 *>(source);
      else
        value = __halves2bfloat162(source[0],
                                   column + 1 < width ? source[1] : __float2bfloat16_rn(0.f));
      // The comparison keeps negative zero, as in the source ReLU.
      uint32_t negative = __hgt2_mask(zero, value);
      __nv_bfloat162_raw positive = value;
      positive.x &= ~negative;
      positive.y &= ~(negative >> 16);
      auto weight = __bfloat162bfloat162(weights[int64_t(row) * 8 + head]);
      float2 product = __bfloat1622float2(__hmul2(__nv_bfloat162(positive), weight));
      // Retain the four independent accumulators of the BF16 head reduction.
      partial[head % 4].x = __fadd_rn(partial[head % 4].x, product.x);
      partial[head % 4].y = __fadd_rn(partial[head % 4].y, product.y);
    }
    float2 sum = partial[0];
#pragma unroll
    for (int i = 1; i < 4; ++i) {
      sum.x = __fadd_rn(sum.x, partial[i].x);
      sum.y = __fadd_rn(sum.y, partial[i].y);
    }
    float2 rounded = __bfloat1622float2(__floats2bfloat162_rn(sum.x, sum.y));
    float *destination = output + int64_t(row) * width + column;
    if constexpr (AlignedPairs) {
      *reinterpret_cast<float2 *>(destination) = rounded;
    } else {
      destination[0] = rounded.x;
      if (column + 1 < width)
        destination[1] = rounded.y;
    }
  }
}

} // namespace

void weighted_heads(TensorView values, TensorView weights, TensorView output) {
  TVM_FFI_ICHECK(values.ndim() == 3 && values.size(1) == 8);
  int64_t rows = values.size(0), width = values.size(2);
  TVM_FFI_ICHECK(rows >= 0 && rows <= std::numeric_limits<int>::max());
  TVM_FFI_ICHECK(width > 0 && width <= std::numeric_limits<int>::max());
  TVM_FFI_ICHECK(rows <= std::numeric_limits<int64_t>::max() / 16 / width);
  tensor_contract(values, values, kDLBfloat, 16, {rows, 8, width});
  tensor_contract(weights, values, kDLBfloat, 16, {rows, 8});
  tensor_contract(output, values, kDLFloat, 32, {rows, width});
  disjoint(values, output);
  disjoint(weights, output);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  int64_t vectors = rows * (width % 8 ? width / 2 + width % 2 : width / 8);
  int64_t blocks = (vectors + 255) / 256;
  unsigned grid = static_cast<unsigned>(blocks > 65535 ? 65535 : blocks);
  if (width % 8 == 0) {
    weighted_heads_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(values.data_ptr()),
        static_cast<const __nv_bfloat16 *>(weights.data_ptr()),
        static_cast<float *>(output.data_ptr()), static_cast<int>(rows), static_cast<int>(width));
  } else if (width % 2) {
    weighted_heads_pairs_kernel<false><<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(values.data_ptr()),
        static_cast<const __nv_bfloat16 *>(weights.data_ptr()),
        static_cast<float *>(output.data_ptr()), static_cast<int>(rows), static_cast<int>(width));
  } else {
    weighted_heads_pairs_kernel<true><<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(values.data_ptr()),
        static_cast<const __nv_bfloat16 *>(weights.data_ptr()),
        static_cast<float *>(output.data_ptr()), static_cast<int>(rows), static_cast<int>(width));
  }
  check_cuda(cudaGetLastError());
}
