#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cute/arch/mma_sm89.hpp>
#include <cute/tensor.hpp>
#include <cstdint>
#include <type_traits>

namespace encoded_projection {

constexpr int fp8_tile_width = 512, fp8_tile_groups = fp8_tile_width / 32;

__device__ __forceinline__ float fp8_scale(uint8_t code) {
  if (code == 255)
    return __uint_as_float(0x7fc00000u);
  return __uint_as_float(code == 0 ? 0x00400000u : uint32_t(code) << 23);
}

__device__ __forceinline__ int fp8_address(int row, int column) {
  return row * fp8_tile_width + (column ^ ((row & 7) * 16));
}

__device__ __forceinline__ uint32_t fp8_shared_four(const uint8_t *values, int row, int rows,
                                                    int column) {
  return row < rows ? *reinterpret_cast<const uint32_t *>(values + fp8_address(row, column)) : 0;
}

template <class Output, class ScaleLayout, int Warps, int RowTiles = 1>
__global__ void
fp8_k32_kernel(const uint8_t *__restrict__ input, const uint8_t *__restrict__ weight,
               const uint8_t *__restrict__ input_scales, const uint8_t *__restrict__ weight_scales,
               Output *__restrict__ output, int rows, int outputs, int width,
               ScaleLayout weight_layout) {
  static_assert(std::is_same_v<Output, float> || std::is_same_v<Output, __nv_bfloat16>);
  using Mma = cute::SM89_16x8x32_F32E4M3E4M3F32_TN;
  constexpr int tile_rows = 16 * RowTiles;
  __shared__ __align__(16) uint8_t data[(tile_rows + Warps * 8) * fp8_tile_width];
  __shared__ uint8_t sa[tile_rows * fp8_tile_groups], sb[Warps * fp8_tile_groups];
  auto *a_shared = data;
  auto *b_shared = data + tile_rows * fp8_tile_width;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int first_row = int(blockIdx.y) * tile_rows, first_column = int(blockIdx.x) * Warps * 8;
  int valid_rows = min(tile_rows, rows - first_row), groups = width / 32;
  int low_row = lane >> 2, first_k = (lane & 3) * 4;
  float total[RowTiles][4] = {};
  for (int start = 0; start < width; start += fp8_tile_width) {
    int columns = min(fp8_tile_width, width - start), stage_groups = columns / 32;
    for (int index = threadIdx.x * 16; index < (valid_rows + Warps * 8) * columns;
         index += blockDim.x * 16) {
      int row = index / columns, column = index % columns;
      bool activation = row < valid_rows;
      int local_row = activation ? row : row - valid_rows;
      const uint8_t *source = activation ? input : weight;
      int global_row = (activation ? first_row : first_column) + local_row;
      auto *destination = activation ? a_shared : b_shared;
      *reinterpret_cast<uint4 *>(destination + fp8_address(local_row, column)) =
          *reinterpret_cast<const uint4 *>(source + int64_t(global_row) * width + start + column);
    }
    for (int index = threadIdx.x; index < valid_rows * stage_groups; index += blockDim.x) {
      int row = index / stage_groups, group = index % stage_groups;
      sa[row * fp8_tile_groups + group] =
          input_scales[int64_t(first_row + row) * groups + start / 32 + group];
    }
    for (int index = threadIdx.x; index < Warps * stage_groups; index += blockDim.x) {
      int tile = index / stage_groups, group = index % stage_groups;
      sb[tile * fp8_tile_groups + group] = weight_scales[weight_layout(
          cute::make_coord(first_column + tile * 8, start + group * 32, 0))];
    }
    __syncthreads();
    if (warp < Warps) {
      for (int group = 0; group < stage_groups; ++group) {
        uint32_t first_a[4];
        if constexpr (RowTiles == 1) {
#pragma unroll
          for (int reg = 0; reg < 4; ++reg)
            first_a[reg] = fp8_shared_four(a_shared, low_row + 8 * (reg & 1), valid_rows,
                                           group * 32 + first_k + 16 * (reg >> 1));
        }
        uint32_t b[2];
#pragma unroll
        for (int reg = 0; reg < 2; ++reg)
          b[reg] = fp8_shared_four(b_shared, warp * 8 + (lane >> 2), Warps * 8,
                                   group * 32 + first_k + 16 * reg);
#pragma unroll
        for (int tile = 0; tile < RowTiles; ++tile) {
          uint32_t a[4];
#pragma unroll
          for (int reg = 0; reg < 4; ++reg) {
            if constexpr (RowTiles == 1)
              a[reg] = first_a[reg];
            else
              a[reg] = fp8_shared_four(a_shared, tile * 16 + low_row + 8 * (reg & 1), valid_rows,
                                       group * 32 + first_k + 16 * (reg >> 1));
          }
          float raw[4];
          Mma::fma(raw[0], raw[1], raw[2], raw[3], a[0], a[1], a[2], a[3], b[0], b[1], 0.f, 0.f,
                   0.f, 0.f);
          float weight_scale = fp8_scale(sb[warp * fp8_tile_groups + group]);
#pragma unroll
          for (int value = 0; value < 4; ++value) {
            int row = tile * 16 + low_row + 8 * (value >> 1);
            float input_scale =
                row < valid_rows ? fp8_scale(sa[row * fp8_tile_groups + group]) : 1.f;
            // The released K32 dot starts at zero, followed by one MUL and one FMA.
            total[tile][value] =
                __fmaf_rn(__fmul_rn(raw[value], input_scale), weight_scale, total[tile][value]);
          }
        }
      }
    }
    __syncthreads();
  }
  if (warp < Warps) {
#pragma unroll
    for (int tile = 0; tile < RowTiles; ++tile) {
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        int row = first_row + tile * 16 + low_row + 8 * (value >> 1);
        int column = first_column + warp * 8 + (lane & 3) * 2 + (value & 1);
        if (row < rows && column < outputs) {
          if constexpr (std::is_same_v<Output, float>)
            output[int64_t(row) * outputs + column] = total[tile][value];
          else
            output[int64_t(row) * outputs + column] = __float2bfloat16_rn(total[tile][value]);
        }
      }
    }
  }
}

template <class Output, class ScaleLayout>
void launch_fp8_k32(const uint8_t *input, const uint8_t *weight, const uint8_t *input_scales,
                    const uint8_t *weight_scales, Output *output, int rows, int outputs, int width,
                    ScaleLayout weight_layout, cudaStream_t stream) {
  if (outputs <= 1024 && width >= 1024)
    fp8_k32_kernel<Output, ScaleLayout, 1><<<dim3(outputs / 8, (rows + 15) / 16), 128, 0, stream>>>(
        input, weight, input_scales, weight_scales, output, rows, outputs, width, weight_layout);
  else if (rows >= 32)
    fp8_k32_kernel<Output, ScaleLayout, 4, 2>
        <<<dim3(outputs / 32, (rows + 31) / 32), 128, 0, stream>>>(input, weight, input_scales,
                                                                   weight_scales, output, rows,
                                                                   outputs, width, weight_layout);
  else
    fp8_k32_kernel<Output, ScaleLayout, 4>
        <<<dim3(outputs / 32, (rows + 15) / 16), 128, 0, stream>>>(input, weight, input_scales,
                                                                   weight_scales, output, rows,
                                                                   outputs, width, weight_layout);
}

} // namespace encoded_projection
