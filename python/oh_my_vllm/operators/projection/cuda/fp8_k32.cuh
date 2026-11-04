#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cute/arch/mma_sm89.hpp>
#include <cute/tensor.hpp>
#include <cstdint>
#include <type_traits>

namespace encoded_projection {

constexpr int fp8_tile_width = 512, fp8_tile_groups = fp8_tile_width / 32;

template <class Output, class ScaleLayout, int Bits, bool Grouped>
void launch_half_k32(const uint8_t *input, const uint8_t *weight, const uint8_t *input_scales,
                     const uint8_t *weight_scales, Output *output, const int64_t *weight_ptrs,
                     const int64_t *scale_ptrs, const int32_t *offsets, const int32_t *tiles,
                     int groups, int rows, int outputs, int width, int weight_stride,
                     ScaleLayout layout, cudaStream_t stream);

template <class Output, class ScaleLayout>
void launch_tcgen_k32(const uint8_t *input, const uint8_t *weight, const uint8_t *input_scales,
                      const uint8_t *weight_scales, Output *output, int rows, int outputs,
                      int width, ScaleLayout weight_layout, cudaStream_t stream);

__device__ __forceinline__ float fp8_scale(uint8_t code) {
  if (code == 255)
    return __uint_as_float(0x7fc00000u);
  return __uint_as_float(code == 0 ? 0x00400000u : uint32_t(code) << 23);
}

__device__ __forceinline__ float fp8_scaled_add(float raw, uint8_t activation, uint8_t weight,
                                                float total) {
  int combined = int(activation) + int(weight);
  // K32 raw dots are multiples of 2^-18 and have magnitude below 2^23.
  // These bounds make the first power-of-two multiply exact, including subnormals.
  if (activation <= 232 && weight != 255 && combined >= 105 && combined <= 381) {
    int exponent = combined - 254;
    uint32_t bits = exponent >= -126 ? uint32_t(exponent + 127) << 23 : 1u << (exponent + 149);
    return __fmaf_rn(raw, __uint_as_float(bits), total);
  }
  return __fmaf_rn(__fmul_rn(raw, fp8_scale(activation)), fp8_scale(weight), total);
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
__global__ void
fp8_decode_kernel(const uint8_t *__restrict__ input, const uint8_t *__restrict__ weight,
                  const uint8_t *__restrict__ input_scales,
                  const uint8_t *__restrict__ weight_scales, Output *__restrict__ output,
                  int outputs, int width, ScaleLayout weight_layout) {
  using Mma = cute::SM89_16x8x32_F32E4M3E4M3F32_TN;
  int lane = threadIdx.x, column = blockIdx.x * 8;
  int k = (lane & 3) * 4, n = lane >> 2, groups = width / 32;
  float total[4] = {};
  for (int start = 0; start < groups; start += 4) {
    uint32_t a_values[4][4], b_values[4][2];
    float sa[4], sb[4];
#pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
      int group = start + stage;
      if (group < groups) {
        uint32_t a0 = *reinterpret_cast<const uint32_t *>(input + group * 32 + k);
        uint32_t a1 = *reinterpret_cast<const uint32_t *>(input + group * 32 + k + 16);
        a_values[stage][0] = a0;
        a_values[stage][1] = 0;
        a_values[stage][2] = a1;
        a_values[stage][3] = 0;
        b_values[stage][0] = *reinterpret_cast<const uint32_t *>(
            weight + int64_t(column + n) * width + group * 32 + k);
        b_values[stage][1] = *reinterpret_cast<const uint32_t *>(
            weight + int64_t(column + n) * width + group * 32 + k + 16);
        sa[stage] = fp8_scale(input_scales[group]);
        sb[stage] =
            fp8_scale(weight_scales[weight_layout(cute::make_coord(column, group * 32, 0))]);
      }
    }
#pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
      if (start + stage < groups) {
        float raw[4];
        Mma::fma(raw[0], raw[1], raw[2], raw[3], a_values[stage][0], a_values[stage][1],
                 a_values[stage][2], a_values[stage][3], b_values[stage][0], b_values[stage][1],
                 0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int value = 0; value < 4; ++value)
          total[value] = __fmaf_rn(__fmul_rn(raw[value], sa[stage]), sb[stage], total[value]);
      }
    }
  }
  if (lane < 4) {
#pragma unroll
    for (int value = 0; value < 2; ++value) {
      int index = column + lane * 2 + value;
      if constexpr (std::is_same_v<Output, float>)
        output[index] = total[value];
      else
        output[index] = __float2bfloat16_rn(total[value]);
    }
  }
}

template <class Output, class ScaleLayout>
__global__ void
fp8_decode_rawdots(const uint8_t *__restrict__ input, const uint8_t *__restrict__ weight,
                   const uint8_t *__restrict__ input_scales,
                   const uint8_t *__restrict__ weight_scales, Output *__restrict__ output,
                   int width, ScaleLayout weight_layout) {
  using Mma = cute::SM89_16x8x32_F32E4M3E4M3F32_TN;
  __shared__ float raw_dots[512][8];
  __shared__ float activation_scales[512], scales[512];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int k = (lane & 3) * 4, n = lane >> 2, column = blockIdx.x * 8, groups = width / 32;
  for (int group = warp; group < groups; group += 8) {
    uint32_t a0 = *reinterpret_cast<const uint32_t *>(input + group * 32 + k);
    uint32_t a1 = *reinterpret_cast<const uint32_t *>(input + group * 32 + k + 16);
    uint32_t b0 =
        *reinterpret_cast<const uint32_t *>(weight + int64_t(column + n) * width + group * 32 + k);
    uint32_t b1 = *reinterpret_cast<const uint32_t *>(weight + int64_t(column + n) * width +
                                                      group * 32 + k + 16);
    float raw[4];
    Mma::fma(raw[0], raw[1], raw[2], raw[3], a0, 0u, a1, 0u, b0, b1, 0.f, 0.f, 0.f, 0.f);
    if (lane < 4) {
      raw_dots[group][lane * 2] = raw[0];
      raw_dots[group][lane * 2 + 1] = raw[1];
    }
    if (lane == 0) {
      activation_scales[group] = fp8_scale(input_scales[group]);
      scales[group] =
          fp8_scale(weight_scales[weight_layout(cute::make_coord(column, group * 32, 0))]);
    }
  }
  __syncthreads();
  if (threadIdx.x < 8) {
    float total = 0.f;
    for (int group = 0; group < groups; ++group)
      total = __fmaf_rn(__fmul_rn(raw_dots[group][threadIdx.x], activation_scales[group]),
                        scales[group], total);
    if constexpr (std::is_same_v<Output, float>)
      output[column + threadIdx.x] = total;
    else
      output[column + threadIdx.x] = __float2bfloat16_rn(total);
  }
}

template <class Output, class ScaleLayout>
void launch_fp8_k32(const uint8_t *input, const uint8_t *weight, const uint8_t *input_scales,
                    const uint8_t *weight_scales, Output *output, int rows, int outputs, int width,
                    ScaleLayout weight_layout, cudaStream_t stream) {
  if (rows == 1 && (width >= 2048 || (width >= 1024 && outputs <= 1024)))
    fp8_decode_rawdots<Output, ScaleLayout><<<outputs / 8, 256, 0, stream>>>(
        input, weight, input_scales, weight_scales, output, width, weight_layout);
  else if (rows == 1)
    fp8_decode_kernel<Output, ScaleLayout><<<outputs / 8, 32, 0, stream>>>(
        input, weight, input_scales, weight_scales, output, outputs, width, weight_layout);
  else if (outputs <= 1024 && width >= 1024)
    fp8_k32_kernel<Output, ScaleLayout, 1><<<dim3(outputs / 8, (rows + 15) / 16), 128, 0, stream>>>(
        input, weight, input_scales, weight_scales, output, rows, outputs, width, weight_layout);
  else if (rows >= 128 && outputs > 1024 && outputs % 256 == 0 && width <= 6144 &&
           reinterpret_cast<uintptr_t>(input) % 128 == 0 &&
           reinterpret_cast<uintptr_t>(weight) % 128 == 0)
    launch_tcgen_k32(input, weight, input_scales, weight_scales, output, rows, outputs, width,
                     weight_layout, stream);
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
