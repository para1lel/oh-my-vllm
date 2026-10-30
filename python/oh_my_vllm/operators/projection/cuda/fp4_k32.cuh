#pragma once

#include "fp8_k32.cuh"

namespace encoded_projection {

__device__ __forceinline__ uint32_t fp4_to_fp8_four(uint16_t packed) {
  // E2M1 magnitudes have exact E4M3 codes. Move their sign bits separately.
  uint32_t values = __byte_perm(0x3c383000u, 0x4c484440u, packed & 0x7777u);
  uint32_t signs = __byte_perm(packed, uint32_t(packed) << 4, 0x1504u) & 0x80808080u;
  return values | signs;
}

__global__ void fp4_group_offsets(const int32_t *counts, int32_t *offsets, int groups, int rows,
                                  int max_rows) {
  int64_t sum = 0;
  bool valid = true;
  for (int group = 0; group < groups; ++group) {
    offsets[group] = static_cast<int32_t>(sum);
    int count = counts[group];
    valid &= count >= 0 && count <= max_rows;
    sum += count;
  }
  offsets[groups] = valid && sum == rows ? rows : -1;
}

template <class Output, class ScaleLayout, bool Grouped>
__global__ void fp4_k32_kernel(const uint8_t *__restrict__ input, const uint8_t *weight,
                               const uint8_t *__restrict__ input_scales,
                               const uint8_t *weight_scales, Output *__restrict__ output,
                               const int64_t *weight_ptrs, const int64_t *weight_scale_ptrs,
                               const int32_t *offsets, int groups, int rows, int outputs, int width,
                               int max_rows, int weight_stride, ScaleLayout layout) {
  static_assert(std::is_same_v<Output, float> || std::is_same_v<Output, __nv_bfloat16>);
  using Mma = cute::SM89_16x8x32_F32E4M3E4M3F32_TN;
  __shared__ __align__(16) uint8_t data[(16 + 32) * fp8_tile_width];
  __shared__ uint8_t sa[16 * fp8_tile_groups], sb[32 * fp8_tile_groups];
  auto *a_shared = data;
  auto *b_shared = data + 16 * fp8_tile_width;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int first_row = (blockIdx.x / (outputs / 32)) * 16;
  int first_column = (blockIdx.x % (outputs / 32)) * 32;
  int group_rows = rows, row_offset = 0;
  if constexpr (Grouped) {
    int group = blockIdx.y;
    if (offsets[groups] < 0) {
      // Invalid device metadata cannot address any input or weight storage.
      int row = group * max_rows + first_row + lane / 4;
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        int r = row + 8 * (value >> 1);
        int column = first_column + warp * 8 + (lane & 3) * 2 + (value & 1);
        if (r < rows && first_row + lane / 4 + 8 * (value >> 1) < max_rows)
          output[int64_t(r) * outputs + column] = Output(fp8_scale(255));
      }
      return;
    }
    row_offset = offsets[group];
    group_rows = offsets[group + 1] - row_offset;
    if (first_row >= group_rows)
      return;
    weight = reinterpret_cast<const uint8_t *>(weight_ptrs[group]);
    weight_scales = reinterpret_cast<const uint8_t *>(weight_scale_ptrs[group]);
  }
  int valid_rows = min(16, group_rows - first_row), width_groups = width / 32;
  int low_row = lane >> 2, first_k = (lane & 3) * 4;
  float total[4] = {};
  for (int start = 0; start < width; start += fp8_tile_width) {
    int columns = min(fp8_tile_width, width - start), stage_groups = columns / 32;
    for (int index = threadIdx.x * 16; index < (valid_rows + 32) * columns;
         index += blockDim.x * 16) {
      int row = index / columns, column = index % columns;
      if (row < valid_rows) {
        int global_row = row_offset + first_row + row;
        *reinterpret_cast<uint4 *>(a_shared + fp8_address(row, column)) =
            *reinterpret_cast<const uint4 *>(input + int64_t(global_row) * width + start + column);
      } else {
        int local_row = row - valid_rows, global_row = first_column + local_row;
        uint64_t packed = *reinterpret_cast<const uint64_t *>(
            weight + int64_t(global_row) * weight_stride + (start + column) / 2);
        uint4 converted;
        converted.x = fp4_to_fp8_four(packed);
        converted.y = fp4_to_fp8_four(packed >> 16);
        converted.z = fp4_to_fp8_four(packed >> 32);
        converted.w = fp4_to_fp8_four(packed >> 48);
        *reinterpret_cast<uint4 *>(b_shared + fp8_address(local_row, column)) = converted;
      }
    }
    for (int index = threadIdx.x; index < valid_rows * stage_groups; index += blockDim.x) {
      int row = index / stage_groups, group = index % stage_groups;
      sa[row * fp8_tile_groups + group] =
          input_scales[int64_t(row_offset + first_row + row) * width_groups + start / 32 + group];
    }
    for (int index = threadIdx.x; index < 32 * stage_groups; index += blockDim.x) {
      int row = index / stage_groups, group = index % stage_groups;
      sb[row * fp8_tile_groups + group] =
          weight_scales[layout(cute::make_coord(first_column + row, start + group * 32, 0))];
    }
    __syncthreads();
    for (int group = 0; group < stage_groups; ++group) {
      uint32_t a[4], b[2];
#pragma unroll
      for (int reg = 0; reg < 4; ++reg)
        a[reg] = fp8_shared_four(a_shared, low_row + 8 * (reg & 1), valid_rows,
                                 group * 32 + first_k + 16 * (reg >> 1));
#pragma unroll
      for (int reg = 0; reg < 2; ++reg)
        b[reg] =
            fp8_shared_four(b_shared, warp * 8 + (lane >> 2), 32, group * 32 + first_k + 16 * reg);
      float raw[4];
      Mma::fma(raw[0], raw[1], raw[2], raw[3], a[0], a[1], a[2], a[3], b[0], b[1], 0.f, 0.f, 0.f,
               0.f);
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        int row = low_row + 8 * (value >> 1);
        int column = warp * 8 + (lane & 3) * 2 + (value & 1);
        float input_scale = row < valid_rows ? fp8_scale(sa[row * fp8_tile_groups + group]) : 1.f;
        float weight_scale = fp8_scale(sb[column * fp8_tile_groups + group]);
        // Each released K32 dot starts at zero, followed by one MUL and one FMA.
        total[value] = __fmaf_rn(__fmul_rn(raw[value], input_scale), weight_scale, total[value]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int value = 0; value < 4; ++value) {
    int row = first_row + low_row + 8 * (value >> 1);
    int column = first_column + warp * 8 + (lane & 3) * 2 + (value & 1);
    if (row < group_rows) {
      if constexpr (std::is_same_v<Output, float>)
        output[int64_t(row_offset + row) * outputs + column] = total[value];
      else
        output[int64_t(row_offset + row) * outputs + column] = __float2bfloat16_rn(total[value]);
    }
  }
}

template <class Output, class ScaleLayout>
void launch_fp4_k32(const uint8_t *input, const uint8_t *weight, const uint8_t *input_scales,
                    const uint8_t *weight_scales, Output *output, int rows, int outputs, int width,
                    int weight_stride, ScaleLayout layout, cudaStream_t stream) {
  int tiles = (outputs / 32) * ((rows + 15) / 16);
  fp4_k32_kernel<Output, ScaleLayout, false><<<tiles, 128, 0, stream>>>(
      input, weight, input_scales, weight_scales, output, nullptr, nullptr, nullptr, 1, rows,
      outputs, width, rows, weight_stride, layout);
}

template <class Output, class ScaleLayout>
void launch_grouped_fp4_k32(const uint8_t *input, const uint8_t *input_scales,
                            const int64_t *weight_ptrs, const int64_t *weight_scale_ptrs,
                            const int32_t *counts, int32_t *offsets, Output *output, int groups,
                            int rows, int outputs, int width, int max_rows, int weight_stride,
                            ScaleLayout layout, cudaStream_t stream) {
  fp4_group_offsets<<<1, 1, 0, stream>>>(counts, offsets, groups, rows, max_rows);
  int tiles = (outputs / 32) * ((max_rows + 15) / 16);
  fp4_k32_kernel<Output, ScaleLayout, true><<<dim3(tiles, groups), 128, 0, stream>>>(
      input, nullptr, input_scales, nullptr, output, weight_ptrs, weight_scale_ptrs, offsets,
      groups, rows, outputs, width, max_rows, weight_stride, layout);
}

} // namespace encoded_projection
