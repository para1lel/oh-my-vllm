#pragma once

#include "half_k32.cuh"

namespace encoded_projection {

constexpr int prefill_rows = 128, prefill_columns = 128, prefill_stage = 128;
constexpr int prefill_raw_a = prefill_rows * prefill_stage;
constexpr int prefill_raw_bytes = prefill_raw_a + prefill_columns * prefill_stage / 2;

__device__ __forceinline__ void prefill_half_store(uint16_t *destination, int row, int column,
                                                   uint4 values) {
  int reordered = (column / 32) * 32 + (column % 32) / 2;
  uint64_t low = uint64_t(values.x) | (uint64_t(values.z) << 32);
  uint64_t high = uint64_t(values.y) | (uint64_t(values.w) << 32);
  int first = (row & 1) * 16;
  *reinterpret_cast<uint64_t *>(destination + half_address(row, reordered + first)) =
      first ? high : low;
  *reinterpret_cast<uint64_t *>(destination + half_address(row, reordered + (first ^ 16))) =
      first ? low : high;
}

__device__ __forceinline__ void prefill_copy_raw(uint8_t *destination, const uint8_t *input,
                                                 const uint8_t *weight, int row_offset,
                                                 int first_row, int first_column, int valid_rows,
                                                 int outputs, int width, int weight_stride,
                                                 int start) {
  for (int index = threadIdx.x * 16; index < prefill_raw_bytes; index += blockDim.x * 16) {
    bool activation = index < prefill_raw_a;
    int local = activation ? index : index - prefill_raw_a;
    int row_width = activation ? prefill_stage : prefill_stage / 2;
    int row = local / row_width, column = local % row_width;
    bool valid = activation ? row < valid_rows : first_column + row < outputs;
    const auto *source =
        activation ? input + int64_t(row_offset + first_row + min(row, valid_rows - 1)) * width +
                         start + column
                   : weight + int64_t(min(first_column + row, outputs - 1)) * weight_stride +
                         start / 2 + column;
    unsigned address = __cvta_generic_to_shared(destination + index);
    if (reinterpret_cast<uintptr_t>(source) % 16 == 0) {
      int bytes = valid ? 16 : 0;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(source),
                   "r"(bytes)
                   : "memory");
    } else {
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        int bytes = valid ? 4 : 0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(address + part * 4),
                     "l"(source + part * 4), "r"(bytes)
                     : "memory");
      }
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <class Output, class ScaleLayout>
__global__ void
grouped_prefill_k32_kernel(const uint8_t *__restrict__ input,
                           const uint8_t *__restrict__ input_scales, const int64_t *weight_ptrs,
                           const int64_t *scale_ptrs, const int32_t *offsets, const int32_t *tiles,
                           Output *__restrict__ output, int groups, int rows, int outputs,
                           int width, int weight_stride, ScaleLayout layout) {
  __shared__ __align__(16) uint8_t raw[2 * prefill_raw_bytes];
  __shared__ __align__(16) uint16_t data[(prefill_rows + prefill_columns) * prefill_stage];
  __shared__ uint8_t sa[prefill_rows * 4], sb[prefill_columns * 4];
  auto *as = data;
  auto *bs = data + prefill_rows * prefill_stage;
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  int warp_row = (warp / (prefill_columns / 16)) * 64;
  int warp_column = warp % (prefill_columns / 16);
  if (offsets[groups] < 0) {
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < int64_t(rows) * outputs; index += int64_t(gridDim.x) * blockDim.x)
      output[index] = Output(fp8_scale(255));
    return;
  }
  int column_tiles = (outputs + prefill_columns - 1) / prefill_columns;
  int width_groups = width / 32;
  for (int task = blockIdx.x; task < tiles[groups]; task += gridDim.x) {
    int low = 0, high = groups;
    while (low + 1 < high) {
      int middle = (low + high) / 2;
      if (tiles[middle] <= task)
        low = middle;
      else
        high = middle;
    }
    int row_offset = offsets[low], group_rows = offsets[low + 1] - row_offset;
    int local_task = task - tiles[low];
    int first_row = local_task / column_tiles * prefill_rows;
    int first_column = local_task % column_tiles * prefill_columns;
    int valid_rows = min(prefill_rows, group_rows - first_row);
    const auto *weight = reinterpret_cast<const uint8_t *>(weight_ptrs[low]);
    const auto *weight_scales = reinterpret_cast<const uint8_t *>(scale_ptrs[low]);
    float total[4][2][4] = {};
    prefill_copy_raw(raw, input, weight, row_offset, first_row, first_column, valid_rows, outputs,
                     width, weight_stride, 0);
    for (int start = 0; start < width; start += prefill_stage) {
      int stage = start / prefill_stage, slot = stage & 1;
      if (start + prefill_stage < width)
        prefill_copy_raw(raw + (slot ^ 1) * prefill_raw_bytes, input, weight, row_offset, first_row,
                         first_column, valid_rows, outputs, width, weight_stride,
                         start + prefill_stage);
      for (int index = threadIdx.x; index < prefill_rows * 4; index += blockDim.x) {
        int row = index / 4, group = index % 4;
        sa[index] = row < valid_rows
                        ? input_scales[int64_t(row_offset + first_row + row) * width_groups +
                                       start / 32 + group]
                        : 127;
      }
      for (int index = threadIdx.x; index < prefill_columns * 4; index += blockDim.x) {
        int row = index / 4, group = index % 4;
        sb[index] =
            first_column + row < outputs
                ? weight_scales[layout(cute::make_coord(first_column + row, start + group * 32, 0))]
                : 127;
      }
      if (start + prefill_stage < width)
        asm volatile("cp.async.wait_group 1;" ::: "memory");
      else
        asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
      auto *loaded = raw + slot * prefill_raw_bytes;
      for (int index = threadIdx.x * 8; index < prefill_rows * prefill_stage;
           index += blockDim.x * 8) {
        int row = index / prefill_stage, column = index % prefill_stage;
        uint64_t packed = *reinterpret_cast<const uint64_t *>(loaded + index);
        uint4 converted = make_uint4(half_pair(packed), half_pair(packed >> 16),
                                     half_pair(packed >> 32), half_pair(packed >> 48));
        prefill_half_store(as, row, column, converted);
      }
      for (int index = threadIdx.x * 8; index < prefill_columns * prefill_stage;
           index += blockDim.x * 8) {
        int row = index / prefill_stage, column = index % prefill_stage;
        uint32_t packed = *reinterpret_cast<const uint32_t *>(loaded + prefill_raw_a + index / 2);
        uint2 low_values = fp4_half_four(packed), high_values = fp4_half_four(packed >> 16);
        prefill_half_store(bs, row, column,
                           make_uint4(low_values.x, low_values.y, high_values.x, high_values.y));
      }
      __syncthreads();
      for (int group = 0; group < 4; ++group) {
        float activation_scales[4][2], weight_scales_tile[2][2];
#pragma unroll
        for (int tile = 0; tile < 4; ++tile)
#pragma unroll
          for (int part = 0; part < 2; ++part) {
            int row = warp_row + tile * 16 + lane / 4 + 8 * part;
            activation_scales[tile][part] = fp8_scale(sa[row * 4 + group]);
          }
#pragma unroll
        for (int tile = 0; tile < 2; ++tile)
#pragma unroll
          for (int part = 0; part < 2; ++part) {
            int column = warp_column * 16 + tile * 8 + (lane & 3) * 2 + part;
            weight_scales_tile[tile][part] = fp8_scale(sb[column * 4 + group]);
          }
        float partial[4][2][4] = {};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          uint32_t a[4][4], b[2][2];
#pragma unroll
          for (int tile = 0; tile < 4; ++tile)
            half_load_a(a[tile], as, warp_row + tile * 16, group * 32 + half * 16, lane);
#pragma unroll
          for (int tile = 0; tile < 2; ++tile)
            half_load_b(b[tile], bs, warp_column * 16 + tile * 8, group * 32 + half * 16, lane);
#pragma unroll
          for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int n = 0; n < 2; ++n)
              half_mma(partial[m][n], a[m], b[n]);
        }
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
          for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int value = 0; value < 4; ++value)
              total[m][n][value] =
                  __fmaf_rn(__fmul_rn(partial[m][n][value], activation_scales[m][value >> 1]),
                            weight_scales_tile[n][value & 1], total[m][n][value]);
      }
      __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
      for (int n = 0; n < 2; ++n)
#pragma unroll
        for (int value = 0; value < 4; ++value) {
          int row = first_row + warp_row + m * 16 + lane / 4 + 8 * (value >> 1);
          int column = first_column + warp_column * 16 + n * 8 + (lane & 3) * 2 + (value & 1);
          if (row < group_rows && column < outputs) {
            if constexpr (std::is_same_v<Output, float>)
              output[int64_t(row_offset + row) * outputs + column] = total[m][n][value];
            else
              output[int64_t(row_offset + row) * outputs + column] =
                  __float2bfloat16_rn(total[m][n][value]);
          }
        }
    __syncthreads();
  }
}

template <class Output, class ScaleLayout>
void launch_grouped_prefill_k32(const uint8_t *input, const uint8_t *input_scales,
                                const int64_t *weight_ptrs, const int64_t *scale_ptrs,
                                const int32_t *offsets, const int32_t *tiles, Output *output,
                                int groups, int rows, int outputs, int width, int weight_stride,
                                ScaleLayout layout, cudaStream_t stream) {
  int bound = ((rows + prefill_rows - 1) / prefill_rows + groups) *
              ((outputs + prefill_columns - 1) / prefill_columns);
  grouped_prefill_k32_kernel<Output, ScaleLayout><<<min(bound, 148), 512, 0, stream>>>(
      input, input_scales, weight_ptrs, scale_ptrs, offsets, tiles, output, groups, rows, outputs,
      width, weight_stride, layout);
}

} // namespace encoded_projection
