#pragma once

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include "fp4_k32.cuh"

namespace encoded_projection::four_row_chunks {

constexpr bool matches_shape(int groups, int rows, int outputs, int width) {
  return groups == 384 && rows == 24 && outputs == 576 && width == 5120;
}

__global__ void compact_offsets(const int32_t *counts, const int64_t *weight_ptrs,
                                const int64_t *scale_ptrs, int groups, int bound, int rows,
                                int max_rows, int32_t *compact_counts, int64_t *compact_weights,
                                int64_t *compact_scales, int32_t *offsets, int32_t *tiles,
                                int32_t *status) {
  using Scan = cub::BlockScan<int, 128>;
  using Reduce = cub::BlockReduce<long long, 128>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ typename Reduce::TempStorage reduce;
  __shared__ int active;
  int tid = threadIdx.x;
  if (tid == 0)
    active = 0;
  __syncthreads();
  long long original_sum = 0;
  bool invalid = false;
  for (int base = 0; base < groups; base += 128) {
    int group = base + tid;
    int count = group < groups ? counts[group] : 0;
    original_sum += count;
    invalid |= count < 0 || count > max_rows;
    int flag = count > 0, position;
    int previous = active;
    Scan(scan).ExclusiveSum(flag, position);
    if (flag && previous + position < bound) {
      int target = previous + position;
      compact_counts[target] = count;
      compact_weights[target] = weight_ptrs[group];
      compact_scales[target] = scale_ptrs[group];
    }
    if (tid == 127)
      active = previous + position + flag;
    __syncthreads();
  }
  for (int index = active + tid; index < bound; index += 128) {
    compact_counts[index] = 0;
    compact_weights[index] = 0;
    compact_scales[index] = 0;
  }
  bool valid = !__syncthreads_or(invalid);
  long long sum = Reduce(reduce).Sum(original_sum);
  if (tid == 0) {
    int64_t compact_sum = 0, tile_sum = 0;
    for (int group = 0; group < bound; ++group) {
      offsets[group] = static_cast<int32_t>(compact_sum);
      tiles[group] = tile_sum;
      int count = compact_counts[group];
      compact_sum += count;
      tile_sum += (count > 0 && count <= max_rows ? (int64_t(count) + 3) / 4 : 0);
    }
    valid = valid && compact_sum == sum && (status ? sum <= rows : sum == rows);
    if (status && !valid)
      atomicOr(status, 1);
    valid = valid && (!status || *status == 0);
    offsets[bound] = valid ? static_cast<int32_t>(sum) : -1;
    tiles[bound] = tile_sum;
  }
}

struct PrefetchedK32 {
  uint32_t a0[4], a1[4];
  uint16_t b0[4], b1[4];
  uint8_t weight_scale[4], activation_scale[4];
};

template <class Layout>
__device__ __forceinline__ void
prefetch(PrefetchedK32 &data, const uint8_t *input, const uint8_t *input_scales,
         const uint8_t *weight, const uint8_t *weight_scales, int first_row, int end, int width,
         int column, int weight_stride, int first_group, int lane, Layout layout) {
  int width_groups = width / 32, k = (lane & 3) * 4, n = lane >> 2;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int group = first_group + i * 8;
    if (group < width_groups) {
      int row = first_row + n;
      data.a0[i] = data.a1[i] = 0;
      if (n < 4 && row < end) {
        data.a0[i] =
            *reinterpret_cast<const uint32_t *>(input + int64_t(row) * width + group * 32 + k);
        data.a1[i] =
            *reinterpret_cast<const uint32_t *>(input + int64_t(row) * width + group * 32 + k + 16);
      }
      const auto *source = weight + int64_t(column + n) * weight_stride + group * 16 + k / 2;
      data.b0[i] = *reinterpret_cast<const uint16_t *>(source);
      data.b1[i] = *reinterpret_cast<const uint16_t *>(source + 8);
      if (lane < 8)
        data.weight_scale[i] =
            weight_scales[layout(cute::make_coord(column + lane, group * 32, 0))];
      if (lane < 4)
        data.activation_scale[i] =
            first_row + lane < end ? input_scales[int64_t(first_row + lane) * width_groups + group]
                                   : 127;
    }
  }
}

template <class Output, class Layout>
__global__ void
rawdots_preload4(const uint8_t *__restrict__ input, const uint8_t *__restrict__ input_scales,
                 const int64_t *weight_ptrs, const int64_t *weight_scale_ptrs,
                 const int32_t *offsets, const int32_t *chunk_offsets, Output *output, int groups,
                 int rows, int outputs, int width, int weight_stride, Layout layout) {
  using Mma = cute::SM89_16x8x32_F32E4M3E4M3F32_TN;
  extern __shared__ __align__(16) uint8_t storage[];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int column = blockIdx.x * 8, chunk = blockIdx.y, width_groups = width / 32;
  auto *raw_dots = reinterpret_cast<float (*)[4][8]>(storage);
  auto *scales = reinterpret_cast<float (*)[8]>(raw_dots + width_groups);
  auto *activation_scales = reinterpret_cast<float (*)[4]>(scales + width_groups);
  if (offsets[groups] < 0) {
    if (chunk == 0 && threadIdx.x < 8)
      for (int row = 0; row < rows; ++row)
        output[int64_t(row) * outputs + column + threadIdx.x] = Output(fp8_scale(255));
    return;
  }
  if (chunk >= chunk_offsets[groups])
    return;
  int low = 0, high = groups;
  while (low < high) {
    int middle = (low + high) / 2;
    if (chunk_offsets[middle + 1] <= chunk)
      low = middle + 1;
    else
      high = middle;
  }
  int expert = low, first = offsets[expert], end = offsets[expert + 1];
  const auto *weight = reinterpret_cast<const uint8_t *>(weight_ptrs[expert]);
  const auto *weight_scales = reinterpret_cast<const uint8_t *>(weight_scale_ptrs[expert]);
  {
    int row = first + 4 * (chunk - chunk_offsets[expert]);
    for (int base = warp; base < width_groups; base += 32) {
      PrefetchedK32 data;
      prefetch(data, input, input_scales, weight, weight_scales, row, end, width, column,
               weight_stride, base, lane, layout);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int group = base + i * 8;
        if (group < width_groups) {
          float raw[4];
          uint32_t b0 = fp4_to_fp8_four(data.b0[i]), b1 = fp4_to_fp8_four(data.b1[i]);
          Mma::fma(raw[0], raw[1], raw[2], raw[3], data.a0[i], 0u, data.a1[i], 0u, b0, b1, 0.f, 0.f,
                   0.f, 0.f);
          if (lane < 16) {
            raw_dots[group][lane >> 2][(lane & 3) * 2] = raw[0];
            raw_dots[group][lane >> 2][(lane & 3) * 2 + 1] = raw[1];
          }
          if (lane < 8)
            scales[group][lane] = fp8_scale(data.weight_scale[i]);
          if (lane < 4)
            activation_scales[group][lane] = fp8_scale(data.activation_scale[i]);
        }
      }
    }
    __syncthreads();
    int local_row = threadIdx.x / 8, local_column = threadIdx.x & 7;
    if (threadIdx.x < 32 && row + local_row < end) {
      float total = 0.f;
      for (int group = 0; group < width_groups; ++group)
        total = __fmaf_rn(__fmul_rn(raw_dots[group][local_row][local_column],
                                    activation_scales[group][local_row]),
                          scales[group][local_column], total);
      if constexpr (std::is_same_v<Output, float>)
        output[int64_t(row + local_row) * outputs + column + local_column] = total;
      else
        output[int64_t(row + local_row) * outputs + column + local_column] =
            __float2bfloat16_rn(total);
    }
    __syncthreads();
  }
}

template <class Output, class Layout>
void launch(const uint8_t *input, const uint8_t *input_scales, const int64_t *weight_ptrs,
            const int64_t *weight_scale_ptrs, const int32_t *counts, uint8_t *scratch,
            Output *output, int groups, int rows, int outputs, int width, int max_rows,
            Layout layout, cudaStream_t stream, int32_t *status) {
  int bound = groups < rows ? groups : rows;
  int chunk_bound = min(int64_t(rows), (int64_t(rows) + 3 * int64_t(bound)) / 4);
  int grid_chunks = chunk_bound > 0 ? chunk_bound : 1;
  int64_t offsets_bytes = ((int64_t(bound) + 1) * 4 + 15) / 16 * 16;
  int64_t counts_bytes = (int64_t(bound) * 4 + 15) / 16 * 16;
  int64_t pointers_bytes = (int64_t(bound) * 8 + 15) / 16 * 16;
  auto *offsets = reinterpret_cast<int32_t *>(scratch);
  auto *compact_counts = reinterpret_cast<int32_t *>(scratch + offsets_bytes);
  auto *compact_weight_ptrs = reinterpret_cast<int64_t *>(scratch + offsets_bytes + counts_bytes);
  auto *compact_scale_ptrs =
      reinterpret_cast<int64_t *>(scratch + offsets_bytes + counts_bytes + pointers_bytes);
  auto *tiles =
      reinterpret_cast<int32_t *>(scratch + offsets_bytes + counts_bytes + 2 * pointers_bytes);
  compact_offsets<<<1, 128, 0, stream>>>(counts, weight_ptrs, weight_scale_ptrs, groups, bound,
                                         rows, max_rows, compact_counts, compact_weight_ptrs,
                                         compact_scale_ptrs, offsets, tiles, status);
  if (!rows)
    return;
  rawdots_preload4<Output, Layout>
      <<<dim3(outputs / 8, grid_chunks), 256, (width / 32) * (4 * 8 + 8 + 4) * sizeof(float),
         stream>>>(input, input_scales, compact_weight_ptrs, compact_scale_ptrs, offsets, tiles,
                   output, bound, rows, outputs, width, (width + 127) / 128 * 64, layout);
  if (status) {
    int blocks = static_cast<int>(min((int64_t(rows) * outputs + 255) / 256, int64_t(148 * 8)));
    fp4_clear_tail<<<blocks, 256, 0, stream>>>(output, offsets, bound, rows, outputs);
  }
  return;
}

} // namespace encoded_projection::four_row_chunks
