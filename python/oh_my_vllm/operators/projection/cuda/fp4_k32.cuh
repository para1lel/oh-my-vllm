#pragma once

#include "fp8_k32.cuh"
#include "grouped_prefill_k32.cuh"

namespace encoded_projection {

__device__ __forceinline__ uint32_t fp4_to_fp8_four(uint16_t packed) {
  // E2M1 magnitudes have exact E4M3 codes. Move their sign bits separately.
  uint32_t values = __byte_perm(0x3c383000u, 0x4c484440u, packed & 0x7777u);
  uint32_t signs = __byte_perm(packed, uint32_t(packed) << 4, 0x1504u) & 0x80808080u;
  return values | signs;
}

template <int TileRows>
__global__ void fp4_group_offsets(const int32_t *counts, int groups, const int32_t *compact_counts,
                                  int32_t *offsets, int32_t *tiles, int bound, int rows,
                                  int max_rows, int column_tiles, int32_t *status) {
  int64_t sum = 0;
  bool valid = true;
  for (int group = 0; group < groups; ++group) {
    int count = counts[group];
    valid &= count >= 0 && count <= max_rows;
    sum += count;
  }
  int64_t compact_sum = 0;
  int64_t tile_sum = 0;
  for (int group = 0; group < bound; ++group) {
    offsets[group] = static_cast<int32_t>(compact_sum);
    tiles[group] = tile_sum;
    int count = compact_counts[group];
    compact_sum += count;
    tile_sum +=
        (count > 0 && count <= max_rows ? (count + TileRows - 1) / TileRows : 0) * column_tiles;
  }
  valid = valid && compact_sum == sum && (status ? sum <= rows : sum == rows);
  if (status && !valid)
    atomicOr(status, 1);
  valid = valid && (!status || *status == 0);
  offsets[bound] = valid ? static_cast<int32_t>(sum) : -1;
  tiles[bound] = tile_sum;
}

template <class Output>
__global__ void fp4_clear_tail(Output *output, const int32_t *offsets, int groups, int capacity,
                               int outputs) {
  int live_rows = offsets[groups];
  if (live_rows < 0)
    live_rows = 0;
  for (int64_t index = int64_t(live_rows) * outputs + blockIdx.x * blockDim.x + threadIdx.x;
       index < int64_t(capacity) * outputs; index += int64_t(gridDim.x) * blockDim.x)
    output[index] = Output(0.f);
}

template <class Output>
__device__ __forceinline__ void fp4_invalid_group_output(Output *output, int rows, int outputs,
                                                         int first_column, int warp, int lane) {
  for (int first_row = 0; first_row < rows; first_row += 16) {
    int row = first_row + lane / 4;
#pragma unroll
    for (int value = 0; value < 4; ++value) {
      int r = row + 8 * (value >> 1);
      int column = first_column + warp * 8 + (lane & 3) * 2 + (value & 1);
      if (r < rows)
        output[int64_t(r) * outputs + column] = Output(fp8_scale(255));
    }
  }
}

template <class ScaleLayout>
__device__ __forceinline__ void
fp4_load_stage(const uint8_t *input, const uint8_t *weight, const uint8_t *input_scales,
               const uint8_t *weight_scales, uint8_t *a_shared, uint8_t *b_shared, uint8_t *sa,
               uint8_t *sb, int valid_rows, int columns, int row_offset, int first_row,
               int first_column, int width, int weight_stride, int width_groups, int start,
               int stage_groups, ScaleLayout layout) {
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
}

__device__ __forceinline__ void fp4_accumulate_stage(const uint8_t *a_shared,
                                                     const uint8_t *b_shared, const uint8_t *sa,
                                                     const uint8_t *sb, float (&total)[4],
                                                     int stage_groups, int valid_rows, int low_row,
                                                     int first_k, int warp, int lane) {
  using Mma = cute::SM89_16x8x32_F32E4M3E4M3F32_TN;
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
}

template <class Output, class ScaleLayout, bool Grouped>
__global__ void fp4_k32_kernel(const uint8_t *__restrict__ input, const uint8_t *weight,
                               const uint8_t *__restrict__ input_scales,
                               const uint8_t *weight_scales, Output *__restrict__ output,
                               const int64_t *weight_ptrs, const int64_t *weight_scale_ptrs,
                               const int32_t *offsets, int groups, int rows, int outputs, int width,
                               int weight_stride, ScaleLayout layout) {
  static_assert(std::is_same_v<Output, float> || std::is_same_v<Output, __nv_bfloat16>);
  using Mma = cute::SM89_16x8x32_F32E4M3E4M3F32_TN;
  __shared__ __align__(16) uint8_t data[(16 + 32) * fp8_tile_width];
  __shared__ uint8_t sa[16 * fp8_tile_groups], sb[32 * fp8_tile_groups];
  auto *a_shared = data;
  auto *b_shared = data + 16 * fp8_tile_width;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int first_column = Grouped ? blockIdx.x * 32 : (blockIdx.x % (outputs / 32)) * 32;
  int group_rows = rows, row_offset = 0;
  if constexpr (Grouped) {
    int group = blockIdx.y;
    if (offsets[groups] < 0) {
      // Invalid device metadata cannot address any input or weight storage.
      // One group fills the complete output, so the grid does not read a row bound.
      if (group != 0)
        return;
      fp4_invalid_group_output(output, rows, outputs, first_column, warp, lane);
      return;
    }
    row_offset = offsets[group];
    group_rows = offsets[group + 1] - row_offset;
    weight = reinterpret_cast<const uint8_t *>(weight_ptrs[group]);
    weight_scales = reinterpret_cast<const uint8_t *>(weight_scale_ptrs[group]);
  }
  // Every block walks the rows of its group. The launch grid covers the column
  // tiles and the groups only, so no host row bound selects its blocks.
  int row_tiles = Grouped ? (group_rows + 15) / 16 : 1;
  int first_row_base = Grouped ? 0 : (blockIdx.x / (outputs / 32)) * 16;
  for (int tile = 0; tile < row_tiles; ++tile) {
    int first_row = first_row_base + tile * 16;
    int valid_rows = min(16, group_rows - first_row), width_groups = width / 32;
    int low_row = lane >> 2, first_k = (lane & 3) * 4;
    float total[4] = {};
    for (int start = 0; start < width; start += fp8_tile_width) {
      int columns = min(fp8_tile_width, width - start), stage_groups = columns / 32;
      fp4_load_stage(input, weight, input_scales, weight_scales, a_shared, b_shared, sa, sb,
                     valid_rows, columns, row_offset, first_row, first_column, width, weight_stride,
                     width_groups, start, stage_groups, layout);
      __syncthreads();
      fp4_accumulate_stage(a_shared, b_shared, sa, sb, total, stage_groups, valid_rows, low_row,
                           first_k, warp, lane);
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
}

template <class Output, class ScaleLayout>
__global__ void fp4_decode_rawdots(const uint8_t *__restrict__ input,
                                   const uint8_t *__restrict__ input_scales,
                                   const int64_t *weight_ptrs, const int64_t *weight_scale_ptrs,
                                   const int32_t *offsets, Output *output, int groups, int rows,
                                   int outputs, int width, int weight_stride, ScaleLayout layout) {
  using Mma = cute::SM89_16x8x32_F32E4M3E4M3F32_TN;
  __shared__ float raw_dots[512][8], scales[512][8], activation_scales[512];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int column = blockIdx.x * 8, expert = blockIdx.y, width_groups = width / 32;
  if (offsets[groups] < 0) {
    if (expert == 0 && threadIdx.x < 8)
      for (int row = 0; row < rows; ++row)
        output[int64_t(row) * outputs + column + threadIdx.x] = Output(fp8_scale(255));
    return;
  }
  int first = offsets[expert], end = offsets[expert + 1];
  if (first == end)
    return;
  const auto *weight = reinterpret_cast<const uint8_t *>(weight_ptrs[expert]);
  const auto *weight_scales = reinterpret_cast<const uint8_t *>(weight_scale_ptrs[expert]);
  int k = (lane & 3) * 4, n = lane >> 2;
  for (int row = first; row < end; ++row) {
    for (int group = warp; group < width_groups; group += 8) {
      uint32_t a0 =
          *reinterpret_cast<const uint32_t *>(input + int64_t(row) * width + group * 32 + k);
      uint32_t a1 =
          *reinterpret_cast<const uint32_t *>(input + int64_t(row) * width + group * 32 + k + 16);
      const auto *source = weight + int64_t(column + n) * weight_stride + group * 16 + k / 2;
      uint32_t b0 = fp4_to_fp8_four(*reinterpret_cast<const uint16_t *>(source));
      uint32_t b1 = fp4_to_fp8_four(*reinterpret_cast<const uint16_t *>(source + 8));
      float raw[4];
      Mma::fma(raw[0], raw[1], raw[2], raw[3], a0, 0u, a1, 0u, b0, b1, 0.f, 0.f, 0.f, 0.f);
      if (lane < 4) {
        raw_dots[group][lane * 2] = raw[0];
        raw_dots[group][lane * 2 + 1] = raw[1];
      }
      if (lane < 8)
        scales[group][lane] =
            fp8_scale(weight_scales[layout(cute::make_coord(column + lane, group * 32, 0))]);
      if (lane == 0)
        activation_scales[group] = fp8_scale(input_scales[int64_t(row) * width_groups + group]);
    }
    __syncthreads();
    if (threadIdx.x < 8) {
      float total = 0.f;
      for (int group = 0; group < width_groups; ++group)
        total = __fmaf_rn(__fmul_rn(raw_dots[group][threadIdx.x], activation_scales[group]),
                          scales[group][threadIdx.x], total);
      if constexpr (std::is_same_v<Output, float>)
        output[int64_t(row) * outputs + column + threadIdx.x] = total;
      else
        output[int64_t(row) * outputs + column + threadIdx.x] = __float2bfloat16_rn(total);
    }
    __syncthreads();
  }
}

template <class Output, class ScaleLayout>
void launch_fp4_k32(const uint8_t *input, const uint8_t *weight, const uint8_t *input_scales,
                    const uint8_t *weight_scales, Output *output, int rows, int outputs, int width,
                    int weight_stride, ScaleLayout layout, cudaStream_t stream) {
  int tiles = (outputs / 32) * ((rows + 15) / 16);
  fp4_k32_kernel<Output, ScaleLayout, false>
      <<<tiles, 128, 0, stream>>>(input, weight, input_scales, weight_scales, output, nullptr,
                                  nullptr, nullptr, 1, rows, outputs, width, weight_stride, layout);
}

/* Copy the nonempty expert groups to the front of the scratch arrays.
 *
 * The grouped launch covers `bound` groups, where `bound` is the smaller of the
 * input group count and the input rows: every nonempty group owns at least one
 * row. Padding entries have a zero count, so they cost one offset read and no
 * weight read. The offsets kernel validates the original counts, so a dropped
 * group cannot hide invalid metadata. */
__global__ void fp4_group_compact(const int32_t *counts, const int64_t *weight_ptrs,
                                  const int64_t *weight_scale_ptrs, int groups, int bound,
                                  int32_t *compact_counts, int64_t *compact_weight_ptrs,
                                  int64_t *compact_scale_ptrs) {
  __shared__ int scan[128];
  int tid = threadIdx.x;
  int active = 0;
  for (int base = 0; base < groups; base += blockDim.x) {
    int group = base + tid;
    int count = group < groups ? counts[group] : 0;
    int flag = count > 0 ? 1 : 0;
    scan[tid] = flag;
    __syncthreads();
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
      int value = tid >= offset ? scan[tid - offset] : 0;
      __syncthreads();
      scan[tid] += value;
      __syncthreads();
    }
    int position = active + (tid == 0 ? 0 : scan[tid - 1]);
    if (flag && position < bound) {
      compact_counts[position] = count;
      compact_weight_ptrs[position] = weight_ptrs[group];
      compact_scale_ptrs[position] = weight_scale_ptrs[group];
    }
    active += scan[blockDim.x - 1];
    __syncthreads();
  }
  for (int index = active + tid; index < bound; index += blockDim.x) {
    compact_counts[index] = 0;
    compact_weight_ptrs[index] = 0;
    compact_scale_ptrs[index] = 0;
  }
}

/* Scratch layout: row offsets, compacted counts, compacted weight pointers,
 * compacted scale pointers, and tile offsets, each region rounded to 16 bytes. */
inline int64_t grouped_scratch_bytes(int bound) {
  int64_t offsets = ((int64_t(bound) + 1) * 4 + 15) / 16 * 16;
  int64_t counts = (int64_t(bound) * 4 + 15) / 16 * 16;
  int64_t pointers = (int64_t(bound) * 8 + 15) / 16 * 16;
  return offsets + counts + pointers + pointers + offsets;
}

template <class Output, class ScaleLayout>
void launch_grouped_fp4_k32(const uint8_t *input, const uint8_t *input_scales,
                            const int64_t *weight_ptrs, const int64_t *weight_scale_ptrs,
                            const int32_t *counts, uint8_t *scratch, Output *output, int groups,
                            int rows, int outputs, int width, int max_rows, int weight_stride,
                            ScaleLayout layout, cudaStream_t stream, int32_t *status) {
  int bound = groups < rows ? groups : rows;
  int64_t offsets_bytes = ((int64_t(bound) + 1) * 4 + 15) / 16 * 16;
  int64_t counts_bytes = (int64_t(bound) * 4 + 15) / 16 * 16;
  int64_t pointers_bytes = (int64_t(bound) * 8 + 15) / 16 * 16;
  auto *offsets = reinterpret_cast<int32_t *>(scratch);
  auto *compact_counts = reinterpret_cast<int32_t *>(scratch + offsets_bytes);
  auto *compact_weight_ptrs = reinterpret_cast<int64_t *>(scratch + offsets_bytes + counts_bytes);
  auto *compact_scale_ptrs =
      reinterpret_cast<int64_t *>(scratch + offsets_bytes + counts_bytes + pointers_bytes);
  auto *tile_offsets =
      reinterpret_cast<int32_t *>(scratch + offsets_bytes + counts_bytes + 2 * pointers_bytes);
  fp4_group_compact<<<1, 128, 0, stream>>>(counts, weight_ptrs, weight_scale_ptrs, groups, bound,
                                           compact_counts, compact_weight_ptrs, compact_scale_ptrs);
  bool prefill = rows >= 4096 && outputs == 2304 && width == 5120 &&
                 reinterpret_cast<uintptr_t>(input) % 16 == 0;
  bool small_tile = rows > 64 && rows <= groups && !prefill;
  if (small_tile)
    fp4_group_offsets<16><<<1, 1, 0, stream>>>(counts, groups, compact_counts, offsets,
                                               tile_offsets, bound, rows, max_rows,
                                               (outputs + 63) / 64, status);
  else if (prefill)
    fp4_group_offsets<prefill_rows>
        <<<1, 1, 0, stream>>>(counts, groups, compact_counts, offsets, tile_offsets, bound, rows,
                              max_rows, (outputs + prefill_columns - 1) / prefill_columns, status);
  else
    fp4_group_offsets<32><<<1, 1, 0, stream>>>(counts, groups, compact_counts, offsets,
                                               tile_offsets, bound, rows, max_rows,
                                               (outputs + 63) / 64, status);
  if (!rows)
    return;
  if (rows <= 64)
    fp4_decode_rawdots<Output, ScaleLayout><<<dim3(outputs / 8, bound), 256, 0, stream>>>(
        input, input_scales, compact_weight_ptrs, compact_scale_ptrs, offsets, output, bound, rows,
        outputs, width, weight_stride, layout);
  else if (prefill)
    launch_grouped_prefill_k32(input, input_scales, compact_weight_ptrs, compact_scale_ptrs,
                               offsets, tile_offsets, output, bound, rows, outputs, width,
                               weight_stride, layout, stream);
  else if (small_tile)
    launch_half_k32<Output, ScaleLayout, 4, true, 16>(
        input, nullptr, input_scales, nullptr, output, compact_weight_ptrs, compact_scale_ptrs,
        offsets, tile_offsets, bound, rows, outputs, width, weight_stride, layout, stream);
  else
    launch_half_k32<Output, ScaleLayout, 4, true>(
        input, nullptr, input_scales, nullptr, output, compact_weight_ptrs, compact_scale_ptrs,
        offsets, tile_offsets, bound, rows, outputs, width, weight_stride, layout, stream);
  if (status) {
    int blocks = static_cast<int>(min((int64_t(rows) * outputs + 255) / 256, int64_t(148 * 8)));
    fp4_clear_tail<<<blocks, 256, 0, stream>>>(output, offsets, bound, rows, outputs);
  }
}

} // namespace encoded_projection
