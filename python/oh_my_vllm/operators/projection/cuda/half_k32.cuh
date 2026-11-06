#pragma once

#include <cuda_fp8.h>
#include <cute/arch/mma_sm80.hpp>

namespace encoded_projection {

constexpr int half_stage = 128, half_rows = 32, half_columns = 64;

__device__ __forceinline__ int half_address(int row, int column) {
  return row * half_stage + (column ^ ((row & 7) * 8));
}

__device__ __forceinline__ uint32_t half_pair(uint16_t values) {
  auto converted = __nv_cvt_fp8x2_to_halfraw2(values, __NV_E4M3);
  return *reinterpret_cast<uint32_t *>(&converted);
}

__device__ __forceinline__ void half_store(uint16_t *destination, int row, int column,
                                           uint4 values) {
  // CUDA 13.4 lowers FP8 MMA to the low two bytes of each K4, then the high two.
  int reordered = (column / 32) * 32 + (column % 32) / 2;
  *reinterpret_cast<uint64_t *>(destination + half_address(row, reordered)) =
      uint64_t(values.x) | (uint64_t(values.z) << 32);
  *reinterpret_cast<uint64_t *>(destination + half_address(row, reordered + 16)) =
      uint64_t(values.y) | (uint64_t(values.w) << 32);
}

__device__ __forceinline__ uint2 fp4_half_four(uint16_t packed) {
  uint32_t high = __byte_perm(0x3e3c3800u, 0x46444240u, packed & 0x7777u);
  uint32_t signs = __byte_perm(packed, uint32_t(packed) << 4, 0x1504u) & 0x80808080u;
  high |= signs;
  return make_uint2(__byte_perm(high, 0u, 0x1404u), __byte_perm(high, 0u, 0x3424u));
}

__device__ __forceinline__ void half_load_a(uint32_t (&values)[4], const uint16_t *data, int row,
                                            int column, int lane) {
  unsigned address =
      __cvta_generic_to_shared(data + half_address(row + lane % 16, column + (lane / 16) * 8));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
               : "r"(address));
}

__device__ __forceinline__ void half_load_b(uint32_t (&values)[2], const uint16_t *data, int row,
                                            int column, int lane) {
  unsigned address =
      __cvta_generic_to_shared(data + half_address(row + lane % 8, column + ((lane / 8) % 2) * 8));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(values[0]), "=r"(values[1])
               : "r"(address));
}

__device__ __forceinline__ void half_mma(float (&raw)[4], const uint32_t (&a)[4],
                                         const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(raw[0]), "+f"(raw[1]), "+f"(raw[2]), "+f"(raw[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct HalfFp4Prefetch {
  uint32_t weights[8];
  uint32_t scales[2];
};

template <class ScaleLayout>
__device__ __forceinline__ void half_prefetch_fp4(HalfFp4Prefetch &next, const uint8_t *weight,
                                                  const uint8_t *scales, int first_column,
                                                  int outputs, int width, int weight_stride,
                                                  int start, ScaleLayout layout) {
  int columns = min(half_stage, width - start);
#pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    int index = threadIdx.x * 8 + slot * 128 * 8;
    int row = index / half_stage, column = index % half_stage;
    uint32_t packed = 0;
    if (first_column + row < outputs && column < columns) {
      auto *address = weight + int64_t(first_column + row) * weight_stride + (start + column) / 2;
      // Keep the immutable load before the current stage's MMA, rather than its next use.
      asm volatile("ld.global.u32 %0, [%1];" : "=r"(packed) : "l"(address) : "memory");
    }
    next.weights[slot] = packed;
  }
#pragma unroll
  for (int slot = 0; slot < 2; ++slot) {
    int index = threadIdx.x + slot * 128;
    int row = index / 4, group = index % 4;
    uint32_t scale = 127;
    if (first_column + row < outputs && group < columns / 32) {
      auto *address = scales + layout(cute::make_coord(first_column + row, start + group * 32, 0));
      asm volatile("ld.global.u8 %0, [%1];" : "=r"(scale) : "l"(address) : "memory");
    }
    next.scales[slot] = scale;
  }
}

__device__ __forceinline__ void half_store_prefetched_fp4(const HalfFp4Prefetch &next,
                                                          uint16_t *data) {
#pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    int index = threadIdx.x * 8 + slot * 128 * 8;
    int row = index / half_stage, column = index % half_stage;
    uint32_t packed = next.weights[slot];
    uint2 low = fp4_half_four(packed), high = fp4_half_four(packed >> 16);
    half_store(data, row, column, make_uint4(low.x, low.y, high.x, high.y));
  }
}

template <int Bits, bool Prefetch, int TileRows, class ScaleLayout>
__device__ __forceinline__ void
half_load_stage(const uint8_t *input, const uint8_t *weight, const uint8_t *input_scales,
                const uint8_t *weight_scales, uint16_t *as, uint16_t *bs, uint8_t *sa, uint8_t *sb,
                const HalfFp4Prefetch &prefetched, int valid_rows, int columns, int row_offset,
                int first_row, int first_column, int outputs, int width, int weight_stride,
                int width_groups, int start, int stage_groups, ScaleLayout layout) {
  for (int index = threadIdx.x * 8; index < TileRows * half_stage; index += blockDim.x * 8) {
    int row = index / half_stage, column = index % half_stage;
    uint64_t packed =
        row < valid_rows && column < columns
            ? *reinterpret_cast<const uint64_t *>(
                  input + int64_t(row_offset + first_row + row) * width + start + column)
            : 0;
    uint4 converted;
    converted.x = half_pair(packed);
    converted.y = half_pair(packed >> 16);
    converted.z = half_pair(packed >> 32);
    converted.w = half_pair(packed >> 48);
    half_store(as, row, column, converted);
  }
  if constexpr (Prefetch) {
    half_store_prefetched_fp4(prefetched, bs);
  } else {
    for (int index = threadIdx.x * 8; index < half_columns * half_stage; index += blockDim.x * 8) {
      int row = index / half_stage, column = index % half_stage;
      bool valid = first_column + row < outputs && column < columns;
      uint4 converted;
      if constexpr (Bits == 8) {
        uint64_t packed =
            valid ? *reinterpret_cast<const uint64_t *>(
                        weight + int64_t(first_column + row) * weight_stride + start + column)
                  : 0;
        converted.x = half_pair(packed);
        converted.y = half_pair(packed >> 16);
        converted.z = half_pair(packed >> 32);
        converted.w = half_pair(packed >> 48);
      } else {
        uint32_t packed =
            valid ? *reinterpret_cast<const uint32_t *>(
                        weight + int64_t(first_column + row) * weight_stride + (start + column) / 2)
                  : 0;
        uint2 low = fp4_half_four(packed), high = fp4_half_four(packed >> 16);
        converted = make_uint4(low.x, low.y, high.x, high.y);
      }
      half_store(bs, row, column, converted);
    }
  }
  for (int index = threadIdx.x; index < TileRows * 4; index += blockDim.x) {
    int row = index / 4, group = index % 4;
    sa[index] = row < valid_rows && group < stage_groups
                    ? input_scales[int64_t(row_offset + first_row + row) * width_groups +
                                   start / 32 + group]
                    : 127;
  }
  if constexpr (Prefetch) {
#pragma unroll
    for (int slot = 0; slot < 2; ++slot)
      sb[threadIdx.x + slot * 128] = static_cast<uint8_t>(prefetched.scales[slot]);
  } else {
    for (int index = threadIdx.x; index < half_columns * 4; index += blockDim.x) {
      int row = index / 4, group = index % 4;
      sb[index] =
          first_column + row < outputs && group < stage_groups
              ? weight_scales[layout(cute::make_coord(first_column + row, start + group * 32, 0))]
              : 127;
    }
  }
}

template <int RowTiles>
__device__ __forceinline__ void
half_accumulate_stage(const uint16_t *as, const uint16_t *bs, const uint8_t *sa, const uint8_t *sb,
                      float (&total)[RowTiles][2][4], int stage_groups, int warp, int lane) {
  for (int group = 0; group < stage_groups; ++group) {
    float activation_scales[RowTiles][2], weight_scales_tile[2][2];
#pragma unroll
    for (int tile = 0; tile < 2; ++tile)
#pragma unroll
      for (int part = 0; part < 2; ++part) {
        int row = tile * 16 + lane / 4 + 8 * part;
        int column = warp * 16 + tile * 8 + (lane & 3) * 2 + part;
        if (tile < RowTiles)
          activation_scales[tile][part] = fp8_scale(sa[row * 4 + group]);
        weight_scales_tile[tile][part] = fp8_scale(sb[column * 4 + group]);
      }
    float raw[RowTiles][2][4] = {};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      uint32_t a[RowTiles][4], b[2][2];
#pragma unroll
      for (int tile = 0; tile < 2; ++tile) {
        if (tile < RowTiles)
          half_load_a(a[tile], as, tile * 16, group * 32 + half * 16, lane);
        half_load_b(b[tile], bs, warp * 16 + tile * 8, group * 32 + half * 16, lane);
      }
#pragma unroll
      for (int m = 0; m < RowTiles; ++m)
#pragma unroll
        for (int n = 0; n < 2; ++n)
          half_mma(raw[m][n], a[m], b[n]);
    }
#pragma unroll
    for (int m = 0; m < RowTiles; ++m)
#pragma unroll
      for (int n = 0; n < 2; ++n)
#pragma unroll
        for (int value = 0; value < 4; ++value)
          total[m][n][value] =
              __fmaf_rn(__fmul_rn(raw[m][n][value], activation_scales[m][value >> 1]),
                        weight_scales_tile[n][value & 1], total[m][n][value]);
  }
}

template <class Output, int RowTiles>
__device__ __forceinline__ void half_store_output(Output *output,
                                                  const float (&total)[RowTiles][2][4],
                                                  int first_row, int first_column, int row_offset,
                                                  int group_rows, int outputs, int warp, int lane) {
#pragma unroll
  for (int m = 0; m < RowTiles; ++m)
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        int row = first_row + m * 16 + lane / 4 + 8 * (value >> 1);
        int column = first_column + warp * 16 + n * 8 + (lane & 3) * 2 + (value & 1);
        if (row < group_rows && column < outputs) {
          if constexpr (std::is_same_v<Output, float>)
            output[int64_t(row_offset + row) * outputs + column] = total[m][n][value];
          else
            output[int64_t(row_offset + row) * outputs + column] =
                __float2bfloat16_rn(total[m][n][value]);
        }
      }
}

template <class Output, class ScaleLayout, int Bits, bool Grouped, bool Prefetch = false,
          int TileRows = half_rows>
__global__ void half_k32_kernel(const uint8_t *__restrict__ input, const uint8_t *weight,
                                const uint8_t *__restrict__ input_scales,
                                const uint8_t *weight_scales, Output *__restrict__ output,
                                const int64_t *weight_ptrs, const int64_t *scale_ptrs,
                                const int32_t *offsets, const int32_t *tiles, int groups, int rows,
                                int outputs, int width, int weight_stride, ScaleLayout layout) {
  static_assert(TileRows == 16 || TileRows == half_rows);
  static_assert(TileRows == half_rows || (Grouped && Bits == 4));
  __shared__ __align__(16) uint16_t data[(TileRows + half_columns) * half_stage];
  __shared__ uint8_t sa[TileRows * 4], sb[half_columns * 4];
  auto *as = data;
  auto *bs = data + TileRows * half_stage;
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  if constexpr (Grouped) {
    if (offsets[groups] < 0) {
      for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
           index < int64_t(rows) * outputs; index += int64_t(gridDim.x) * blockDim.x)
        output[index] = Output(fp8_scale(255));
      return;
    }
  }
  int column_tiles = (outputs + half_columns - 1) / half_columns;
  int work = Grouped ? tiles[groups] : ((rows + TileRows - 1) / TileRows) * column_tiles;
  for (int task = blockIdx.x; task < work; task += gridDim.x) {
    int row_offset = 0, group_rows = rows, local_task = task;
    if constexpr (Grouped) {
      int low = 0, high = groups;
      while (low + 1 < high) {
        int middle = (low + high) / 2;
        if (tiles[middle] <= task)
          low = middle;
        else
          high = middle;
      }
      row_offset = offsets[low];
      group_rows = offsets[low + 1] - row_offset;
      local_task -= tiles[low];
      weight = reinterpret_cast<const uint8_t *>(weight_ptrs[low]);
      weight_scales = reinterpret_cast<const uint8_t *>(scale_ptrs[low]);
    }
    int first_row = (local_task / column_tiles) * TileRows;
    int first_column = (local_task % column_tiles) * half_columns;
    int valid_rows = min(TileRows, group_rows - first_row), width_groups = width / 32;
    float total[TileRows / 16][2][4] = {};
    HalfFp4Prefetch prefetched;
    if constexpr (Prefetch) {
      static_assert(Grouped && Bits == 4 && std::is_same_v<Output, __nv_bfloat16>);
      half_prefetch_fp4(prefetched, weight, weight_scales, first_column, outputs, width,
                        weight_stride, 0, layout);
    }
    for (int start = 0; start < width; start += half_stage) {
      int columns = min(half_stage, width - start), stage_groups = columns / 32;
      half_load_stage<Bits, Prefetch, TileRows>(
          input, weight, input_scales, weight_scales, as, bs, sa, sb, prefetched, valid_rows,
          columns, row_offset, first_row, first_column, outputs, width, weight_stride, width_groups,
          start, stage_groups, layout);
      __syncthreads();
      if constexpr (Prefetch) {
        if (start + half_stage < width)
          half_prefetch_fp4(prefetched, weight, weight_scales, first_column, outputs, width,
                            weight_stride, start + half_stage, layout);
      }
      half_accumulate_stage(as, bs, sa, sb, total, stage_groups, warp, lane);
      __syncthreads();
    }
    half_store_output(output, total, first_row, first_column, row_offset, group_rows, outputs, warp,
                      lane);
    __syncthreads();
  }
}

template <class Output, class ScaleLayout, int Bits, bool Grouped, int TileRows>
void launch_half_k32(const uint8_t *input, const uint8_t *weight, const uint8_t *input_scales,
                     const uint8_t *weight_scales, Output *output, const int64_t *weight_ptrs,
                     const int64_t *scale_ptrs, const int32_t *offsets, const int32_t *tiles,
                     int groups, int rows, int outputs, int width, int weight_stride,
                     ScaleLayout layout, cudaStream_t stream) {
  int bound = ((rows + TileRows - 1) / TileRows + (Grouped ? groups : 0)) *
              ((outputs + half_columns - 1) / half_columns);
  constexpr int blocks_per_sm =
      Grouped && Bits == 4 && std::is_same_v<Output, __nv_bfloat16> ? (TileRows == 16 ? 10 : 7) : 8;
  if constexpr (Grouped && Bits == 4 && std::is_same_v<Output, __nv_bfloat16>) {
    if (rows > 64 && (TileRows == half_rows || rows <= groups) && outputs == 576 && width == 5120) {
      half_k32_kernel<Output, ScaleLayout, Bits, Grouped, true, TileRows>
          <<<min(bound, 148 * blocks_per_sm), 128, 0, stream>>>(
              input, weight, input_scales, weight_scales, output, weight_ptrs, scale_ptrs, offsets,
              tiles, groups, rows, outputs, width, weight_stride, layout);
      return;
    }
  }
  half_k32_kernel<Output, ScaleLayout, Bits, Grouped, false, TileRows>
      <<<min(bound, 148 * blocks_per_sm), 128, 0, stream>>>(
          input, weight, input_scales, weight_scales, output, weight_ptrs, scale_ptrs, offsets,
          tiles, groups, rows, outputs, width, weight_stride, layout);
}

} // namespace encoded_projection
