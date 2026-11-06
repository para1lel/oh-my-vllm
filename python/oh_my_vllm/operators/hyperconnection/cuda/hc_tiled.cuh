#pragma once

__global__ void released_chunk_kernel(const float *input, const float *weight, float *output,
                                      int rows, int outputs, int width, int input_stride,
                                      int output_stride, int chunk) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * outputs)
    return;
  int row = index / outputs, out = index % outputs;
  float total = 0.f;
  for (int start = 0; start < width; start += chunk) {
    int end = start + chunk < width ? start + chunk : width;
    float partial = 0.f;
    for (int k = start; k < end; ++k)
      partial = __fmaf_rn(input[row * input_stride + k], weight[out * width + k], partial);
    total = __fadd_rn(total, partial);
  }
  output[row * output_stride + out] = total;
}

inline void multiply_released_chunked(const float *input, const float *weight, float *output,
                                      int rows, int outputs, int width, int input_stride,
                                      int output_stride, int chunk, cudaStream_t stream) {
  int count = rows * outputs;
  int blocks = (count + 255) / 256;
  released_chunk_kernel<<<blocks, 256, 0, stream>>>(input, weight, output, rows, outputs, width,
                                                    input_stride, output_stride, chunk);
  check_cuda(cudaGetLastError());
}

#include "released_tile.cuh"

template <int TileRows, int TileOutputs, int TileK>
__device__ __forceinline__ void
load_released_tile(const float *input, const float *weight, float (&inputs)[TileK][TileRows + 1],
                   float (&weights)[TileK][TileOutputs + 1], int rows, int outputs, int width,
                   int input_stride, int row_base, int out_base, int base, int end, int thread) {
  __syncthreads();
  for (int index = thread; index < TileRows * TileK / 4; index += 128) {
    int row = index / (TileK / 4), k = (index % (TileK / 4)) * 4;
    float4 value = {};
    if (row_base + row < rows && base + k < end)
      value = *reinterpret_cast<const float4 *>(input + (row_base + row) * input_stride + base + k);
    inputs[k][row] = value.x;
    inputs[k + 1][row] = value.y;
    inputs[k + 2][row] = value.z;
    inputs[k + 3][row] = value.w;
  }
  for (int index = thread; index < TileOutputs * TileK / 4; index += 128) {
    int out = index / (TileK / 4), k = (index % (TileK / 4)) * 4;
    float4 value = {};
    if (out_base + out < outputs && base + k < end)
      value = *reinterpret_cast<const float4 *>(weight + (out_base + out) * width + base + k);
    weights[k][out] = value.x;
    weights[k + 1][out] = value.y;
    weights[k + 2][out] = value.z;
    weights[k + 3][out] = value.w;
  }
}

// Each accumulator keeps the sequential K order. Independent rows and outputs
// share coalesced global reads without changing any FP32 operation.
template <int RowValues, bool Split, bool Whole = false>
__global__ void released_tile_kernel(const float *input, const float *weight, float *output,
                                     int rows, int outputs, int width, int input_stride,
                                     int output_stride, int chunk, int first_chunk = 0) {
  constexpr int ColValues = RowValues == 1 ? 4 : 8;
  constexpr int TileRows = 8 * RowValues, TileOutputs = 16 * ColValues, TileK = 32;
  int thread = threadIdx.x, r = thread / 16, o = thread % 16;
  int row_base = blockIdx.y * TileRows, out_base = blockIdx.x * TileOutputs;
  __shared__ float inputs[TileK][TileRows + 1];
  __shared__ float weights[TileK][TileOutputs + 1];
  float total[RowValues][ColValues] = {};
  int first = Split ? (int(blockIdx.z) + first_chunk) * chunk : 0;
  int limit = Split ? min(first + chunk, width) : width;
  for (int start = first; start < limit; start += chunk) {
    float partial[RowValues][ColValues] = {};
    int end = min(start + chunk, width);
    for (int base = start; base < end; base += TileK) {
      load_released_tile<TileRows, TileOutputs, TileK>(input, weight, inputs, weights, rows,
                                                       outputs, width, input_stride, row_base,
                                                       out_base, base, end, thread);
      __syncthreads();
#pragma unroll
      for (int k = 0; k < TileK; ++k) {
        float a[RowValues], b[ColValues];
#pragma unroll
        for (int i = 0; i < RowValues; ++i)
          a[i] = inputs[k][r + i * 8];
#pragma unroll
        for (int j = 0; j < ColValues; ++j)
          b[j] = weights[k][o + j * 16];
#pragma unroll
        for (int i = 0; i < RowValues; ++i)
#pragma unroll
          for (int j = 0; j < ColValues; ++j)
            partial[i][j] = __fmaf_rn(a[i], b[j], partial[i][j]);
      }
    }
#pragma unroll
    for (int i = 0; i < RowValues; ++i)
#pragma unroll
      for (int j = 0; j < ColValues; ++j)
        if constexpr (Split)
          total[i][j] = partial[i][j];
        else
          total[i][j] = __fadd_rn(Whole ? 0.f : total[i][j], partial[i][j]);
    if constexpr (Split || Whole)
      break;
  }
#pragma unroll
  for (int i = 0; i < RowValues; ++i) {
    int row = row_base + r + i * 8;
#pragma unroll
    for (int j = 0; j < ColValues; ++j) {
      int out = out_base + o + j * 16;
      if (row < rows && out < outputs) {
        if constexpr (Split)
          output[(int64_t(blockIdx.z) * rows + row) * outputs + out] = total[i][j];
        else
          output[row * output_stride + out] = total[i][j];
      }
    }
  }
}

__global__ void released_tile_reduce(const float *partials, float *output, int rows, int outputs,
                                     int chunks, int output_stride, bool accumulate) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * outputs)
    return;
  int offset = (index / outputs) * output_stride + index % outputs;
  float total = accumulate ? output[offset] : 0.f;
  for (int part = 0; part < chunks; ++part)
    total = __fadd_rn(total, partials[int64_t(part) * rows * outputs + index]);
  output[offset] = total;
}

template <int RowValues>
inline void multiply_released_tiled(const float *input, const float *weight, float *output,
                                    int rows, int outputs, int width, int input_stride,
                                    int output_stride, int chunk, void *workspace,
                                    int64_t workspace_bytes, cudaStream_t stream) {
  int chunks = (width + chunk - 1) / chunk;
  constexpr int TileOutputs = RowValues == 1 ? 64 : 128;
  dim3 grid((outputs + TileOutputs - 1) / TileOutputs, (rows + 8 * RowValues - 1) / (8 * RowValues),
            1);
  int64_t partial_bytes = int64_t(rows) * outputs * sizeof(float);
  int capacity = static_cast<int>(workspace_bytes / partial_bytes);
  if (chunks > 1 && workspace && capacity > 0) {
    auto partials = static_cast<float *>(workspace);
    for (int first = 0; first < chunks; first += capacity) {
      grid.z = min(capacity, chunks - first);
      released_tile_kernel<RowValues, true><<<grid, 128, 0, stream>>>(
          input, weight, partials, rows, outputs, width, input_stride, outputs, chunk, first);
      check_cuda(cudaGetLastError());
      released_tile_reduce<<<(rows * outputs + 255) / 256, 256, 0, stream>>>(
          partials, output, rows, outputs, grid.z, output_stride, first != 0);
      check_cuda(cudaGetLastError());
    }
  } else if (chunks == 1) {
    released_tile_kernel<RowValues, false, true><<<grid, 128, 0, stream>>>(
        input, weight, output, rows, outputs, width, input_stride, output_stride, chunk);
  } else {
    released_tile_kernel<RowValues, false><<<grid, 128, 0, stream>>>(
        input, weight, output, rows, outputs, width, input_stride, output_stride, chunk);
  }
  check_cuda(cudaGetLastError());
}

inline void multiply_released_tiled(const float *input, const float *weight, float *output,
                                    int rows, int outputs, int width, int input_stride,
                                    int output_stride, int chunk, void *workspace,
                                    int64_t workspace_bytes, cudaStream_t stream) {
  if (chunk == width && rows >= 64) {
    check_cuda(cudaFuncSetAttribute(released_whole_kernel<128>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    released_whole_kernel<128><<<dim3((outputs + 127) / 128, (rows + 63) / 64), 128, 0, stream>>>(
        input, weight, output, rows, outputs, width, input_stride, output_stride);
    check_cuda(cudaGetLastError());
  } else if (rows <= 8)
    multiply_released_tiled<1>(input, weight, output, rows, outputs, width, input_stride,
                               output_stride, chunk, workspace, workspace_bytes, stream);
  else
    multiply_released_tiled<8>(input, weight, output, rows, outputs, width, input_stride,
                               output_stride, chunk, workspace, workspace_bytes, stream);
}

// Each (row, output, chunk) partial keeps the released sequential FMA order.
// The separate reduction adds the partials in chunk order.
constexpr int ChunkTile = 256;
constexpr int ChunkRows = 8;
constexpr int ChunkThreads = ChunkRows * Mixes;

inline int chunk_count(int width, int chunk) { return (width + chunk - 1) / chunk; }

inline int64_t chunk_workspace_bytes(int rows, int width, int chunk) {
  return int64_t(rows) * Mixes * chunk_count(width, chunk) * sizeof(float);
}

__global__ void chunk_partial_kernel(const float *input, const float *weight, float *partials,
                                     int rows, int width, int input_stride, int chunk) {
  int chunk_index = blockIdx.x, row_base = blockIdx.y * ChunkRows;
  int start = chunk_index * chunk;
  int end = start + chunk < width ? start + chunk : width;
  int thread = threadIdx.x, row = thread / Mixes, out = thread - row * Mixes;
  int global_row = row_base + row;
  bool active = global_row < rows;
  float partial = 0.f;
  __shared__ float shared_weight[ChunkTile][Mixes + 1];
  __shared__ float shared_input[ChunkTile][ChunkRows + 1];
  for (int base = start; base < end; base += ChunkTile) {
    int tile = end - base < ChunkTile ? end - base : ChunkTile;
    __syncthreads();
    for (int index = thread; index < Mixes * tile; index += ChunkThreads) {
      int o = index / tile, k = index - o * tile;
      shared_weight[k][o] = weight[o * width + base + k];
    }
    for (int index = thread; index < ChunkRows * tile; index += ChunkThreads) {
      int r = index / tile, k = index - r * tile;
      int source = row_base + r;
      shared_input[k][r] = source < rows ? input[source * input_stride + base + k] : 0.f;
    }
    __syncthreads();
    if (active)
      for (int k = 0; k < tile; ++k)
        partial = __fmaf_rn(shared_input[k][row], shared_weight[k][out], partial);
  }
  if (active)
    partials[chunk_index * rows * Mixes + global_row * Mixes + out] = partial;
}

__global__ void chunk_reduce_kernel(const float *partials, float *output, int count, int chunks,
                                    int output_stride) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  float total = 0.f;
  for (int chunk_index = 0; chunk_index < chunks; ++chunk_index)
    total = __fadd_rn(total, partials[chunk_index * count + index]);
  int row = index / Mixes, out = index - row * Mixes;
  output[row * output_stride + out] = total;
}

#include "source_tile.cuh"

inline void multiply_released_chunked_parallel(const float *input, const float *weight,
                                               float *output, int rows, int outputs, int width,
                                               int input_stride, int output_stride, int chunk,
                                               void *workspace, int64_t workspace_bytes,
                                               cudaStream_t stream) {
  if (outputs == Mixes && chunk == width && rows >= 64) {
    released_whole_kernel<32><<<dim3(1, (rows + 63) / 64), 128, 0, stream>>>(
        input, weight, output, rows, outputs, width, input_stride, output_stride);
    check_cuda(cudaGetLastError());
    return;
  }
  int chunks = chunk_count(width, chunk);
  int64_t needed = int64_t(rows) * outputs * chunks * int64_t(sizeof(float));
  if (outputs != Mixes || !workspace || workspace_bytes < needed) {
    multiply_released_chunked(input, weight, output, rows, outputs, width, input_stride,
                              output_stride, chunk, stream);
    return;
  }
  auto *partials = static_cast<float *>(workspace);
  dim3 grid(chunks, (rows + ChunkRows - 1) / ChunkRows);
  chunk_partial_kernel<<<grid, ChunkThreads, 0, stream>>>(input, weight, partials, rows, width,
                                                          input_stride, chunk);
  check_cuda(cudaGetLastError());
  int count = rows * Mixes;
  chunk_reduce_kernel<<<(count + 255) / 256, 256, 0, stream>>>(partials, output, count, chunks,
                                                               output_stride);
  check_cuda(cudaGetLastError());
}
