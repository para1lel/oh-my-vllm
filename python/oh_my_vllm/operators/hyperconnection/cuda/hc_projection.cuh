#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <dlfcn.h>
#include <memory>
#include <unordered_map>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>

namespace hc_projection {

constexpr int FlatWidth = 20480, Mixes = 24;
constexpr int64_t BlasBytes = 32 * 1024 * 1024;

inline void check_blas(cublasStatus_t status) {
  TVM_FFI_ICHECK(status == CUBLAS_STATUS_SUCCESS) << "cuBLAS status " << int(status);
}

struct BlasOwner {
  cublasHandle_t handle = nullptr;
  cublasHandle_t limited = nullptr;
  int device;

  explicit BlasOwner(int device) : device(device) {
    int previous = 0;
    check_cuda(cudaGetDevice(&previous));
    check_cuda(cudaSetDevice(device));
    cublasStatus_t status = cublasCreate(&handle);
    if (status == CUBLAS_STATUS_SUCCESS)
      status = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    if (status == CUBLAS_STATUS_SUCCESS)
      status = cublasCreate(&limited);
    if (status == CUBLAS_STATUS_SUCCESS)
      status = cublasSetMathMode(limited, CUBLAS_DEFAULT_MATH);
    // The released BF16 multi-row projection uses a smaller SM target. This
    // avoids the split-K kernel that the current library selects by default.
    if (status == CUBLAS_STATUS_SUCCESS)
      status = cublasSetSmCountTarget(limited, 8);
    if (status != CUBLAS_STATUS_SUCCESS) {
      if (limited)
        cublasDestroy(limited);
      if (handle)
        cublasDestroy(handle);
      check_cuda(cudaSetDevice(previous));
      check_blas(status);
    }
    check_cuda(cudaSetDevice(previous));
  }

  ~BlasOwner() {
    int previous = 0;
    if (cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device) == cudaSuccess) {
      cublasDestroy(handle);
      cublasDestroy(limited);
      cudaSetDevice(previous);
    }
  }
};

inline BlasOwner &owner(int device, cudaStream_t stream) {
  static thread_local std::unordered_map<int, std::unique_ptr<BlasOwner>> owners;
  auto found = owners.find(device);
  if (found == owners.end()) {
    cudaStreamCaptureStatus capture;
    check_cuda(cudaStreamIsCapturing(stream, &capture));
    TVM_FFI_ICHECK(capture == cudaStreamCaptureStatusNone) << "prepare HC before Graph capture";
    found = owners.emplace(device, std::make_unique<BlasOwner>(device)).first;
  }
  return *found->second;
}

inline cublasHandle_t handle(int device, cudaStream_t stream) {
  return owner(device, stream).handle;
}

inline cublasHandle_t limited_handle(int device, cudaStream_t stream) {
  return owner(device, stream).limited;
}

// CUDA 13.0 released a single-row dot as 32 split blocks of 128 threads. Each
// thread accumulates the even and odd element of one interleaved pair. The
// block tree and the four-group split reduction keep the released order.
constexpr int ReleasedSplits = 32;
constexpr int ReleasedGroups = 4;

// Generated from the released CUDA 13.0 cuBLAS image in torch 2.13.0+cu130.
// Generator: classify_chunks.cu (external artifact)
// Row counts: measured 1..32768; N=1 uses the dot order.
struct ChunkRange {
  int first;
  int last;
  int chunk;
};

constexpr ChunkRange kChunkRanges[] = {
    {2, 5, 192},        {6, 7, 224},         {8, 16, 192},       {17, 128, 160},
    {129, 192, 448},    {193, 256, 288},     {257, 384, 448},    {385, 512, 576},
    {513, 572, 736},    {573, 639, 352},     {640, 640, 736},    {641, 726, 864},
    {727, 768, 448},    {769, 896, 512},     {897, 992, 576},    {993, 1024, 224},
    {1025, 1152, 256},  {1153, 1280, 288},   {1281, 1408, 320},  {1409, 1536, 352},
    {1537, 1664, 384},  {1665, 1792, 416},   {1793, 2048, 448},  {2049, 2176, 480},
    {2177, 2304, 2560}, {2305, 2432, 544},   {2433, 2560, 576},  {2561, 2688, 2944},
    {2689, 2944, 640},  {2945, 3072, 3424},  {3073, 3328, 736},  {3329, 3456, 768},
    {3457, 3584, 800},  {3585, 3712, 832},   {3713, 3840, 864},  {3841, 4096, 896},
    {4097, 4224, 960},  {4225, 4480, 992},   {4481, 4736, 1024}, {4737, 4864, 1088},
    {4865, 5179, 1152}, {5180, 5184, 20480}, {5185, 5186, 1152}, {5187, 32768, 20480},
};

inline int released_chunk(int rows) {
  if (rows <= 1)
    return 0;
  int low = 0, high = int(sizeof(kChunkRanges) / sizeof(kChunkRanges[0])) - 1;
  while (low <= high) {
    int middle = (low + high) / 2;
    if (rows < kChunkRanges[middle].first)
      high = middle - 1;
    else if (rows > kChunkRanges[middle].last)
      low = middle + 1;
    else
      return kChunkRanges[middle].chunk;
  }
  // A hole in the table must stop the call, not reuse the last layout.
  return 0;
}

// Measured from the same CUDA 13.0 cuBLAS image for the routed-expert router
// (M=384, K=5120) and the ratio-based compressor (M=512, K=5120).
constexpr ChunkRange kRouterChunkRanges[] = {
    {2, 5, 160},     {6, 7, 96},      {8, 16, 128},    {17, 32, 160},    {33, 64, 128},
    {65, 127, 160},  {128, 128, 96},  {129, 256, 128}, {257, 384, 160},  {385, 461, 224},
    {462, 512, 288}, {513, 572, 384}, {573, 639, 288}, {640, 640, 384},  {641, 726, 448},
    {727, 768, 320}, {769, 896, 384}, {897, 992, 448}, {993, 1024, 224}, {1025, 32768, 5120},
};

constexpr ChunkRange kDsparkChunkRanges[] = {
    {2, 5, 128},       {6, 7, 96},        {8, 16, 128},      {17, 28, 64},        {29, 32, 96},
    {33, 53, 64},      {54, 64, 96},      {65, 81, 160},     {82, 128, 96},       {129, 192, 128},
    {193, 256, 160},   {257, 384, 128},   {385, 512, 160},   {513, 640, 192},     {641, 768, 224},
    {769, 896, 256},   {897, 1024, 288},  {1025, 1152, 320}, {1153, 1280, 384},   {1281, 1536, 224},
    {1537, 1792, 256}, {1793, 2048, 288}, {2049, 2304, 320}, {2305, 2432, 352},   {2433, 2688, 384},
    {2689, 2816, 416}, {2817, 3072, 448}, {3073, 3168, 480}, {3169, 32768, 5120},
};

constexpr ChunkRange kCompressorChunkRanges[] = {
    {2, 5, 128},        {6, 7, 96},        {8, 16, 128},        {17, 32, 160},
    {33, 64, 64},       {65, 256, 160},    {257, 384, 224},     {385, 512, 288},
    {513, 640, 384},    {641, 726, 448},   {727, 768, 224},     {769, 827, 256},
    {828, 896, 512},    {897, 1248, 5120}, {1249, 1280, 736},   {1281, 1408, 864},
    {1409, 1473, 5120}, {1474, 1536, 864}, {1537, 32768, 5120},
};

inline int find_chunk(const ChunkRange *ranges, int count, int rows) {
  int low = 0, high = count - 1;
  while (low <= high) {
    int middle = (low + high) / 2;
    if (rows < ranges[middle].first)
      high = middle - 1;
    else if (rows > ranges[middle].last)
      low = middle + 1;
    else
      return ranges[middle].chunk;
  }
  return 0;
}

// Zero means that the shape or row count has no measured released layout.
inline int project_chunk(int rows, int outputs, int width) {
  if (rows < 2 || rows > 32768 || width != 5120)
    return 0;
  if (outputs == 384)
    return find_chunk(kRouterChunkRanges,
                      int(sizeof(kRouterChunkRanges) / sizeof(kRouterChunkRanges[0])), rows);
  if (outputs == 128)
    return find_chunk(kDsparkChunkRanges,
                      int(sizeof(kDsparkChunkRanges) / sizeof(kDsparkChunkRanges[0])), rows);
  if (outputs == 512)
    return find_chunk(kCompressorChunkRanges,
                      int(sizeof(kCompressorChunkRanges) / sizeof(kCompressorChunkRanges[0])),
                      rows);
  return 0;
}

__global__ void released_dot_kernel(const float *input, const float *weight, float *partials,
                                    int width, int outputs) {
  int split = blockIdx.x, out = blockIdx.y, thread = threadIdx.x;
  int pairs = width / 2;
  float even = 0.f, odd = 0.f;
  for (int pair = split * 128 + thread; pair < pairs; pair += ReleasedSplits * 128) {
    int k = 2 * pair;
    even = __fmaf_rn(input[k], weight[out * width + k], even);
    odd = __fmaf_rn(input[k + 1], weight[out * width + k + 1], odd);
  }
  float value = __fadd_rn(even, odd);
  __shared__ float shared[128];
  shared[thread] = value;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (thread < stride)
      shared[thread] = __fadd_rn(shared[thread], shared[thread + stride]);
    __syncthreads();
  }
  if (thread == 0)
    partials[out * ReleasedSplits + split] = shared[0];
}

__global__ void released_dot_reduce(const float *partials, float *output) {
  int out = blockIdx.x;
  if (threadIdx.x != 0)
    return;
  float groups[ReleasedGroups];
  for (int group = 0; group < ReleasedGroups; ++group) {
    float value = partials[out * ReleasedSplits + group];
    for (int index = group + ReleasedGroups; index < ReleasedSplits; index += ReleasedGroups)
      value = __fadd_rn(value, partials[out * ReleasedSplits + index]);
    groups[group] = value;
  }
  float total = groups[0];
  for (int group = 1; group < ReleasedGroups; ++group)
    total = __fadd_rn(total, groups[group]);
  output[out] = total;
}

__global__ void released_gemv_kernel(const float *input, const float *weight, float *output,
                                     int width, int outputs) {
  int out = blockIdx.x * blockDim.y + threadIdx.y;
  int lane = threadIdx.x;
  if (out >= outputs)
    return;
  float value = 0.f;
  for (int k = lane; k < width; k += 32)
    value = __fmaf_rn(input[k], weight[out * width + k], value);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value = __fadd_rn(value, __shfl_down_sync(0xffffffff, value, offset));
  if (lane == 0)
    output[out] = value;
}

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

inline void multiply_released_one(const float *input, const float *weight, float *output, int width,
                                  int outputs, void *workspace, cudaStream_t stream) {
  auto *partials = static_cast<float *>(workspace);
  released_dot_kernel<<<dim3(ReleasedSplits, outputs), 128, 0, stream>>>(input, weight, partials,
                                                                         width, outputs);
  check_cuda(cudaGetLastError());
  released_dot_reduce<<<outputs, 1, 0, stream>>>(partials, output);
  check_cuda(cudaGetLastError());
}

inline void multiply_released_gemv(const float *input, const float *weight, float *output,
                                   int width, int outputs, cudaStream_t stream) {
  dim3 block(32, 4);
  released_gemv_kernel<<<(outputs + 3) / 4, block, 0, stream>>>(input, weight, output, width,
                                                                outputs);
  check_cuda(cudaGetLastError());
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

inline void multiply_released_chunked_parallel(const float *input, const float *weight,
                                               float *output, int rows, int outputs, int width,
                                               int input_stride, int output_stride, int chunk,
                                               void *workspace, int64_t workspace_bytes,
                                               cudaStream_t stream) {
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

template <int X>
__global__ void prepare_kernel(const __nv_bfloat16 *residual, float *flat, float *statistics) {
  constexpr int Y = 512 / X;
  int row = blockIdx.x, x = threadIdx.x, y = threadIdx.y, thread = y * X + x;
  float sums[4] = {};
  for (int index = thread; index < FlatWidth / 4; index += 512)
    for (int k = 0; k < 4; ++k) {
      int offset = row * FlatWidth + index * 4 + k;
      float value = __bfloat162float(residual[offset]);
      flat[offset] = value;
      sums[k] = __fadd_rn(sums[k], __fmul_rn(value, value));
    }
  float value = sums[0];
  for (int k = 1; k < 4; ++k)
    value = __fadd_rn(value, sums[k]);
  __shared__ float scratch[512];
  if constexpr (X > 32) {
    scratch[thread] = value;
    for (int offset = X / 2; offset >= 32; offset >>= 1) {
      __syncthreads();
      if (x < offset) {
        value = __fadd_rn(value, scratch[thread + offset]);
        scratch[thread] = value;
      }
    }
  }
  __syncthreads();
  for (int offset = 16; offset; offset >>= 1)
    value = __fadd_rn(value, __shfl_down_sync(0xffffffff, value, offset));
  if constexpr (Y > 1) {
    scratch[thread] = value;
    for (int offset = Y / 2; offset; offset >>= 1) {
      __syncthreads();
      if (y < offset) {
        value = __fadd_rn(value, scratch[thread + offset * X]);
        scratch[thread] = value;
      }
    }
  }
  if (x == 0 && y == 0) {
    float mean = __fmul_rn(value, 1.f / FlatWidth);
    statistics[row * 2] = mean;
    statistics[row * 2 + 1] = rsqrtf(__fadd_rn(mean, 1e-20f));
  }
}

__global__ void normalize_kernel(float *mixed, const float *statistics, int rows) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < rows * Mixes)
    mixed[index] = __fmul_rn(mixed[index], statistics[(index / Mixes) * 2 + 1]);
}

template <int X>
void prepare(const __nv_bfloat16 *input, float *flat, float *statistics, int rows,
             cudaStream_t stream) {
  prepare_kernel<X><<<rows, dim3(X, 512 / X), 0, stream>>>(input, flat, statistics);
}

inline void project(TensorView residual, TensorView weight, TensorView mixed, TensorView flat,
                    TensorView statistics, TensorView workspace, int rows, int logical_rows,
                    cudaStream_t stream) {
  auto input = static_cast<const __nv_bfloat16 *>(residual.data_ptr());
  auto values = static_cast<float *>(flat.data_ptr());
  auto stats = static_cast<float *>(statistics.data_ptr());
  int height = 1;
  while (height < 16 && height * 2 <= logical_rows)
    height *= 2;
  // The complete logical row count selects the released reduction tree.
  switch (height) {
  case 1:
    prepare<512>(input, values, stats, rows, stream);
    break;
  case 2:
    prepare<256>(input, values, stats, rows, stream);
    break;
  case 4:
    prepare<128>(input, values, stats, rows, stream);
    break;
  case 8:
    prepare<64>(input, values, stats, rows, stream);
    break;
  case 16:
    prepare<32>(input, values, stats, rows, stream);
    break;
  }
  check_cuda(cudaGetLastError());
  {
    int steps = rows / logical_rows;
    auto weight_values = static_cast<const float *>(weight.data_ptr());
    auto mixed_values = static_cast<float *>(mixed.data_ptr());
    for (int step = 0; step < steps; ++step) {
      if (logical_rows == 1) {
        multiply_released_one(values + step * FlatWidth, weight_values, mixed_values + step * Mixes,
                              FlatWidth, Mixes, workspace.data_ptr(), stream);
      } else {
        int chunk = released_chunk(logical_rows);
        TVM_FFI_ICHECK(chunk > 0) << "HC logical rows have no measured released layout";
        multiply_released_chunked_parallel(values + step * FlatWidth, weight_values,
                                           mixed_values + step * Mixes, logical_rows, Mixes,
                                           FlatWidth, steps * FlatWidth, steps * Mixes, chunk,
                                           workspace.data_ptr(), BlasBytes, stream);
      }
    }
  }
  normalize_kernel<<<(rows * Mixes + 255) / 256, 256, 0, stream>>>(
      static_cast<float *>(mixed.data_ptr()), stats, rows);
  check_cuda(cudaGetLastError());
}

__global__ void released_gemv8_kernel(const float *input, const float *weight, float *output,
                                      int rows, int width, int outputs) {
  int row = blockIdx.y;
  int out = blockIdx.x * blockDim.y + threadIdx.y;
  int lane = threadIdx.x;
  bool active = row < rows && out < outputs;
  float value = 0.f;
  if (active) {
    const float *row_input = input + int64_t(row) * width;
    for (int k = lane; k < width; k += 8)
      value = __fmaf_rn(row_input[k], weight[int64_t(out) * width + k], value);
  }
  // Every lane of the warp must execute the shuffle, even for a partial output block.
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1)
    value = __fadd_rn(value, __shfl_down_sync(0xffffffff, value, offset, 8));
  if (active && lane == 0)
    output[int64_t(row) * outputs + out] = value;
}

inline void multiply_released_gemv8(const float *input, const float *weight, float *output,
                                    int rows, int width, int outputs, cudaStream_t stream) {
  dim3 block(8, 32);
  released_gemv8_kernel<<<dim3((outputs + 31) / 32, rows), block, 0, stream>>>(
      input, weight, output, rows, width, outputs);
  check_cuda(cudaGetLastError());
}

__global__ void released_dot_fused_kernel(const float *input, const float *weight, float *output,
                                          int width, int outputs) {
  int index = blockIdx.x, out = index % outputs, row = index / outputs;
  int thread = threadIdx.x, pairs = width / 2;
  __shared__ float tree[128];
  __shared__ float partials[ReleasedSplits];
  for (int split = 0; split < ReleasedSplits; ++split) {
    float even = 0.f, odd = 0.f;
    for (int pair = split * 128 + thread; pair < pairs; pair += ReleasedSplits * 128) {
      int k = 2 * pair;
      const float *row_input = input + int64_t(row) * width;
      even = __fmaf_rn(row_input[k], weight[out * width + k], even);
      odd = __fmaf_rn(row_input[k + 1], weight[out * width + k + 1], odd);
    }
    tree[thread] = __fadd_rn(even, odd);
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
      if (thread < stride)
        tree[thread] = __fadd_rn(tree[thread], tree[thread + stride]);
      __syncthreads();
    }
    if (thread == 0)
      partials[split] = tree[0];
    __syncthreads();
  }
  if (thread != 0)
    return;
  float groups[ReleasedGroups];
  for (int group = 0; group < ReleasedGroups; ++group) {
    float value = partials[group];
    for (int i = group + ReleasedGroups; i < ReleasedSplits; i += ReleasedGroups)
      value = __fadd_rn(value, partials[i]);
    groups[group] = value;
  }
  float total = groups[0];
  for (int group = 1; group < ReleasedGroups; ++group)
    total = __fadd_rn(total, groups[group]);
  output[index] = total;
}

inline void multiply_released_dot(const float *input, const float *weight, float *output, int rows,
                                  int outputs, int width, cudaStream_t stream) {
  int count = rows * outputs;
  released_dot_fused_kernel<<<count, 128, 0, stream>>>(input, weight, output, width, outputs);
  check_cuda(cudaGetLastError());
}

inline void bf16_multiply(int device, const void *input, const void *weight, void *output, int rows,
                          int outputs, int width, cudaStream_t stream) {
  auto blas = rows == 1 ? handle(device, stream) : limited_handle(device, stream);
  check_blas(cublasSetStream(blas, stream));
  float alpha = 1.f, beta = 0.f;
  check_blas(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, outputs, rows, width, &alpha, weight,
                          CUDA_R_16BF, width, input, CUDA_R_16BF, width, &beta, output, CUDA_R_16BF,
                          outputs, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

} // namespace hc_projection
