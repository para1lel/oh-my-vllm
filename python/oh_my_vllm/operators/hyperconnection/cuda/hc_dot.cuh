#pragma once

// CUDA 13.0 released a single-row dot as 32 split blocks of 128 threads. Each
// thread accumulates the even and odd element of one interleaved pair. The
// block tree and the four-group split reduction keep the released order.
constexpr int ReleasedSplits = 32;
constexpr int ReleasedGroups = 4;

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
  input += blockIdx.y * width;
  output += blockIdx.y * outputs;
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

inline void multiply_released_one(const float *input, const float *weight, float *output, int width,
                                  int outputs, void *workspace, cudaStream_t stream) {
  auto *partials = static_cast<float *>(workspace);
  released_dot_kernel<<<dim3(ReleasedSplits, outputs), 128, 0, stream>>>(input, weight, partials,
                                                                         width, outputs);
  check_cuda(cudaGetLastError());
  released_dot_reduce<<<outputs, 1, 0, stream>>>(partials, output);
  check_cuda(cudaGetLastError());
}

inline void multiply_released_gemv(const float *input, const float *weight, float *output, int rows,
                                   int width, int outputs, cudaStream_t stream) {
  dim3 block(32, 4);
  released_gemv_kernel<<<dim3((outputs + 3) / 4, rows), block, 0, stream>>>(input, weight, output,
                                                                            width, outputs);
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

__global__ void released_dot_warp_kernel(const float *input, const float *weight, float *output,
                                         int width, int outputs) {
  int index = blockIdx.x, out = index % outputs, row = index / outputs;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32, pairs = width / 2;
  const float *row_input = input + int64_t(row) * width;
  const float *row_weight = weight + int64_t(out) * width;
  __shared__ float partials[ReleasedSplits];
  for (int split = warp; split < ReleasedSplits; split += 4) {
    float even[4] = {0.f, 0.f, 0.f, 0.f};
    float odd[4] = {0.f, 0.f, 0.f, 0.f};
    for (int base = split * 128 + lane; base < pairs; base += ReleasedSplits * 128) {
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        int k = 2 * (base + part * 32);
        if (k < width) {
          even[part] = __fmaf_rn(row_input[k], row_weight[k], even[part]);
          odd[part] = __fmaf_rn(row_input[k + 1], row_weight[k + 1], odd[part]);
        }
      }
    }
    float values[4];
#pragma unroll
    for (int part = 0; part < 4; ++part)
      values[part] = __fadd_rn(even[part], odd[part]);
    // Keep the original stride-64, stride-32, and lane tree operands.
    float left = __fadd_rn(values[0], values[2]);
    float right = __fadd_rn(values[1], values[3]);
    float value = __fadd_rn(left, right);
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
      float other = __shfl_down_sync(0xffffffff, value, stride);
      if (lane < stride)
        value = __fadd_rn(value, other);
    }
    if (lane == 0)
      partials[split] = value;
  }
  __syncthreads();
  if (threadIdx.x != 0)
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
  if (rows >= 1024)
    released_dot_warp_kernel<<<count, 128, 0, stream>>>(input, weight, output, width, outputs);
  else
    released_dot_fused_kernel<<<count, 128, 0, stream>>>(input, weight, output, width, outputs);
  check_cuda(cudaGetLastError());
}
