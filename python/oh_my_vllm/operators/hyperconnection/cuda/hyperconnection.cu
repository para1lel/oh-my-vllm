#include <cuda_bf16.h>
#include "../../runtime/cuda/contracts.cuh"

namespace {

constexpr int width = 5120, copies = 4, flat_width = width * copies, mixes = 24;

__device__ float warp_sum(float value) {
  for (int offset = 16; offset; offset /= 2)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

__global__ void project_kernel(const __nv_bfloat16 *residual, const float *weight, float *mixed) {
  int row = blockIdx.x, coefficient = blockIdx.y, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  float dot = 0.f, norm = 0.f;
  for (int column = threadIdx.x; column < flat_width; column += blockDim.x) {
    float value = __bfloat162float(residual[row * flat_width + column]);
    dot = fmaf(value, weight[coefficient * flat_width + column], dot);
    norm = fmaf(value, value, norm);
  }
  dot = warp_sum(dot);
  norm = warp_sum(norm);
  __shared__ float partials[2][8];
  if (lane == 0) {
    partials[0][warp] = dot;
    partials[1][warp] = norm;
  }
  __syncthreads();
  if (warp == 0) {
    dot = warp_sum(lane < 8 ? partials[0][lane] : 0.f);
    norm = warp_sum(lane < 8 ? partials[1][lane] : 0.f);
    if (lane == 0)
      mixed[row * mixes + coefficient] = dot * rsqrtf(norm / flat_width + 1e-20f);
  }
}

__device__ float row_sum(float value) {
  value += __shfl_xor_sync(0xffff, value, 1, 16);
  return value + __shfl_xor_sync(0xffff, value, 2, 16);
}

__device__ float column_sum(float value) {
  value += __shfl_xor_sync(0xffff, value, 4, 16);
  return value + __shfl_xor_sync(0xffff, value, 8, 16);
}

__global__ void split_kernel(const float *mixed, const float *scale, const float *base, float *pre,
                             float *post, float *comb) {
  int row = blockIdx.x, lane = threadIdx.x;
  if (lane < copies) {
    float before = mixed[row * mixes + lane] * scale[0] + base[lane];
    float after = mixed[row * mixes + copies + lane] * scale[1] + base[copies + lane];
    pre[row * copies + lane] = 1.f / (1.f + expf(-before)) + 1e-6f;
    post[row * copies + lane] = 2.f / (1.f + expf(-after));
  }
  if (lane < 16) {
    float value = mixed[row * mixes + 8 + lane] * scale[2] + base[8 + lane];
    float peak = fmaxf(value, __shfl_xor_sync(0xffff, value, 1, 16));
    peak = fmaxf(peak, __shfl_xor_sync(0xffff, peak, 2, 16));
    value = expf(value - peak);
    value = value / row_sum(value) + 1e-6f;
    value /= column_sum(value) + 1e-6f;
    for (int iteration = 1; iteration < 20; ++iteration) {
      value /= row_sum(value) + 1e-6f;
      value /= column_sum(value) + 1e-6f;
    }
    comb[row * 16 + lane] = value;
  }
}

__global__ void collapse_kernel(const __nv_bfloat16 *residual, const float *incoming,
                                __nv_bfloat16 *output) {
  int row = blockIdx.x, column = blockIdx.y * blockDim.x + threadIdx.x;
  if (column >= width)
    return;
  float value = 0.f;
  for (int copy = 0; copy < copies; ++copy)
    value += incoming[row * copies + copy] *
             __bfloat162float(residual[row * flat_width + copy * width + column]);
  output[row * width + column] = __float2bfloat16_rn(value);
}

__global__ void expand_kernel(const __nv_bfloat16 *residual, const __nv_bfloat16 *branch,
                              const float *post, const float *comb, __nv_bfloat16 *output) {
  int row = blockIdx.x, column = blockIdx.y * blockDim.x + threadIdx.x;
  if (column >= width)
    return;
  float value = __bfloat162float(branch[row * width + column]);
  for (int destination = 0; destination < copies; ++destination) {
    float result = post[row * copies + destination] * value;
    for (int source = 0; source < copies; ++source)
      result += comb[row * 16 + source * copies + destination] *
                __bfloat162float(residual[row * flat_width + source * width + column]);
    output[row * flat_width + destination * width + column] = __float2bfloat16_rn(result);
  }
}

int rows_contract(TensorView residual) {
  TVM_FFI_ICHECK(residual.ndim() == 3 && residual.size(0) >= 0 && residual.size(0) <= 384);
  int rows = residual.size(0);
  tensor_contract(residual, residual, kDLBfloat, 16, {rows, copies, width});
  return rows;
}

void separate(std::initializer_list<TensorView> values) {
  for (auto left = values.begin(); left != values.end(); ++left)
    for (auto right = left + 1; right != values.end(); ++right)
      disjoint(*left, *right);
}

} // namespace

void hc_collapse(TensorView residual, TensorView incoming, TensorView collapsed) {
  int rows = rows_contract(residual);
  tensor_contract(incoming, residual, kDLFloat, 32, {rows, copies});
  tensor_contract(collapsed, residual, kDLBfloat, 16, {rows, width});
  separate({residual, incoming, collapsed});
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(residual.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, residual.device().device_id));
  collapse_kernel<<<dim3(rows, (width + 255) / 256), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
      static_cast<const float *>(incoming.data_ptr()),
      static_cast<__nv_bfloat16 *>(collapsed.data_ptr()));
  check_cuda(cudaGetLastError());
}

void hc_prepare(TensorView residual, TensorView incoming, TensorView weight, TensorView scale,
                TensorView base, TensorView mixed, TensorView post, TensorView comb,
                TensorView collapsed, TensorView next_pre) {
  int rows = rows_contract(residual);
  tensor_contract(incoming, residual, kDLFloat, 32, {rows, copies});
  tensor_contract(weight, residual, kDLFloat, 32, {mixes, flat_width});
  tensor_contract(scale, residual, kDLFloat, 32, {3});
  tensor_contract(base, residual, kDLFloat, 32, {mixes});
  tensor_contract(mixed, residual, kDLFloat, 32, {rows, mixes});
  tensor_contract(post, residual, kDLFloat, 32, {rows, copies});
  tensor_contract(comb, residual, kDLFloat, 32, {rows, copies, copies});
  tensor_contract(collapsed, residual, kDLBfloat, 16, {rows, width});
  tensor_contract(next_pre, residual, kDLFloat, 32, {rows, copies});
  separate({residual, incoming, weight, scale, base, mixed, post, comb, collapsed, next_pre});
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(residual.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, residual.device().device_id));
  project_kernel<<<dim3(rows, mixes), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
      static_cast<const float *>(weight.data_ptr()), static_cast<float *>(mixed.data_ptr()));
  check_cuda(cudaGetLastError());
  split_kernel<<<rows, 32, 0, stream>>>(
      static_cast<const float *>(mixed.data_ptr()), static_cast<const float *>(scale.data_ptr()),
      static_cast<const float *>(base.data_ptr()), static_cast<float *>(next_pre.data_ptr()),
      static_cast<float *>(post.data_ptr()), static_cast<float *>(comb.data_ptr()));
  check_cuda(cudaGetLastError());
  hc_collapse(residual, incoming, collapsed);
}

void hc_expand(TensorView residual, TensorView branch, TensorView post, TensorView comb,
               TensorView expanded) {
  int rows = rows_contract(residual);
  tensor_contract(branch, residual, kDLBfloat, 16, {rows, width});
  tensor_contract(post, residual, kDLFloat, 32, {rows, copies});
  tensor_contract(comb, residual, kDLFloat, 32, {rows, copies, copies});
  tensor_contract(expanded, residual, kDLBfloat, 16, {rows, copies, width});
  separate({residual, branch, post, comb, expanded});
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(residual.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, residual.device().device_id));
  expand_kernel<<<dim3(rows, (width + 255) / 256), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
      static_cast<const __nv_bfloat16 *>(branch.data_ptr()),
      static_cast<const float *>(post.data_ptr()), static_cast<const float *>(comb.data_ptr()),
      static_cast<__nv_bfloat16 *>(expanded.data_ptr()));
  check_cuda(cudaGetLastError());
}
