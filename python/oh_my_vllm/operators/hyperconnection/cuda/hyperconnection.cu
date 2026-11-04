#include <cuda_bf16.h>
#include "../../runtime/cuda/contracts.cuh"
#include "hc_projection.cuh"

namespace {

constexpr int width = 5120, copies = 4, flat_width = width * copies, mixes = 24;

__device__ float row_sum(float value) {
  value += __shfl_xor_sync(0xffff, value, 2, 16);
  return value + __shfl_xor_sync(0xffff, value, 1, 16);
}

__device__ float column_sum(float value) {
  value += __shfl_xor_sync(0xffff, value, 8, 16);
  return value + __shfl_xor_sync(0xffff, value, 4, 16);
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
    value = __fadd_rn(
        value, __fmul_rn(incoming[row * copies + copy],
                         __bfloat162float(residual[row * flat_width + copy * width + column])));
  output[row * width + column] = __float2bfloat16_rn(value);
}

__global__ void expand_kernel(const __nv_bfloat16 *residual, const __nv_bfloat16 *branch,
                              const float *post, const float *comb, __nv_bfloat16 *output) {
  int row = blockIdx.x, column = blockIdx.y * blockDim.x + threadIdx.x;
  if (column >= width)
    return;
  float value = __bfloat162float(branch[row * width + column]);
  for (int destination = 0; destination < copies; ++destination) {
    float result = 0.f;
    for (int source = 0; source < copies; ++source)
      result = __fadd_rn(
          result,
          __fmul_rn(comb[row * 16 + source * copies + destination],
                    __bfloat162float(residual[row * flat_width + source * width + column])));
    result = __fadd_rn(result, __fmul_rn(post[row * copies + destination], value));
    output[row * flat_width + destination * width + column] = __float2bfloat16_rn(result);
  }
}

int rows_contract(TensorView residual) {
  TVM_FFI_ICHECK(residual.ndim() == 3 && residual.size(0) >= 0 && residual.size(0) <= 32768);
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

int64_t workspace_size(int64_t rows) {
  TVM_FFI_ICHECK(rows >= 0 && rows <= 32768);
  return rows ? hc_projection::BlasBytes : 0;
}

void hc_warmup(TensorView reference) {
  TVM_FFI_ICHECK(reference.device().device_type == kDLCUDA);
  tvm::ffi::CUDADeviceGuard guard(reference.device().device_id);
  auto stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, reference.device().device_id));
  cudaStreamCaptureStatus capture;
  check_cuda(cudaStreamIsCapturing(stream, &capture));
  TVM_FFI_ICHECK(capture == cudaStreamCaptureStatusNone) << "prepare HC outside Graph capture";
  hc_projection::handle(reference.device().device_id, stream);
}

tvm::ffi::Array<tvm::ffi::String> hc_images() {
  tvm::ffi::Array<tvm::ffi::String> paths;
  for (const void *symbol : {reinterpret_cast<const void *>(&cublasCreate_v2),
                             reinterpret_cast<const void *>(&cublasLtCreate)}) {
    Dl_info image{};
    TVM_FFI_ICHECK(dladdr(symbol, &image) && image.dli_fname);
    paths.push_back(tvm::ffi::String(image.dli_fname));
  }
  return paths;
}

void hc_prepare_grouped(TensorView residual, TensorView incoming, TensorView weight,
                        TensorView scale, TensorView base, TensorView mixed, TensorView post,
                        TensorView comb, TensorView collapsed, TensorView next_pre, TensorView flat,
                        TensorView statistics, TensorView blas_workspace, int64_t logical_rows) {
  int rows = rows_contract(residual);
  TVM_FFI_ICHECK((rows == 0 && logical_rows == 0) ||
                 (logical_rows > 0 && logical_rows <= rows && rows % logical_rows == 0))
      << "HC logical rows must divide the physical rows";
  tensor_contract(incoming, residual, kDLFloat, 32, {rows, copies});
  tensor_contract(weight, residual, kDLFloat, 32, {mixes, flat_width});
  tensor_contract(scale, residual, kDLFloat, 32, {3});
  tensor_contract(base, residual, kDLFloat, 32, {mixes});
  tensor_contract(mixed, residual, kDLFloat, 32, {rows, mixes});
  tensor_contract(post, residual, kDLFloat, 32, {rows, copies});
  tensor_contract(comb, residual, kDLFloat, 32, {rows, copies, copies});
  tensor_contract(collapsed, residual, kDLBfloat, 16, {rows, width});
  tensor_contract(next_pre, residual, kDLFloat, 32, {rows, copies});
  tensor_contract(flat, residual, kDLFloat, 32, {rows, flat_width});
  tensor_contract(statistics, residual, kDLFloat, 32, {rows, 2});
  tensor_contract(blas_workspace, residual, kDLUInt, 8, {workspace_size(rows)});
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(blas_workspace.data_ptr()) % 256 == 0);
  separate({residual, incoming, weight, scale, base, mixed, post, comb, collapsed, next_pre, flat,
            statistics, blas_workspace});
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(residual.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, residual.device().device_id));
  hc_projection::project(residual, weight, mixed, flat, statistics, blas_workspace, rows,
                         static_cast<int>(logical_rows), stream);
  split_kernel<<<rows, 32, 0, stream>>>(
      static_cast<const float *>(mixed.data_ptr()), static_cast<const float *>(scale.data_ptr()),
      static_cast<const float *>(base.data_ptr()), static_cast<float *>(next_pre.data_ptr()),
      static_cast<float *>(post.data_ptr()), static_cast<float *>(comb.data_ptr()));
  check_cuda(cudaGetLastError());
  collapse_kernel<<<dim3(rows, (width + 255) / 256), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
      static_cast<const float *>(incoming.data_ptr()),
      static_cast<__nv_bfloat16 *>(collapsed.data_ptr()));
  check_cuda(cudaGetLastError());
}

void hc_prepare(TensorView residual, TensorView incoming, TensorView weight, TensorView scale,
                TensorView base, TensorView mixed, TensorView post, TensorView comb,
                TensorView collapsed, TensorView next_pre, TensorView flat, TensorView statistics,
                TensorView blas_workspace) {
  TVM_FFI_ICHECK(residual.ndim() == 3);
  hc_prepare_grouped(residual, incoming, weight, scale, base, mixed, post, comb, collapsed,
                     next_pre, flat, statistics, blas_workspace, residual.size(0));
}

void fp32_project(TensorView values, TensorView weight, TensorView output,
                  TensorView blas_workspace, int64_t logical_rows) {
  TVM_FFI_ICHECK(values.ndim() == 2 && values.size(0) >= 0 && values.size(0) <= 32768);
  TVM_FFI_ICHECK(weight.ndim() == 2 && weight.size(0) >= 1);
  int rows = values.size(0), outputs = weight.size(0);
  TVM_FFI_ICHECK(logical_rows >= 0 && logical_rows <= rows &&
                 (!logical_rows || rows % logical_rows == 0))
      << "projection logical rows must divide the physical rows";
  tensor_contract(values, values, kDLFloat, 32, {rows, width});
  tensor_contract(weight, values, kDLFloat, 32, {outputs, width});
  tensor_contract(output, values, kDLFloat, 32, {rows, outputs});
  TVM_FFI_ICHECK(blas_workspace.ndim() == 1);
  tensor_contract(blas_workspace, values, kDLUInt, 8, {blas_workspace.size(0)});
  TVM_FFI_ICHECK(blas_workspace.size(0) == 0 || blas_workspace.size(0) >= workspace_size(rows));
  TVM_FFI_ICHECK(blas_workspace.size(0) == 0 ||
                 reinterpret_cast<uintptr_t>(blas_workspace.data_ptr()) % 256 == 0);
  if (blas_workspace.size(0))
    separate({values, weight, output, blas_workspace});
  else
    separate({values, weight, output});
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  auto input = static_cast<const float *>(values.data_ptr());
  auto weight_values = static_cast<const float *>(weight.data_ptr());
  auto output_values = static_cast<float *>(output.data_ptr());
  int step_rows = logical_rows ? static_cast<int>(logical_rows) : rows;
  if (step_rows == 1) {
    hc_projection::multiply_released_gemv(input, weight_values, output_values, rows, width, outputs,
                                          stream);
  } else {
    int chunk = hc_projection::project_chunk(step_rows, outputs, width);
    TVM_FFI_ICHECK(chunk > 0)
        << "FP32 projection shape and row count have no measured released layout";
    if (blas_workspace.size(0))
      hc_projection::multiply_released_tiled(
          input, weight_values, output_values, rows, outputs, width, width, outputs, chunk,
          blas_workspace.data_ptr(), blas_workspace.size(0), stream);
    else
      hc_projection::multiply_released_chunked(input, weight_values, output_values, rows, outputs,
                                               width, width, outputs, chunk, stream);
  }
  check_cuda(cudaGetLastError());
}

void markov_linear(TensorView values, TensorView weight, TensorView output) {
  TVM_FFI_ICHECK(values.ndim() == 2 && weight.ndim() == 2);
  int rows = values.size(0);
  int width = weight.size(1), outputs = weight.size(0);
  TVM_FFI_ICHECK(values.numel() == rows * width);
  TVM_FFI_ICHECK(int64_t(rows) * outputs <= 0x7fffffff) << "markov output needs int64 indexing";
  TVM_FFI_ICHECK(rows <= 65535) << "markov rows exceed the grid.y limit";
  tensor_contract(values, values, kDLFloat, 32, {rows, width});
  tensor_contract(weight, values, kDLFloat, 32, {outputs, width});
  tensor_contract(output, values, kDLFloat, 32, {rows, outputs});
  separate({values, weight, output});
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  hc_projection::multiply_released_gemv8(
      static_cast<const float *>(values.data_ptr()), static_cast<const float *>(weight.data_ptr()),
      static_cast<float *>(output.data_ptr()), rows, width, outputs, stream);
  check_cuda(cudaGetLastError());
}

void dot_linear(TensorView values, TensorView weight, TensorView output) {
  TVM_FFI_ICHECK(values.ndim() == 2 && weight.ndim() == 2);
  int rows = values.size(0);
  int width = weight.size(1), outputs = weight.size(0);
  TVM_FFI_ICHECK(values.numel() == rows * width && width % 2 == 0);
  TVM_FFI_ICHECK(int64_t(rows) * outputs <= 0x7fffffff) << "dot output needs int64 indexing";
  tensor_contract(values, values, kDLFloat, 32, {rows, width});
  tensor_contract(weight, values, kDLFloat, 32, {outputs, width});
  tensor_contract(output, values, kDLFloat, 32, {rows, outputs});
  separate({values, weight, output});
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  hc_projection::multiply_released_dot(
      static_cast<const float *>(values.data_ptr()), static_cast<const float *>(weight.data_ptr()),
      static_cast<float *>(output.data_ptr()), rows, outputs, width, stream);
  check_cuda(cudaGetLastError());
}

int64_t hc_chunk(int64_t rows) {
  if (rows < 0 || rows > 32768)
    TVM_FFI_ICHECK(false) << "HC row count is outside the operator limit";
  int chunk = hc_projection::released_chunk(static_cast<int>(rows));
  TVM_FFI_ICHECK(rows <= 1 || chunk > 0) << "HC row count has no measured released layout";
  return chunk;
}

int64_t project_chunk(int64_t rows, int64_t outputs) {
  if (rows < 0 || rows > 32768 || outputs < 0 || outputs > 0x7fffffff)
    TVM_FFI_ICHECK(false) << "projection shape is outside the operator limit";
  return hc_projection::project_chunk(static_cast<int>(rows), static_cast<int>(outputs), width);
}

void hc_chunked_project(TensorView values, TensorView weight, TensorView output,
                        TensorView workspace) {
  TVM_FFI_ICHECK(values.ndim() == 2 && values.size(0) >= 2 && values.size(0) <= 32768)
      << "hc_chunked_project rows must be in the measured range 2..32768";
  int rows = values.size(0);
  TVM_FFI_ICHECK(weight.ndim() == 2 && weight.size(0) == mixes && weight.size(1) == flat_width);
  tensor_contract(values, values, kDLFloat, 32, {rows, flat_width});
  tensor_contract(weight, values, kDLFloat, 32, {mixes, flat_width});
  tensor_contract(output, values, kDLFloat, 32, {rows, mixes});
  int chunk = hc_projection::released_chunk(rows);
  TVM_FFI_ICHECK(chunk > 0) << "HC row count has no measured released layout";
  int64_t needed = hc_projection::chunk_workspace_bytes(rows, flat_width, chunk);
  TVM_FFI_ICHECK(workspace.device().device_type == kDLCUDA);
  TVM_FFI_ICHECK(workspace.ndim() == 1 && workspace.dtype().code == kDLUInt &&
                 workspace.dtype().bits == 8 && workspace.dtype().lanes == 1);
  TVM_FFI_ICHECK(workspace.size(0) == 0 || workspace.size(0) >= needed);
  TVM_FFI_ICHECK(workspace.size(0) == 0 ||
                 reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0);
  if (workspace.size(0))
    separate({values, weight, output, workspace});
  else
    separate({values, weight, output});
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  hc_projection::multiply_released_chunked_parallel(
      static_cast<const float *>(values.data_ptr()), static_cast<const float *>(weight.data_ptr()),
      static_cast<float *>(output.data_ptr()), rows, mixes, flat_width, flat_width, mixes, chunk,
      workspace.size(0) ? workspace.data_ptr() : nullptr, workspace.size(0), stream);
  check_cuda(cudaGetLastError());
}

void bf16_linear(TensorView values, TensorView weight, TensorView output) {
  TVM_FFI_ICHECK(values.ndim() == 2 && values.size(0) >= 0 && values.size(0) <= 32768);
  TVM_FFI_ICHECK(weight.ndim() == 2 && weight.size(0) >= 1);
  int rows = values.size(0), width = values.size(1), outputs = weight.size(0);
  TVM_FFI_ICHECK(weight.size(1) == width);
  tensor_contract(values, values, kDLBfloat, 16, {rows, width});
  tensor_contract(weight, values, kDLBfloat, 16, {outputs, width});
  tensor_contract(output, values, kDLBfloat, 16, {rows, outputs});
  separate({values, weight, output});
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  hc_projection::bf16_multiply(values.device().device_id, values.data_ptr(), weight.data_ptr(),
                               output.data_ptr(), rows, outputs, width, stream);
  check_cuda(cudaGetLastError());
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
