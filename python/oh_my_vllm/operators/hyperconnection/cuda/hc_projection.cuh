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
  int device;

  explicit BlasOwner(int device) : device(device) {
    check_blas(cublasCreate(&handle));
    auto status = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cublasDestroy(handle);
      check_blas(status);
    }
  }

  ~BlasOwner() {
    int previous = 0;
    if (cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device) == cudaSuccess) {
      cublasDestroy(handle);
      cudaSetDevice(previous);
    }
  }
};

inline cublasHandle_t handle(int device, cudaStream_t stream) {
  static thread_local std::unordered_map<int, std::unique_ptr<BlasOwner>> owners;
  auto found = owners.find(device);
  if (found == owners.end()) {
    cudaStreamCaptureStatus capture;
    check_cuda(cudaStreamIsCapturing(stream, &capture));
    TVM_FFI_ICHECK(capture == cudaStreamCaptureStatusNone) << "prepare HC before Graph capture";
    found = owners.emplace(device, std::make_unique<BlasOwner>(device)).first;
  }
  return found->second->handle;
}

inline cublasHandle_t bind(int device, TensorView workspace, cudaStream_t stream) {
  auto blas = handle(device, stream);
  check_blas(cublasSetStream(blas, stream));
  check_blas(cublasSetWorkspace(blas, workspace.data_ptr(), BlasBytes));
  return blas;
}

inline void multiply(cublasHandle_t blas, const float *input, const float *weight, float *output,
                     int rows, int width, int outputs, int logical_rows) {
  int steps = rows / logical_rows;
  float alpha = 1.f, beta = 0.f;
  // A step keeps its source SGEMM shape without copying batch-major inputs.
  for (int step = 0; step < steps; ++step)
    check_blas(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, outputs, logical_rows, width, &alpha,
                           weight, width, input + step * width, steps * width, &beta,
                           output + step * outputs, steps * outputs));
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
  auto blas = bind(residual.device().device_id, workspace, stream);
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
  multiply(blas, values, static_cast<const float *>(weight.data_ptr()),
           static_cast<float *>(mixed.data_ptr()), rows, FlatWidth, Mixes, logical_rows);
  normalize_kernel<<<(rows * Mixes + 255) / 256, 256, 0, stream>>>(
      static_cast<float *>(mixed.data_ptr()), stats, rows);
  check_cuda(cudaGetLastError());
}

} // namespace hc_projection
