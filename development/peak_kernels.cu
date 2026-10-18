#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

void blas_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS status " + std::to_string(int(status)));
}

extern "C" int gpu_details(int64_t *output) {
  cudaDeviceProp properties;
  auto status = cudaGetDeviceProperties(&properties, 0);
  if (status != cudaSuccess)
    return int(status);
  output[0] = properties.multiProcessorCount;
  output[1] = properties.l2CacheSize;
  output[2] = properties.major;
  output[3] = properties.minor;
  output[4] = cublasLtGetVersion();
  return 0;
}

struct Buffer {
  void *data = nullptr;

  explicit Buffer(size_t bytes) { check(cudaMalloc(&data, bytes)); }

  ~Buffer() { cudaFree(data); }

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
};

struct Stream {
  cudaStream_t value = nullptr;

  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }

  ~Stream() { cudaStreamDestroy(value); }
};

struct Event {
  cudaEvent_t value = nullptr;

  Event() { check(cudaEventCreate(&value)); }

  ~Event() { cudaEventDestroy(value); }
};

template <class T> __global__ void fill(T *data, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    data[i] = T(0.125f);
}

__global__ void copy_vectors(uint4 *output, const uint4 *input, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    output[i] = input[i];
}

__host__ __device__ unsigned int pattern(size_t index) {
  unsigned int value = static_cast<unsigned int>(index) + 0x9e3779b9u;
  value = (value ^ (value >> 16)) * 0x85ebca6bu;
  return (value ^ (value >> 13)) * 0xc2b2ae35u;
}

__global__ void initialize_words(unsigned int *data, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x)
    data[i] = pattern(i);
}

template <class T> __global__ void scalar_fma(T *output, int iterations) {
  T values[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    values[i] = T(0.125) + T(threadIdx.x + i) * T(0.00001);
  for (int step = 0; step < iterations; ++step) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if constexpr (sizeof(T) == 4)
        asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                     : "+f"(values[i])
                     : "f"(T(0.999)), "f"(T(0.0001)));
      else
        asm volatile("fma.rn.f64 %0, %0, %1, %2;"
                     : "+d"(values[i])
                     : "d"(T(0.999)), "d"(T(0.0001)));
    }
  }
  T sum = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i)
    sum += values[i];
  output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

template <class Operation>
void time_operation(Operation operation, cudaStream_t stream, double *output) {
  auto begin = std::chrono::steady_clock::now();
  do {
    for (int i = 0; i < 10; ++i)
      operation();
    check(cudaStreamSynchronize(stream));
  } while (std::chrono::steady_clock::now() - begin < std::chrono::milliseconds(200));
  Event start, stop;
  for (int sample = 0; sample < 7; ++sample) {
    check(cudaEventRecord(start.value, stream));
    for (int i = 0; i < 10; ++i)
      operation();
    check(cudaEventRecord(stop.value, stream));
    check(cudaEventSynchronize(stop.value));
    float milliseconds = 0;
    check(cudaEventElapsedTime(&milliseconds, start.value, stop.value));
    output[sample + 1] = milliseconds * 1e-4;
  }
}

struct Matmul {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t input = nullptr, output = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  ~Matmul() {
    if (preference)
      cublasLtMatmulPreferenceDestroy(preference);
    if (output)
      cublasLtMatrixLayoutDestroy(output);
    if (input)
      cublasLtMatrixLayoutDestroy(input);
    if (operation)
      cublasLtMatmulDescDestroy(operation);
    if (handle)
      cublasLtDestroy(handle);
  }
};

bool measure_gemm(int resource, int n, int candidate, cudaStream_t stream, double *result) {
  auto type = resource == 1 ? CUDA_R_8F_E4M3 : resource == 2 ? CUDA_R_16BF : CUDA_R_32F;
  auto compute = resource == 3 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  size_t count = size_t(n) * n;
  size_t bytes = count * (resource == 1 ? 1 : resource == 2 ? 2 : 4);
  constexpr size_t workspace_bytes = 64 * 1024 * 1024;
  Buffer a(bytes), b(bytes), c(count * sizeof(float)), workspace(workspace_bytes),
      scale(sizeof(float));
  if (resource == 1) {
    fill<<<1024, 256, 0, stream>>>(static_cast<__nv_fp8_e4m3 *>(a.data), count);
    fill<<<1024, 256, 0, stream>>>(static_cast<__nv_fp8_e4m3 *>(b.data), count);
  } else if (resource == 2) {
    fill<<<1024, 256, 0, stream>>>(static_cast<__nv_bfloat16 *>(a.data), count);
    fill<<<1024, 256, 0, stream>>>(static_cast<__nv_bfloat16 *>(b.data), count);
  } else {
    fill<<<1024, 256, 0, stream>>>(static_cast<float *>(a.data), count);
    fill<<<1024, 256, 0, stream>>>(static_cast<float *>(b.data), count);
  }
  float alpha = 1, beta = 0;
  check(cudaMemsetAsync(c.data, 0, count * sizeof(float), stream));
  check(cudaMemcpyAsync(scale.data, &alpha, sizeof(alpha), cudaMemcpyHostToDevice, stream));
  Matmul matmul;
  blas_check(cublasLtCreate(&matmul.handle));
  blas_check(cublasLtMatmulDescCreate(&matmul.operation, compute, CUDA_R_32F));
  auto transpose = CUBLAS_OP_T;
  blas_check(cublasLtMatmulDescSetAttribute(matmul.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                            &transpose, sizeof(transpose)));
  if (resource == 1) {
    blas_check(cublasLtMatmulDescSetAttribute(
        matmul.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale.data, sizeof(scale.data)));
    blas_check(cublasLtMatmulDescSetAttribute(
        matmul.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale.data, sizeof(scale.data)));
  }
  blas_check(cublasLtMatrixLayoutCreate(&matmul.input, type, n, n, n));
  blas_check(cublasLtMatrixLayoutCreate(&matmul.output, CUDA_R_32F, n, n, n));
  blas_check(cublasLtMatmulPreferenceCreate(&matmul.preference));
  blas_check(cublasLtMatmulPreferenceSetAttribute(matmul.preference,
                                                  CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_bytes, sizeof(workspace_bytes)));
  cublasLtMatmulHeuristicResult_t algorithms[8];
  int returned = 0;
  blas_check(cublasLtMatmulAlgoGetHeuristic(matmul.handle, matmul.operation, matmul.input,
                                            matmul.input, matmul.output, matmul.output,
                                            matmul.preference, 8, algorithms, &returned));
  if (candidate >= returned || algorithms[candidate].state != CUBLAS_STATUS_SUCCESS)
    return false;
  auto run = [&] {
    blas_check(cublasLtMatmul(matmul.handle, matmul.operation, &alpha, a.data, matmul.input, b.data,
                              matmul.input, &beta, c.data, matmul.output, c.data, matmul.output,
                              &algorithms[candidate].algo, workspace.data, workspace_bytes,
                              stream));
  };
  result[0] = 2.0 * n * n * n;
  time_operation(run, stream, result);
  for (size_t position : {size_t(0), count / 2, count - 1}) {
    float value = 0;
    check(cudaMemcpy(&value, static_cast<float *>(c.data) + position, sizeof(value),
                     cudaMemcpyDeviceToHost));
    if (value != n * 0.015625f)
      throw std::runtime_error("GEMM calibration output is incorrect");
  }
  return true;
}

extern "C" int measure_peak(int resource, int64_t size, int option, double *result, char *error,
                            size_t capacity) {
  try {
    Stream stream;
    cudaDeviceProp properties;
    check(cudaGetDeviceProperties(&properties, 0));
    int blocks = properties.multiProcessorCount * option;
    if (resource == 0) {
      if (size <= 4 * properties.l2CacheSize || size % 16)
        throw std::runtime_error("HBM buffer must exceed four L2 capacities and align to 16 bytes");
      Buffer input(size), output(size);
      initialize_words<<<blocks, 256, 0, stream.value>>>(static_cast<unsigned int *>(input.data),
                                                         size / 4);
      check(cudaMemsetAsync(output.data, 0, size, stream.value));
      result[0] = 2.0 * size;
      time_operation(
          [&] {
            copy_vectors<<<blocks, 256, 0, stream.value>>>(
                static_cast<uint4 *>(output.data), static_cast<uint4 *>(input.data), size / 16);
          },
          stream.value, result);
      for (int64_t position : {int64_t(0), size / 8, size / 4 - 1}) {
        unsigned int value = 0;
        check(cudaMemcpy(&value, static_cast<unsigned int *>(output.data) + position, sizeof(value),
                         cudaMemcpyDeviceToHost));
        if (value != pattern(position))
          throw std::runtime_error("HBM calibration copy is incorrect");
      }
    } else if (resource <= 3) {
      return measure_gemm(resource, int(size), option, stream.value, result) ? 0 : 1;
    } else {
      Buffer output(size_t(blocks) * 256 * sizeof(double));
      result[0] = double(blocks) * 256 * 16 * size * 2;
      time_operation(
          [&] {
            if (resource == 4)
              scalar_fma<<<blocks, 256, 0, stream.value>>>(static_cast<float *>(output.data),
                                                           int(size));
            else
              scalar_fma<<<blocks, 256, 0, stream.value>>>(static_cast<double *>(output.data),
                                                           int(size));
          },
          stream.value, result);
      double value = 0;
      if (resource == 4) {
        float single = 0;
        check(cudaMemcpy(&single, output.data, sizeof(single), cudaMemcpyDeviceToHost));
        value = single;
      } else {
        check(cudaMemcpy(&value, output.data, sizeof(value), cudaMemcpyDeviceToHost));
      }
      if (!std::isfinite(value) || value <= 0)
        throw std::runtime_error("FMA calibration output is incorrect");
    }
    check(cudaGetLastError());
    return 0;
  } catch (const std::exception &exception) {
    std::snprintf(error, capacity, "%s", exception.what());
    return -1;
  }
}
