#include "common.cuh"
#include <atomic>
#include <vector>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

using tvm::ffi::Shape;
using tvm::ffi::String;
using tvm::ffi::Tensor;
using tvm::ffi::TensorView;

std::atomic<int64_t> allocated_storage{0};

void cuda_check(cudaError_t status) {
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

struct DeviceAllocator {
  void AllocData(DLTensor *tensor) {
    tvm::ffi::CUDADeviceGuard guard(tensor->device.device_id);
    auto bytes = tvm::ffi::GetDataSize(*tensor);
    cuda_check(cudaMalloc(&tensor->data, bytes ? bytes : 1));
    allocated_storage.fetch_add(bytes ? bytes : 1, std::memory_order_relaxed);
  }

  void FreeData(DLTensor *tensor) {
    tvm::ffi::CUDADeviceGuard guard(tensor->device.device_id);
    cudaFree(tensor->data);
    auto bytes = tvm::ffi::GetDataSize(*tensor);
    allocated_storage.fetch_sub(bytes ? bytes : 1, std::memory_order_relaxed);
  }
};

int64_t allocated_bytes() { return allocated_storage.load(std::memory_order_relaxed); }

Tensor allocate(Shape shape, String dtype, int64_t device) {
  TVM_FFI_ICHECK(device >= 0 && device <= INT32_MAX);
  for (auto extent : shape)
    TVM_FFI_ICHECK(extent >= 0);
  return Tensor::FromNDAlloc(DeviceAllocator{}, shape, tvm::ffi::StringToDLDataType(dtype),
                             DLDevice{kDLCUDA, static_cast<int>(device)});
}

void copy_bytes(TensorView destination, TensorView source) {
  TVM_FFI_ICHECK(destination.IsContiguous() && source.IsContiguous());
  TVM_FFI_ICHECK(tvm::ffi::GetDataSize(destination) == tvm::ffi::GetDataSize(source));
  auto device =
      destination.device().device_type == kDLCUDA ? destination.device() : source.device();
  TVM_FFI_ICHECK(device.device_type == kDLCUDA);
  for (auto tensor : {source, destination})
    TVM_FFI_ICHECK(
        tensor.device().device_type == kDLCPU ||
        (tensor.device().device_type == kDLCUDA && tensor.device().device_id == device.device_id));
  tvm::ffi::CUDADeviceGuard guard(device.device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device.device_id));
  cuda_check(cudaMemcpyAsync(destination.data_ptr(), source.data_ptr(),
                             tvm::ffi::GetDataSize(source), cudaMemcpyDefault, stream));
  // Synchronize only when the destination is on the CPU: the caller reads it
  // immediately after returning.  Device-destination copies are consumed by
  // downstream kernels on the same stream and do not need a CPU stall.
  if (destination.device().device_type == kDLCPU)
    cuda_check(cudaStreamSynchronize(stream));
}

void synchronize(int64_t device) {
  tvm::ffi::CUDADeviceGuard guard(static_cast<int>(device));
  cuda_check(cudaDeviceSynchronize());
}

int64_t l2_bytes() {
  tvm::ffi::CUDADeviceGuard guard(0);
  int bytes = 0;
  cuda_check(cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, 0));
  return bytes;
}

__global__ void evict_kernel(const uint32_t *input, uint32_t *output, int64_t elements) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    uint32_t value;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(input + index));
    output[index] = value + 1;
  }
}

void evict(TensorView input, TensorView output) {
  TVM_FFI_ICHECK(input.IsContiguous() && output.IsContiguous());
  TVM_FFI_ICHECK(input.dtype().code == kDLUInt && input.dtype().bits == 32 &&
                 input.dtype().lanes == 1);
  TVM_FFI_ICHECK(output.dtype() == input.dtype() && output.numel() == input.numel());
  TVM_FFI_ICHECK(input.device().device_type == kDLCUDA && output.device().device_type == kDLCUDA &&
                 input.device().device_id == output.device().device_id);
  TVM_FFI_ICHECK(input.data_ptr() != output.data_ptr());
  tvm::ffi::CUDADeviceGuard guard(input.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, input.device().device_id));
  evict_kernel<<<(input.numel() + 255) / 256, 256, 0, stream>>>(
      static_cast<const uint32_t *>(input.data_ptr()), static_cast<uint32_t *>(output.data_ptr()),
      input.numel());
  cuda_check(cudaGetLastError());
}

Tensor view(Tensor source, Shape shape, int64_t offset) {
  TVM_FFI_ICHECK(source.IsContiguous() && offset >= 0 && offset <= source.numel());
  std::vector<int64_t> strides(shape.size());
  int64_t elements = 1;
  for (int64_t i = shape.size(); i > 0; --i) {
    int64_t extent = shape[i - 1];
    TVM_FFI_ICHECK(extent >= 0);
    strides[i - 1] = elements;
    TVM_FFI_ICHECK(extent == 0 || elements <= INT64_MAX / extent);
    elements *= extent;
  }
  TVM_FFI_ICHECK(elements <= source.numel() - offset);
  return source.as_strided(shape, Shape(strides), offset);
}

void zero(TensorView value) {
  TVM_FFI_ICHECK(value.IsContiguous() && value.device().device_type == kDLCUDA);
  tvm::ffi::CUDADeviceGuard guard(value.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, value.device().device_id));
  cuda_check(cudaMemsetAsync(value.data_ptr(), 0, tvm::ffi::GetDataSize(value), stream));
}

int64_t free_bytes() {
  tvm::ffi::CUDADeviceGuard guard(0);
  size_t available, total;
  cuda_check(cudaMemGetInfo(&available, &total));
  return static_cast<int64_t>(available);
}

__global__ void gather_rows_kernel(uint8_t *destination, const uint8_t *source, const int *indices,
                                   int rows, int source_rows, int64_t row_bytes) {
  int row = blockIdx.x;
  int index = indices[row];
  if (index < 0 || index >= source_rows) {
    asm volatile("trap;");
    return;
  }
  // Use 16-byte vectorized loads when the row is 16-byte aligned and a
  // multiple of 16 bytes.  All model tensor rows satisfy this (BF16 widths
  // 5120, 6144, 10240 are all divisible by 8 elements = 16 bytes).
  if ((row_bytes & 15) == 0 && (reinterpret_cast<uintptr_t>(destination) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(source) & 15) == 0) {
    int64_t vec_count = row_bytes / 16;
    const uint4 *src4 = reinterpret_cast<const uint4 *>(source) + int64_t(index) * vec_count;
    uint4 *dst4 = reinterpret_cast<uint4 *>(destination) + int64_t(row) * vec_count;
    for (int64_t i = threadIdx.x; i < vec_count; i += blockDim.x)
      dst4[i] = src4[i];
  } else {
    for (int64_t byte = threadIdx.x; byte < row_bytes; byte += blockDim.x)
      destination[int64_t(row) * row_bytes + byte] = source[int64_t(index) * row_bytes + byte];
  }
}

void gather_rows(TensorView destination, TensorView source, TensorView indices) {
  TVM_FFI_ICHECK(source.ndim() > 0 && destination.ndim() == source.ndim());
  TVM_FFI_ICHECK(source.IsContiguous() && destination.IsContiguous() && indices.IsContiguous());
  TVM_FFI_ICHECK(source.device().device_type == kDLCUDA &&
                 destination.device().device_type == kDLCUDA &&
                 indices.device().device_type == kDLCUDA);
  TVM_FFI_ICHECK(source.device().device_id == destination.device().device_id &&
                 source.device().device_id == indices.device().device_id);
  TVM_FFI_ICHECK(source.dtype() == destination.dtype());
  TVM_FFI_ICHECK(indices.dtype().code == kDLInt && indices.dtype().bits == 32 &&
                 indices.dtype().lanes == 1 && indices.ndim() == 1);
  TVM_FFI_ICHECK(source.size(0) > 0 && source.size(0) <= INT32_MAX &&
                 destination.size(0) <= INT32_MAX && indices.size(0) == destination.size(0));
  for (int i = 1; i < source.ndim(); ++i)
    TVM_FFI_ICHECK(source.size(i) == destination.size(i));
  disjoint(source, destination);
  disjoint(indices, destination);
  if (destination.size(0) == 0)
    return;
  tvm::ffi::CUDADeviceGuard guard(source.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, source.device().device_id));
  gather_rows_kernel<<<destination.size(0), 256, 0, stream>>>(
      static_cast<uint8_t *>(destination.data_ptr()),
      static_cast<const uint8_t *>(source.data_ptr()), static_cast<const int *>(indices.data_ptr()),
      destination.size(0), source.size(0), tvm::ffi::GetDataSize(source) / source.size(0));
  cuda_check(cudaGetLastError());
}

__global__ void final_hidden_kernel(const __nv_bfloat16 *value, const __nv_bfloat16 *residual,
                                    const float *weight, __nv_bfloat16 *output) {
  int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  float values[8], square = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int column = threadIdx.x + i * 640;
    values[i] = __bfloat162float(value[row * 5120 + column]) +
                __bfloat162float(residual[row * 5120 + column]);
    square += values[i] * values[i];
  }
  __shared__ float partial[20];
  square = warp_sum(square);
  if (lane == 0)
    partial[warp] = square;
  __syncthreads();
  float total = warp_sum(lane < 20 ? partial[lane] : 0.f);
  float inverse = rsqrtf(total / 5120.f + 1e-6f);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int column = threadIdx.x + i * 640;
    output[row * 5120 + column] = __float2bfloat16_rn(values[i] * inverse * (weight[column] + 1.f));
  }
}

void final_hidden(TensorView value, TensorView residual, TensorView weight, TensorView output) {
  TVM_FFI_ICHECK(value.device().device_type == kDLCUDA && value.ndim() == 2 && value.size(0) > 0 &&
                 value.size(0) <= 32768);
  int rows = value.size(0);
  for (auto tensor : {value, residual, output})
    tensor_contract(tensor, value, kDLBfloat, 16, {rows, 5120});
  tensor_contract(weight, value, kDLFloat, 32, {5120});
  for (auto tensor : {value, residual, weight})
    disjoint(tensor, output);
  tvm::ffi::CUDADeviceGuard guard(value.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, value.device().device_id));
  final_hidden_kernel<<<rows, 640, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(value.data_ptr()),
      static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
      static_cast<const float *>(weight.data_ptr()),
      static_cast<__nv_bfloat16 *>(output.data_ptr()));
  cuda_check(cudaGetLastError());
}
