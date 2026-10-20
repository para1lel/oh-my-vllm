#include "contracts.cuh"
#include "../../primitives/cuda/gather.cuh"
#include "scratch_allocator.cuh"
#include <algorithm>
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

int64_t reserved_bytes() {
  return allocated_bytes() - scratch_live_bytes.load(std::memory_order_relaxed) +
         scratch_reserved_bytes();
}

Tensor allocate(Shape shape, String dtype, int64_t device) {
  TVM_FFI_ICHECK(device >= 0 && device <= INT32_MAX);
  for (auto extent : shape)
    TVM_FFI_ICHECK(extent >= 0);
  return Tensor::FromNDAlloc(DeviceAllocator{}, shape, tvm::ffi::StringToDLDataType(dtype),
                             DLDevice{kDLCUDA, static_cast<int>(device)});
}

Tensor allocate_scratch(Shape shape, String dtype, int64_t device) {
  TVM_FFI_ICHECK(device >= 0 && device <= INT32_MAX);
  for (auto extent : shape)
    TVM_FFI_ICHECK(extent >= 0);
  return Tensor::FromNDAlloc(ScratchAllocator{}, shape, tvm::ffi::StringToDLDataType(dtype),
                             DLDevice{kDLCUDA, static_cast<int>(device)});
}

void reserve_scratch(int64_t requested) {
  TVM_FFI_ICHECK(requested >= 0);
  tvm::ffi::CUDADeviceGuard guard(0);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, 0));
  cudaStreamCaptureStatus capture;
  cuda_check(cudaStreamIsCapturing(stream, &capture));
  TVM_FFI_ICHECK(capture == cudaStreamCaptureStatusNone);
  size_t available, total;
  cuda_check(cudaMemGetInfo(&available, &total));
  uint64_t bytes = std::min<uint64_t>(requested, available / 2);
  if (bytes == 0)
    return;
  auto pool = scratch_pool(0);
  uint64_t threshold = std::max<uint64_t>(uint64_t{1} << 30, bytes);
  cuda_check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  void *storage;
  cuda_check(cudaMallocFromPoolAsync(&storage, bytes, pool, stream));
  cuda_check(cudaFreeAsync(storage, stream));
  cuda_check(cudaStreamSynchronize(stream));
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
