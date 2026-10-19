#pragma once

#include "common.cuh"
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

extern std::atomic<int64_t> allocated_storage;
void cuda_check(cudaError_t status);

inline std::atomic<int64_t> scratch_live_bytes{0};
inline std::mutex scratch_pool_mutex;
inline std::map<int, cudaMemPool_t> scratch_pools;

inline cudaMemPool_t scratch_pool(int device) {
  std::lock_guard<std::mutex> lock(scratch_pool_mutex);
  auto found = scratch_pools.find(device);
  if (found != scratch_pools.end())
    return found->second;
  cudaMemPoolProps properties{};
  properties.allocType = cudaMemAllocationTypePinned;
  properties.handleTypes = cudaMemHandleTypeNone;
  properties.location.type = cudaMemLocationTypeDevice;
  properties.location.id = device;
  cudaMemPool_t pool;
  cuda_check(cudaMemPoolCreate(&pool, &properties));
  uint64_t retained = uint64_t{1} << 30;
  cuda_check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &retained));
  scratch_pools.emplace(device, pool);
  return pool;
}

inline int64_t scratch_reserved_bytes() {
  std::lock_guard<std::mutex> lock(scratch_pool_mutex);
  uint64_t reserved = 0;
  for (auto [device, pool] : scratch_pools) {
    tvm::ffi::CUDADeviceGuard guard(device);
    uint64_t current = 0;
    cuda_check(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &current));
    reserved += current;
  }
  return static_cast<int64_t>(reserved);
}

struct ScratchAllocator {
  void AllocData(DLTensor *tensor) {
    tvm::ffi::CUDADeviceGuard guard(tensor->device.device_id);
    auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, tensor->device.device_id));
    cudaStreamCaptureStatus capture;
    cuda_check(cudaStreamIsCapturing(stream, &capture));
    TVM_FFI_ICHECK(capture == cudaStreamCaptureStatusNone)
        << "scratch buffers must be allocated before graph capture";
    auto bytes = std::max<size_t>(tvm::ffi::GetDataSize(*tensor), 1);
    cuda_check(cudaMallocFromPoolAsync(&tensor->data, bytes, scratch_pool(tensor->device.device_id),
                                       stream));
    allocated_storage.fetch_add(bytes, std::memory_order_relaxed);
    scratch_live_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }

  void FreeData(DLTensor *tensor) {
    tvm::ffi::CUDADeviceGuard guard(tensor->device.device_id);
    auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, tensor->device.device_id));
    cudaFreeAsync(tensor->data, stream);
    auto bytes = std::max<size_t>(tvm::ffi::GetDataSize(*tensor), 1);
    allocated_storage.fetch_sub(bytes, std::memory_order_relaxed);
    scratch_live_bytes.fetch_sub(bytes, std::memory_order_relaxed);
  }
};
