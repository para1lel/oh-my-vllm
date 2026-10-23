#pragma once

#include "../../runtime/cuda/contracts.cuh"
#include <atomic>
#include <climits>
#include <fcntl.h>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

std::atomic<int64_t> engram_pinned_storage{0};

struct EngramFile {
  int descriptor;

  ~EngramFile() {
    if (descriptor >= 0)
      close(descriptor);
  }
};

struct EngramMapping {
  void *host = nullptr, *device_pointer = nullptr;
  int64_t bytes;
  int device;
  bool reserved = false, locked = false, registered = false, accessible = false;

  EngramMapping(int64_t size, int ordinal) : bytes(size), device(ordinal) {}

  ~EngramMapping() {
    int previous = device;
    if (registered || accessible) {
      // A failed CUDA context keeps its mapping until process teardown.
      if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess)
        return;
      auto completed = cudaDeviceSynchronize();
      auto released = completed == cudaSuccess && registered ? cudaHostUnregister(host) : completed;
      cudaSetDevice(previous);
      if (released != cudaSuccess)
        return;
    }
    if (locked)
      munlock(host, bytes);
    if (host)
      munmap(host, bytes);
    if (reserved)
      engram_pinned_storage.fetch_sub(bytes, std::memory_order_relaxed);
  }
};

struct EngramAllocator {
  std::shared_ptr<EngramMapping> mapping;

  void AllocData(DLTensor *tensor) { tensor->data = mapping->device_pointer; }

  void FreeData(DLTensor *) { mapping.reset(); }
};

} // namespace

int64_t engram_pinned_bytes() { return engram_pinned_storage.load(std::memory_order_relaxed); }

tvm::ffi::Tensor engram_register(int64_t descriptor, int64_t size, int64_t limit) {
  TVM_FFI_ICHECK(descriptor >= 0 && descriptor <= INT_MAX && size > 0 && size <= limit);
  long page = sysconf(_SC_PAGESIZE);
  TVM_FFI_ICHECK(page > 0 && size <= INT64_MAX - (page - 1));
  int64_t mapped = ((size + page - 1) / page) * page;
  TVM_FFI_ICHECK(mapped <= limit);
  EngramFile file{fcntl(static_cast<int>(descriptor), F_DUPFD_CLOEXEC, 0)};
  TVM_FFI_ICHECK(file.descriptor >= 0) << "cannot duplicate shared weight descriptor";
  int seals = fcntl(file.descriptor, F_GET_SEALS);
  constexpr int required = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
  TVM_FFI_ICHECK(seals >= 0 && (seals & required) == required)
      << "shared weights must be sealed before registration";
  struct stat info{};
  TVM_FFI_ICHECK(fstat(file.descriptor, &info) == 0 && S_ISREG(info.st_mode) &&
                 info.st_size == mapped);
  tvm::ffi::CUDADeviceGuard guard(0);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, 0));
  cudaStreamCaptureStatus capture;
  check_cuda(cudaStreamIsCapturing(stream, &capture));
  TVM_FFI_ICHECK(capture == cudaStreamCaptureStatusNone)
      << "shared weight registration cannot occur during capture";
  int supported = 0, readonly = 0, host_tables = 0, pageable = 0;
  check_cuda(cudaDeviceGetAttribute(&supported, cudaDevAttrHostRegisterSupported, 0));
  check_cuda(cudaDeviceGetAttribute(&readonly, cudaDevAttrHostRegisterReadOnlySupported, 0));
  check_cuda(
      cudaDeviceGetAttribute(&host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0));
  check_cuda(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0));
  bool use_registration = supported && readonly;
  TVM_FFI_ICHECK(use_registration || pageable)
      << "device does not support read-only shared host memory";
  auto mapping = std::make_shared<EngramMapping>(mapped, 0);
  int64_t current = engram_pinned_bytes();
  do {
    TVM_FFI_ICHECK(current <= limit - mapped) << "shared weight pinned budget exceeded";
  } while (!engram_pinned_storage.compare_exchange_weak(current, current + mapped,
                                                        std::memory_order_relaxed));
  mapping->reserved = true;
  mapping->host = mmap(nullptr, mapped, PROT_READ, MAP_SHARED, file.descriptor, 0);
  if (mapping->host == MAP_FAILED) {
    mapping->host = nullptr;
    TVM_FFI_THROW(RuntimeError) << "cannot map shared weights";
  }
  if (host_tables || !use_registration) {
    TVM_FFI_ICHECK(mlock(mapping->host, mapped) == 0) << "cannot pin shared weights";
    mapping->locked = true;
  }
  check_cuda(cudaDeviceSynchronize());
  if (use_registration) {
    check_cuda(
        cudaHostRegister(mapping->host, mapped, cudaHostRegisterReadOnly | cudaHostRegisterMapped));
    mapping->registered = true;
    check_cuda(cudaHostGetDevicePointer(&mapping->device_pointer, mapping->host, 0));
  } else {
    check_cuda(cudaMemAdvise(mapping->host, mapped, cudaMemAdviseSetPreferredLocation,
                             cudaMemLocation{cudaMemLocationTypeHost, 0}));
    check_cuda(cudaMemAdvise(mapping->host, mapped, cudaMemAdviseSetAccessedBy,
                             cudaMemLocation{cudaMemLocationTypeDevice, 0}));
    mapping->device_pointer = mapping->host;
  }
  mapping->accessible = true;
  return tvm::ffi::Tensor::FromNDAlloc(EngramAllocator{mapping}, tvm::ffi::Shape{size},
                                       DLDataType{kDLUInt, 8, 1}, DLDevice{kDLCUDA, 0});
}
