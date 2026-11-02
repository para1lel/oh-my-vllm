#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

#include <atomic>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

using tvm::ffi::Shape;
using tvm::ffi::String;
using tvm::ffi::Tensor;
using tvm::ffi::TensorView;

namespace paged {

void check_cuda(cudaError_t status) {
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

template <class Function> Function entry(const char *name) {
  void *pointer = nullptr;
  cudaDriverEntryPointQueryResult status;
  check_cuda(cudaGetDriverEntryPointByVersion(name, &pointer, 12000, cudaEnableDefault, &status));
  TVM_FFI_ICHECK(status == cudaDriverEntryPointSuccess && pointer != nullptr) << name;
  return reinterpret_cast<Function>(pointer);
}

struct Driver {
  decltype(&cuMemAddressReserve) reserve = entry<decltype(reserve)>("cuMemAddressReserve");
  decltype(&cuMemAddressFree) free = entry<decltype(free)>("cuMemAddressFree");
  decltype(&cuMemGetAllocationGranularity) granularity =
      entry<decltype(granularity)>("cuMemGetAllocationGranularity");
  decltype(&cuMemCreate) create = entry<decltype(create)>("cuMemCreate");
  decltype(&cuMemRelease) release = entry<decltype(release)>("cuMemRelease");
  decltype(&cuMemMap) map = entry<decltype(map)>("cuMemMap");
  decltype(&cuMemUnmap) unmap = entry<decltype(unmap)>("cuMemUnmap");
  decltype(&cuMemSetAccess) access = entry<decltype(access)>("cuMemSetAccess");
  decltype(&cuGetErrorString) error_string = entry<decltype(error_string)>("cuGetErrorString");
};

Driver &driver() {
  static Driver api;
  return api;
}

void check_driver(CUresult status) {
  if (status == CUDA_SUCCESS)
    return;
  const char *message = "CUDA driver failure";
  driver().error_string(status, &message);
  TVM_FFI_THROW(RuntimeError) << message << " (driver status " << int(status) << ")";
}

size_t multiply(size_t left, size_t right) {
  TVM_FFI_ICHECK(right == 0 || left <= size_t(INT64_MAX) / right) << "paged size overflow";
  return left * right;
}

size_t round_up(size_t bytes, size_t alignment) {
  TVM_FFI_ICHECK(alignment > 0 && bytes <= size_t(INT64_MAX) - (alignment - 1));
  return (bytes + alignment - 1) / alignment * alignment;
}

cudaStream_t eager_stream(int device) {
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
  cudaStreamCaptureStatus capture;
  check_cuda(cudaStreamIsCapturing(stream, &capture));
  TVM_FFI_ICHECK(capture == cudaStreamCaptureStatusNone)
      << "paged storage mapping and release do not support CUDA capture";
  return stream;
}

struct Mapping {
  CUdeviceptr address;
  size_t bytes;
  CUmemGenericAllocationHandle handle = 0;
  bool owns_handle = false;
  bool mapped = false;
};

struct Slot {
  size_t bytes = 0;
  cudaEvent_t ready = nullptr;
  bool failed = false;
  std::vector<Mapping> mappings;
};

struct Storage {
  CUdeviceptr address = 0;
  int device;
  DLDataType dtype;
  size_t granularity, row_bytes, slot_bytes, reserved_bytes;
  std::vector<int64_t> shape, strides;
  std::vector<Slot> slots;
  bool pending_cleanup = false;
};

struct Registry {
  std::mutex mutex;
  std::map<CUdeviceptr, std::shared_ptr<Storage>> storage;
  std::atomic<int64_t> physical_bytes{0};
  size_t pending_cleanup = 0;
};

Registry &registry() {
  // Failed cleanup must retain its handles and VA, including during interpreter
  // shutdown.
  static auto *value = new Registry;
  return *value;
}

CUmemAllocationProp properties(int device) {
  CUmemAllocationProp value{};
  value.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  value.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  value.location.id = device;
  return value;
}

Storage &owned(TensorView tensor) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA && tensor.byte_offset() == 0);
  auto iterator = registry().storage.find(reinterpret_cast<CUdeviceptr>(tensor.data_ptr()));
  TVM_FFI_ICHECK(iterator != registry().storage.end()) << "not a paged storage allocation";
  auto &state = *iterator->second;
  auto dtype = tensor.dtype();
  TVM_FFI_ICHECK(tensor.device().device_id == state.device && dtype.code == state.dtype.code &&
                 dtype.bits == state.dtype.bits && dtype.lanes == state.dtype.lanes &&
                 tensor.ndim() == int(state.shape.size()));
  for (int dimension = 0; dimension < tensor.ndim(); ++dimension) {
    TVM_FFI_ICHECK(tensor.size(dimension) == state.shape[dimension] &&
                   tensor.stride(dimension) == state.strides[dimension])
        << "paged operations require the full allocation shape and strides";
  }
  return state;
}

Slot &checked_slot(Storage &state, int64_t slot) {
  TVM_FFI_ICHECK(slot >= 0 && uint64_t(slot) < state.slots.size()) << "invalid paged slot";
  return state.slots[slot];
}

void free_slot(Slot &slot) {
  for (auto &mapping : slot.mappings) {
    if (mapping.mapped) {
      check_driver(driver().unmap(mapping.address, mapping.bytes));
      mapping.mapped = false;
    }
    if (mapping.owns_handle) {
      check_driver(driver().release(mapping.handle));
      mapping.owns_handle = false;
      registry().physical_bytes.fetch_sub(mapping.bytes, std::memory_order_relaxed);
    }
  }
  if (slot.ready) {
    check_cuda(cudaEventDestroy(slot.ready));
    slot.ready = nullptr;
  }
  slot.mappings.clear();
  slot.bytes = 0;
  slot.failed = false;
}

bool cleanup(Storage &state) {
  tvm::ffi::CUDADeviceGuard guard(state.device);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, state.device));
  cudaStreamCaptureStatus capture;
  auto status = cudaStreamIsCapturing(stream, &capture);
  if (status == cudaErrorStreamCaptureImplicit)
    return false;
  check_cuda(status);
  if (capture != cudaStreamCaptureStatusNone)
    return false;
  check_cuda(cudaDeviceSynchronize());
  for (auto &slot : state.slots)
    free_slot(slot);
  check_driver(driver().free(state.address, state.reserved_bytes));
  return true;
}

void collect_pending(int device) noexcept {
  if (registry().pending_cleanup == 0)
    return;
  for (auto iterator = registry().storage.begin(); iterator != registry().storage.end();) {
    if (iterator->second->pending_cleanup && iterator->second->device == device) {
      try {
        if (cleanup(*iterator->second)) {
          --registry().pending_cleanup;
          iterator = registry().storage.erase(iterator);
          continue;
        }
      } catch (const std::exception &error) {
        std::fprintf(stderr, "paged CUDA cleanup failed; storage retained for retry: %s\n",
                     error.what());
      } catch (...) {
        std::fprintf(stderr, "paged CUDA cleanup failed; storage retained for retry\n");
      }
    }
    ++iterator;
  }
}

struct Allocator {
  std::shared_ptr<Storage> state;

  void AllocData(DLTensor *tensor) {
    tvm::ffi::CUDADeviceGuard guard(state->device);
    std::lock_guard<std::mutex> lock(registry().mutex);
    eager_stream(state->device);
    collect_pending(state->device);
    // Prepare the registry node before reserving VA so host allocation cannot
    // leak VA.
    std::map<CUdeviceptr, std::shared_ptr<Storage>> pending;
    pending.emplace(0, state);
    check_driver(
        driver().reserve(&state->address, state->reserved_bytes, state->granularity, 0, 0));
    auto node = pending.extract(0);
    node.key() = state->address;
    registry().storage.insert(std::move(node));
    tensor->data = reinterpret_cast<void *>(state->address);
  }

  void FreeData(DLTensor *) noexcept {
    try {
      std::lock_guard<std::mutex> lock(registry().mutex);
      state->pending_cleanup = true;
      ++registry().pending_cleanup;
    } catch (const std::exception &error) {
      std::fprintf(stderr, "paged CUDA cleanup failed; storage retained: %s\n", error.what());
    } catch (...) {
      std::fprintf(stderr, "paged CUDA cleanup failed; storage retained\n");
    }
  }
};

} // namespace paged

Tensor allocate(Shape shape, String dtype_name, int64_t device) {
  using namespace paged;
  TVM_FFI_ICHECK(device >= 0 && device <= INT32_MAX);
  TVM_FFI_ICHECK(shape.size() >= 2 && shape.size() <= 32);
  auto dtype = tvm::ffi::StringToDLDataType(dtype_name);
  TVM_FFI_ICHECK(dtype.lanes == 1 &&
                 ((dtype.code == kDLInt || dtype.code == kDLUInt || dtype.code == kDLFloat) ||
                  (dtype.code == kDLBfloat && dtype.bits == 16)) &&
                 (dtype.bits == 8 || dtype.bits == 16 || dtype.bits == 32 || dtype.bits == 64))
      << "paged storage requires a scalar byte-aligned numeric dtype";
  size_t logical_bytes = dtype.bits / 8;
  for (auto extent : shape) {
    TVM_FFI_ICHECK(extent > 0) << "paged dimensions must be positive";
    logical_bytes = multiply(logical_bytes, extent);
  }
  tvm::ffi::CUDADeviceGuard guard(static_cast<int>(device));
  eager_stream(device);
  auto state = std::make_shared<Storage>();
  state->device = device;
  state->dtype = dtype;
  state->shape.assign(shape.begin(), shape.end());
  auto prop = properties(device);
  check_driver(driver().granularity(&state->granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  state->strides.resize(shape.size());
  size_t elements = 1;
  for (size_t i = shape.size(); i > 1; --i) {
    state->strides[i - 1] = elements;
    elements = multiply(elements, shape[i - 1]);
  }
  state->row_bytes = multiply(state->strides[1], dtype.bits / 8);
  state->slot_bytes = round_up(multiply(elements, dtype.bits / 8), state->granularity);
  state->strides[0] = state->slot_bytes / (dtype.bits / 8);
  state->reserved_bytes = multiply(state->slot_bytes, shape[0]);
  state->slots.resize(shape[0]);
  return Tensor::FromNDAllocStrided(Allocator{state}, shape, Shape(state->strides), dtype,
                                    DLDevice{kDLCUDA, static_cast<int>(device)});
}

void commit_rows(TensorView tensor, int64_t slot_index, int64_t stop) {
  using namespace paged;
  std::lock_guard<std::mutex> lock(registry().mutex);
  auto &state = owned(tensor);
  auto &slot = checked_slot(state, slot_index);
  TVM_FFI_ICHECK(stop >= 0 && stop <= state.shape[1]) << "invalid paged row range";
  TVM_FFI_ICHECK(!slot.failed) << "release the slot after a failed mapping operation";
  tvm::ffi::CUDADeviceGuard guard(state.device);
  auto stream = eager_stream(state.device);
  collect_pending(state.device);
  auto bytes = round_up(multiply(stop, state.row_bytes), state.granularity);
  if (slot.ready)
    check_cuda(cudaStreamWaitEvent(stream, slot.ready, 0));
  if (bytes <= slot.bytes)
    return;
  try {
    if (!slot.ready)
      check_cuda(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming));
    slot.mappings.push_back(
        {state.address + slot_index * state.slot_bytes + slot.bytes, bytes - slot.bytes});
    auto &mapping = slot.mappings.back();
    auto prop = properties(state.device);
    check_driver(driver().create(&mapping.handle, mapping.bytes, &prop, 0));
    mapping.owns_handle = true;
    registry().physical_bytes.fetch_add(mapping.bytes, std::memory_order_relaxed);
    check_driver(driver().map(mapping.address, mapping.bytes, 0, mapping.handle, 0));
    mapping.mapped = true;
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check_driver(driver().access(mapping.address, mapping.bytes, &access, 1));
    check_cuda(
        cudaMemsetAsync(reinterpret_cast<void *>(mapping.address), 0, mapping.bytes, stream));
    check_cuda(cudaEventRecord(slot.ready, stream));
    slot.bytes = bytes;
  } catch (...) {
    slot.failed = true;
    throw;
  }
}

void release_slot(TensorView tensor, int64_t slot_index) {
  using namespace paged;
  std::lock_guard<std::mutex> lock(registry().mutex);
  auto &state = owned(tensor);
  auto &slot = checked_slot(state, slot_index);
  tvm::ffi::CUDADeviceGuard guard(state.device);
  eager_stream(state.device);
  collect_pending(state.device);
  try {
    check_cuda(cudaDeviceSynchronize());
    free_slot(slot);
  } catch (...) {
    slot.failed = true;
    throw;
  }
}

int64_t mapped_bytes() { return paged::registry().physical_bytes.load(std::memory_order_relaxed); }
