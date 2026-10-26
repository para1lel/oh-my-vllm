#pragma once

#include "collective_contracts.cuh"
#include <algorithm>
#include <atomic>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

inline void collective_cuda(cudaError_t status) {
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

inline void collective_nccl(ncclResult_t status) {
  TVM_FFI_ICHECK(status == ncclSuccess) << ncclGetErrorString(status);
}

inline cudaStream_t collective_stream() {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, 0));
}

inline void collective_outside_capture(cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  collective_cuda(cudaStreamIsCapturing(stream, &status));
  TVM_FFI_ICHECK(status == cudaStreamCaptureStatusNone)
      << "collective setup and polling must be outside capture";
}

struct CollectiveState;
struct CollectivePlanState;

struct CollectiveGraphLease {
  std::shared_ptr<CollectivePlanState> plan;
  cudaUserObject_t object = nullptr;
  unsigned long long capture_id = 0;
  bool attached = false;
  std::atomic<bool> done{false};
};

struct CollectivePlanState {
  std::shared_ptr<CollectiveState> owner;
  String kind;
  Array<Tensor> tensors, buffers;
  Shape directions, peers, staging;
  cudaEvent_t event = nullptr;
  cudaStream_t stream = nullptr;
  bool active = false, recorded = false, warmed = false, capturing = false;
  CollectiveGraphLease *graph = nullptr;

  ~CollectivePlanState() {
    if (event)
      cudaEventDestroy(event);
  }
};

struct CollectiveState {
  std::mutex mutex;
  ncclComm_t comm = nullptr;
  int world, rank;
  bool initialized = false, failed = false, capture_pending = false;
  cudaEvent_t completion = nullptr;
  bool completion_recorded = false;
  std::shared_ptr<CollectivePlanState> enqueue;
  std::vector<std::shared_ptr<CollectivePlanState>> flights;
  std::vector<std::unique_ptr<CollectiveGraphLease>> graphs;

  ~CollectiveState() {
    if (completion)
      cudaEventDestroy(completion);
  }
};

class CollectiveOwnerObj : public tvm::ffi::Object {
public:
  std::shared_ptr<CollectiveState> state;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("oh_my_vllm.CollectiveOwner", CollectiveOwnerObj,
                                    tvm::ffi::Object);
};

class CollectiveOwner : public tvm::ffi::ObjectRef {
public:
  explicit CollectiveOwner(tvm::ffi::ObjectPtr<CollectiveOwnerObj> object)
      : ObjectRef(std::move(object)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CollectiveOwner, tvm::ffi::ObjectRef,
                                                CollectiveOwnerObj);
};

class CollectivePlanObj : public tvm::ffi::Object {
public:
  std::shared_ptr<CollectivePlanState> state;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("oh_my_vllm.CollectivePlan", CollectivePlanObj,
                                    tvm::ffi::Object);
};

class CollectivePlan : public tvm::ffi::ObjectRef {
public:
  explicit CollectivePlan(tvm::ffi::ObjectPtr<CollectivePlanObj> object)
      : ObjectRef(std::move(object)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CollectivePlan, tvm::ffi::ObjectRef,
                                                CollectivePlanObj);
};

struct CollectiveRoots {
  std::mutex mutex;
  std::vector<std::shared_ptr<CollectiveState>> owners;
};

inline CollectiveRoots &collective_roots() {
  // Process teardown owns resources that were not explicitly drained and aborted.
  static auto *roots = new CollectiveRoots;
  return *roots;
}

inline void collective_unroot(const std::shared_ptr<CollectiveState> &state) {
  if (state->comm || state->enqueue || !state->flights.empty() || !state->graphs.empty() ||
      state->completion_recorded)
    return;
  auto &roots = collective_roots();
  std::lock_guard<std::mutex> lock(roots.mutex);
  roots.owners.erase(std::remove(roots.owners.begin(), roots.owners.end(), state),
                     roots.owners.end());
}

inline ncclResult_t collective_async(CollectiveState &state) {
  TVM_FFI_ICHECK(state.comm) << "communicator is aborted";
  ncclResult_t status = ncclSuccess;
  auto result = ncclCommGetAsyncError(state.comm, &status);
  if (result != ncclSuccess || (status != ncclSuccess && status != ncclInProgress)) {
    state.failed = true;
    collective_nccl(result != ncclSuccess ? result : status);
  }
  return status;
}

inline void collective_ready(CollectiveState &state) {
  TVM_FFI_ICHECK(!state.failed) << "communicator failed; abort or terminate the rank";
  auto status = collective_async(state);
  TVM_FFI_ICHECK(status == ncclSuccess && state.initialized && !state.enqueue &&
                 !state.capture_pending)
      << "communicator needs enqueue polling, finish_enqueue, or completion polling";
}

inline void collective_reap_graphs(CollectiveState &state) {
  auto end = std::remove_if(state.graphs.begin(), state.graphs.end(), [](auto &lease) {
    if (!lease->done.load(std::memory_order_acquire))
      return false;
    lease->plan->graph = nullptr;
    return true;
  });
  state.graphs.erase(end, state.graphs.end());
}

inline void CUDART_CB collective_graph_done(void *pointer) {
  // CUDA callbacks cannot release tensors or call CUDA.
  static_cast<CollectiveGraphLease *>(pointer)->done.store(true, std::memory_order_release);
}

inline void collective_pin_image() {
  static void *image = [] {
    Dl_info info;
    TVM_FFI_ICHECK(dladdr(reinterpret_cast<void *>(&collective_graph_done), &info));
    auto handle = dlopen(info.dli_fname, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    TVM_FFI_ICHECK(handle) << "cannot retain the native collective image";
    return handle;
  }();
  (void)image;
}

inline void collective_record(const std::shared_ptr<CollectivePlanState> &plan) {
  auto status = cudaEventRecord(plan->event, plan->stream);
  if (status != cudaSuccess)
    plan->owner->failed = true;
  collective_cuda(status);
  plan->recorded = true;
}
