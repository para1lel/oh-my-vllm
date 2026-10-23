#include "collective_owner.cuh"
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <tvm/ffi/function.h>

inline void collective_copy(const std::shared_ptr<CollectivePlanState> &plan, bool input) {
  for (size_t i = 0; i < plan->tensors.size(); ++i) {
    bool source = plan->kind == "routing" ? plan->directions[i] == 0 : i == 0;
    if (!plan->staging[i] || source != input)
      continue;
    auto original = plan->tensors[i].data_ptr(), staged = plan->buffers[i].data_ptr();
    auto bytes = uint64_t(plan->tensors[i].size(0)) * (plan->tensors[i].dtype().bits / 8);
    auto status = cudaMemcpyAsync(input ? staged : original, input ? original : staged, bytes,
                                  cudaMemcpyDeviceToDevice, plan->stream);
    if (status != cudaSuccess)
      plan->owner->failed = true;
    collective_cuda(status);
  }
}

inline void collective_finish(std::shared_ptr<CollectivePlanState> plan) {
  collective_copy(plan, false);
  if (!plan->capturing)
    collective_record(plan);
  plan->owner->enqueue.reset();
}

inline void collective_storage_check(const CollectivePlanState &plan) {
  TVM_FFI_ICHECK(plan.buffers.size() == plan.tensors.size());
  for (size_t i = 0; i < plan.buffers.size(); ++i) {
    const auto &buffer = plan.buffers[i];
    if (plan.staging[i]) {
      collective_tensor(buffer);
      auto dtype = buffer.dtype();
      TVM_FFI_ICHECK(dtype.code == kDLUInt && dtype.bits == 8 &&
                     buffer.size(0) == plan.staging[i] &&
                     reinterpret_cast<uintptr_t>(buffer.data_ptr()) % 16 == 0)
          << "invalid collective staging capacity or alignment";
      for (const auto &original : plan.tensors)
        collective_disjoint(buffer, original);
    }
    for (size_t j = 0; j < i; ++j)
      collective_disjoint(buffer, plan.buffers[j]);
  }
}

Bytes collective_unique_id() {
  static_assert(sizeof(ncclUniqueId) == 128);
  ncclUniqueId id;
  collective_nccl(ncclGetUniqueId(&id));
  return Bytes(id.internal, sizeof(id.internal));
}

Shape collective_versions() {
  int runtime = 0;
  collective_nccl(ncclGetVersion(&runtime));
  return Shape({NCCL_VERSION_CODE, runtime});
}

Array<String> collective_local_device() {
  int count = 0;
  collective_cuda(cudaGetDeviceCount(&count));
  String uuid = "";
  if (count == 1) {
    cudaDeviceProp properties;
    collective_cuda(cudaGetDeviceProperties(&properties, 0));
    auto id = properties.uuid;
    char text[41] = "GPU-";
    size_t offset = 4;
    for (size_t i = 0; i < sizeof(id.bytes); ++i) {
      if (i == 4 || i == 6 || i == 8 || i == 10)
        text[offset++] = '-';
      std::snprintf(text + offset, 3, "%02x", static_cast<unsigned char>(id.bytes[i]));
      offset += 2;
    }
    uuid = String(text);
  }
  return {String(std::to_string(count)), uuid};
}

Array<String> collective_images() {
  Array<String> result;
  for (auto address :
       {reinterpret_cast<void *>(&ncclGetVersion), reinterpret_cast<void *>(&cudaGetDeviceCount)}) {
    Dl_info info;
    TVM_FFI_ICHECK(dladdr(address, &info) && info.dli_fname);
    result.push_back(String(info.dli_fname));
  }
  return result;
}

CollectiveOwner collective_init(int64_t world, int64_t rank, Bytes unique_id) {
  collective_bounds(world, rank);
  TVM_FFI_ICHECK(unique_id.size() == sizeof(ncclUniqueId)) << "unique ID must contain 128 bytes";
  auto versions = collective_versions();
  TVM_FFI_ICHECK(versions[0] == versions[1]) << "NCCL header and runtime versions differ";
  auto identity = collective_local_device();
  TVM_FFI_ICHECK(identity[0] == "1") << "a rank must see exactly one CUDA device";
  collective_pin_image();
  tvm::ffi::CUDADeviceGuard guard(0);
  collective_outside_capture(collective_stream());
  ncclUniqueId id;
  std::memcpy(id.internal, unique_id.data(), sizeof(id.internal));
  auto state = std::make_shared<CollectiveState>();
  state->world = world;
  state->rank = rank;
  collective_cuda(cudaEventCreateWithFlags(&state->completion, cudaEventDisableTiming));
  auto object = tvm::ffi::make_object<CollectiveOwnerObj>();
  object->state = state;
  {
    auto &roots = collective_roots();
    std::lock_guard<std::mutex> lock(roots.mutex);
    roots.owners.push_back(state);
  }
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;
  auto status = ncclCommInitRankConfig(&state->comm, world, id, rank, &config);
  if (status != ncclSuccess && status != ncclInProgress) {
    state->failed = true;
    collective_nccl(status);
  }
  return CollectiveOwner(std::move(object));
}

String collective_poll_enqueue(CollectiveOwner owner) {
  auto state = owner->state;
  std::lock_guard<std::mutex> lock(state->mutex);
  if (!state->comm)
    return "aborted";
  TVM_FFI_ICHECK(!state->failed) << "communicator failed; abort or terminate the rank";
  if (collective_async(*state) == ncclInProgress)
    return state->initialized ? "pending" : "initializing";
  state->initialized = true;
  state->capture_pending = false;
  return "ready";
}

String collective_poll(CollectiveOwner owner) {
  auto state = owner->state;
  std::lock_guard<std::mutex> lock(state->mutex);
  if (state->comm) {
    if (collective_async(*state) == ncclInProgress)
      return state->initialized ? "pending" : "initializing";
    TVM_FFI_ICHECK(!state->failed) << "communicator failed; abort or terminate the rank";
    state->initialized = true;
    state->capture_pending = false;
  }
  TVM_FFI_ICHECK(!state->enqueue || !state->enqueue->capturing)
      << "capture requires finish_enqueue before completion polling";
  tvm::ffi::CUDADeviceGuard guard(0);
  collective_outside_capture(collective_stream());
  collective_reap_graphs(*state);
  if (state->comm && state->enqueue)
    collective_finish(state->enqueue);
  for (auto &plan : state->flights) {
    if (!plan->recorded)
      collective_record(plan);
    auto status = cudaEventQuery(plan->event);
    if (status == cudaErrorNotReady)
      continue;
    collective_cuda(status);
    plan->active = false;
    plan->warmed = state->comm != nullptr;
  }
  auto end = std::remove_if(state->flights.begin(), state->flights.end(),
                            [](auto &plan) { return !plan->active; });
  state->flights.erase(end, state->flights.end());
  if (state->completion_recorded) {
    auto status = cudaEventQuery(state->completion);
    if (status == cudaSuccess)
      state->completion_recorded = false;
    else if (status != cudaErrorNotReady)
      collective_cuda(status);
  }
  collective_unroot(state);
  if (!state->flights.empty() || state->completion_recorded ||
      (!state->comm && !state->graphs.empty()))
    return "pending";
  return state->comm ? "ready" : "aborted";
}

void collective_record_completion(CollectiveOwner owner) {
  auto state = owner->state;
  std::lock_guard<std::mutex> lock(state->mutex);
  collective_ready(*state);
  tvm::ffi::CUDADeviceGuard guard(0);
  auto stream = collective_stream();
  collective_outside_capture(stream);
  collective_cuda(cudaEventRecord(state->completion, stream));
  state->completion_recorded = true;
}

bool collective_abort_ready(CollectiveOwner owner) {
  auto state = owner->state;
  std::lock_guard<std::mutex> lock(state->mutex);
  for (auto &lease : state->graphs)
    if (lease->attached && !lease->done.load(std::memory_order_acquire))
      return false;
  return true;
}

void collective_abort(CollectiveOwner owner) {
  auto state = owner->state;
  std::lock_guard<std::mutex> lock(state->mutex);
  for (auto &lease : state->graphs)
    TVM_FFI_ICHECK(!lease->attached || lease->done.load(std::memory_order_acquire))
        << "destroy collective graphs before abort; use process termination on timeout";
  if (state->comm) {
    collective_nccl(ncclCommAbort(state->comm));
    state->comm = nullptr;
  }
  collective_reap_graphs(*state);
  for (auto &lease : state->graphs)
    if (lease->object) {
      collective_cuda(cudaUserObjectRelease(lease->object));
      lease->object = nullptr;
    }
  state->enqueue.reset();
  collective_unroot(state);
}

CollectivePlan collective_prepare(CollectiveOwner owner, String kind, Array<Tensor> tensors,
                                  Shape directions, Shape peers, tvm::ffi::Function allocate) {
  auto state = owner->state;
  std::lock_guard<std::mutex> lock(state->mutex);
  auto staging = collective_check(state->world, state->rank, kind, tensors, directions, peers);
  collective_ready(*state);
  tvm::ffi::CUDADeviceGuard guard(0);
  collective_outside_capture(collective_stream());
  auto plan = std::make_shared<CollectivePlanState>();
  plan->owner = state;
  plan->kind = std::move(kind);
  plan->tensors = std::move(tensors);
  plan->directions = std::move(directions);
  plan->peers = std::move(peers);
  plan->staging = std::move(staging);
  for (size_t i = 0; i < plan->tensors.size(); ++i) {
    auto buffer =
        plan->staging[i]
            ? allocate(Shape({plan->staging[i]}), String("uint8"), int64_t(0)).cast<Tensor>()
            : plan->tensors[i];
    plan->buffers.push_back(std::move(buffer));
  }
  collective_storage_check(*plan);
  for (size_t i = 0; i < plan->buffers.size(); ++i)
    if (plan->staging[i])
      collective_cuda(
          cudaMemsetAsync(plan->buffers[i].data_ptr(), 0, plan->staging[i], collective_stream()));
  collective_cuda(cudaEventCreateWithFlags(&plan->event, cudaEventDisableTiming));
  auto object = tvm::ffi::make_object<CollectivePlanObj>();
  object->state = std::move(plan);
  return CollectivePlan(std::move(object));
}

void collective_prepare_capture(CollectivePlan handle) {
  auto plan = handle->state;
  auto state = plan->owner;
  std::lock_guard<std::mutex> lock(state->mutex);
  collective_ready(*state);
  tvm::ffi::CUDADeviceGuard guard(0);
  collective_outside_capture(collective_stream());
  collective_reap_graphs(*state);
  TVM_FFI_ICHECK(plan->warmed && !plan->active && !plan->graph)
      << "capture requires completed eager warmup and no existing graph lease";
  auto lease = std::make_unique<CollectiveGraphLease>();
  lease->plan = plan;
  auto pointer = lease.get();
  state->graphs.push_back(std::move(lease));
  collective_cuda(cudaUserObjectCreate(&pointer->object, pointer, collective_graph_done, 1,
                                       cudaUserObjectNoDestructorSync));
  plan->graph = pointer;
}

inline ncclResult_t collective_launch(const CollectivePlanState &plan, ncclComm_t comm,
                                      cudaStream_t stream) {
  if (plan.kind == "routing") {
    auto result = ncclGroupStart();
    if (result != ncclSuccess)
      return result;
    ncclResult_t first = ncclSuccess;
    for (size_t i = 0; i < plan.tensors.size(); ++i) {
      const auto &tensor = plan.tensors[i];
      auto dtype = collective_dtype(tensor.dtype());
      auto status = plan.directions[i] == 0 ? ncclSend(plan.buffers[i].data_ptr(), tensor.size(0),
                                                       dtype, plan.peers[i], comm, stream)
                                            : ncclRecv(plan.buffers[i].data_ptr(), tensor.size(0),
                                                       dtype, plan.peers[i], comm, stream);
      if (status != ncclSuccess) {
        first = status;
        break;
      }
    }
    auto end = ncclGroupEnd();
    if (end != ncclSuccess && end != ncclInProgress)
      return end;
    return first == ncclSuccess ? end : first;
  }
  auto source = plan.buffers[0].data_ptr(), destination = plan.buffers[1].data_ptr();
  auto dtype = collective_dtype(plan.tensors[0].dtype());
  if (plan.kind == "all_reduce")
    return ncclAllReduce(source, destination, plan.tensors[0].size(0), dtype, ncclSum, comm,
                         stream);
  if (plan.kind == "all_gather")
    return ncclAllGather(source, destination, plan.tensors[0].size(0), dtype, comm, stream);
  return ncclReduceScatter(source, destination, plan.tensors[1].size(0), dtype, ncclSum, comm,
                           stream);
}

void collective_submit(CollectivePlan handle) {
  auto plan = handle->state;
  auto state = plan->owner;
  std::lock_guard<std::mutex> lock(state->mutex);
  collective_ready(*state);
  collective_check(state->world, state->rank, plan->kind, plan->tensors, plan->directions,
                   plan->peers);
  collective_storage_check(*plan);
  tvm::ffi::CUDADeviceGuard guard(0);
  auto stream = collective_stream();
  cudaStreamCaptureStatus capture;
  unsigned long long id = 0;
  cudaGraph_t graph = nullptr;
  collective_cuda(cudaStreamGetCaptureInfo(stream, &capture, &id, &graph));
  TVM_FFI_ICHECK(capture != cudaStreamCaptureStatusInvalidated);
  bool capturing = capture == cudaStreamCaptureStatusActive;
  if (capturing) {
    auto lease = plan->graph;
    TVM_FFI_ICHECK(plan->warmed && !plan->active && lease && !lease->done.load())
        << "prepare_capture must precede capture";
    if (!lease->attached) {
      collective_cuda(cudaGraphRetainUserObject(graph, lease->object, 1, cudaGraphUserObjectMove));
      lease->object = nullptr;
      lease->attached = true;
      lease->capture_id = id;
    }
    TVM_FFI_ICHECK(lease->capture_id == id) << "a collective plan is bound to another graph";
  } else {
    TVM_FFI_ICHECK(!plan->active && !plan->graph) << "collective plan is still in use";
    plan->active = true;
    plan->recorded = false;
    state->flights.push_back(plan);
  }
  plan->stream = stream;
  plan->capturing = capturing;
  state->enqueue = plan;
  collective_copy(plan, true);
  auto status = collective_launch(*plan, state->comm, stream);
  if (status == ncclInProgress) {
    if (capturing) {
      state->capture_pending = true;
    }
  } else if (status != ncclSuccess) {
    state->failed = true;
    collective_nccl(status);
  }
}

void collective_finish_enqueue(CollectivePlan handle) {
  auto plan = handle->state;
  auto state = plan->owner;
  std::lock_guard<std::mutex> lock(state->mutex);
  TVM_FFI_ICHECK(!state->failed) << "communicator failed; terminate the rank";
  auto status = collective_async(*state);
  TVM_FFI_ICHECK(status == ncclSuccess && !state->capture_pending && state->enqueue == plan &&
                 plan->capturing)
      << "finish_enqueue requires capture host polling to ready";
  tvm::ffi::CUDADeviceGuard guard(0);
  auto stream = collective_stream();
  cudaStreamCaptureStatus capture;
  unsigned long long id = 0;
  collective_cuda(cudaStreamGetCaptureInfo(stream, &capture, &id));
  TVM_FFI_ICHECK(stream == plan->stream && capture == cudaStreamCaptureStatusActive &&
                 plan->graph && plan->graph->capture_id == id)
      << "finish_enqueue must use the original capture and stream";
  collective_finish(plan);
}
