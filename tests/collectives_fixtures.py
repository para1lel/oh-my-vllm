import importlib
import os
import subprocess
from pathlib import Path

CAPTURE_PENDING_CHECK = r"""
#include "collectives.cu"
#include <iostream>

int pending = 0, cuda_calls = 0, events = 0;
int64_t live_bytes = 0;
bool capturing = false, event_done = false;
ncclResult_t async_error = ncclSuccess;
struct Copy { void *dst; const void *src; size_t bytes; };
std::vector<Copy> copies;
std::vector<size_t> zeroed, routed;
std::vector<std::string> order;

void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

void cuda_call() {
  ++cuda_calls;
  require(!pending, "CUDA called while NCCL enqueue is pending");
}

extern "C" cudaError_t cudaGetDevice(int *device) {
  cuda_call();
  *device = 0;
  return cudaSuccess;
}

extern "C" cudaError_t cudaStreamGetCaptureInfo(
    cudaStream_t, cudaStreamCaptureStatus *status, unsigned long long *id,
    cudaGraph_t *graph, const cudaGraphNode_t **,
    const cudaGraphEdgeData **, size_t *) {
  cuda_call();
  *status = capturing ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
  if (id) *id = 7;
  if (graph) *graph = reinterpret_cast<cudaGraph_t>(1);
  return cudaSuccess;
}

extern "C" cudaError_t cudaGraphRetainUserObject(
    cudaGraph_t, cudaUserObject_t, unsigned int, unsigned int) {
  cuda_call();
  return cudaSuccess;
}

extern "C" cudaError_t cudaUserObjectCreate(
    cudaUserObject_t *object, void *, cudaHostFn_t, unsigned int, unsigned int) {
  cuda_call();
  *object = reinterpret_cast<cudaUserObject_t>(1);
  return cudaSuccess;
}

extern "C" cudaError_t cudaUserObjectRelease(cudaUserObject_t, unsigned int) {
  cuda_call();
  return cudaSuccess;
}

extern "C" cudaError_t cudaStreamIsCapturing(
    cudaStream_t, cudaStreamCaptureStatus *status) {
  cuda_call();
  *status = capturing ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
  return cudaSuccess;
}

extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int) {
  cuda_call();
  *event = reinterpret_cast<cudaEvent_t>(1);
  return cudaSuccess;
}

extern "C" cudaError_t cudaEventQuery(cudaEvent_t) {
  cuda_call();
  return event_done ? cudaSuccess : cudaErrorNotReady;
}

extern "C" cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) {
  cuda_call();
  ++events;
  order.push_back("event");
  return cudaSuccess;
}

extern "C" cudaError_t cudaEventDestroy(cudaEvent_t) {
  cuda_call();
  return cudaSuccess;
}

extern "C" cudaError_t cudaMemsetAsync(void *, int value, size_t bytes, cudaStream_t) {
  cuda_call();
  require(value == 0, "staging padding must be initialized");
  zeroed.push_back(bytes);
  return cudaSuccess;
}

extern "C" cudaError_t cudaMemcpyAsync(
    void *dst, const void *src, size_t bytes, cudaMemcpyKind kind, cudaStream_t) {
  cuda_call();
  require(kind == cudaMemcpyDeviceToDevice, "staging copies must stay on device");
  copies.push_back({dst, src, bytes});
  order.push_back("copy");
  return cudaSuccess;
}

extern "C" ncclResult_t ncclCommGetAsyncError(ncclComm_t, ncclResult_t *status) {
  *status = async_error != ncclSuccess ? async_error
                                      : (pending > 0 ? ncclInProgress : ncclSuccess);
  if (pending > 0) --pending;
  return ncclSuccess;
}

extern "C" ncclResult_t ncclAllReduce(
    const void *src, void *dst, size_t count, ncclDataType_t dtype,
    ncclRedOp_t, ncclComm_t, cudaStream_t) {
  require(reinterpret_cast<uintptr_t>(src) >= 0x10000 &&
          reinterpret_cast<uintptr_t>(dst) >= 0x10000 && count == 3 &&
          dtype == ncclBfloat16, "NCCL must see staged pointers and logical counts");
  order.push_back("launch");
  pending = 2;
  return ncclInProgress;
}

extern "C" ncclResult_t ncclGroupStart() { return ncclSuccess; }
extern "C" ncclResult_t ncclCommAbort(ncclComm_t) { return ncclSuccess; }
extern "C" ncclResult_t ncclGroupEnd() {
  pending = 2;
  return ncclInProgress;
}
extern "C" ncclResult_t ncclSend(
    const void *, size_t count, ncclDataType_t, int, ncclComm_t, cudaStream_t) {
  routed.push_back(count);
  return ncclSuccess;
}
extern "C" ncclResult_t ncclRecv(
    void *, size_t count, ncclDataType_t, int, ncclComm_t, cudaStream_t) {
  routed.push_back(count);
  return ncclSuccess;
}

struct MetadataAllocator {
  uintptr_t pointer;
  int64_t tracked = 0;
  void AllocData(DLTensor *tensor) {
    tensor->data = reinterpret_cast<void *>(pointer);
    live_bytes += tracked;
  }
  void FreeData(DLTensor *) { live_bytes -= tracked; }
};

Tensor metadata(int64_t count, uintptr_t pointer, int64_t tracked = 0) {
  return Tensor::FromNDAlloc(MetadataAllocator{pointer, tracked}, Shape({count}),
      tracked ? DLDataType{kDLUInt, 8, 1} : DLDataType{kDLBfloat, 16, 1},
      DLDevice{kDLCUDA, 0});
}

CollectiveOwner owner() {
  auto object = tvm::ffi::make_object<CollectiveOwnerObj>();
  object->state = std::make_shared<CollectiveState>();
  object->state->world = 2;
  object->state->rank = 0;
  object->state->initialized = true;
  object->state->comm = reinterpret_cast<ncclComm_t>(1);
  return CollectiveOwner(std::move(object));
}

template<typename F> void rejected_without_cuda(F run) {
  int before = cuda_calls;
  bool rejected = false;
  try { run(); } catch (const std::exception &) { rejected = true; }
  require(rejected && cuda_calls == before, "unsafe call must reject before CUDA");
}

int main() {
  try {
    uintptr_t next_pointer = 0x10000;
    auto allocate = tvm::ffi::Function::FromTyped(
        [&](Shape shape, String dtype, int64_t device) {
          require(dtype == "uint8" && device == 0, "use accounted byte allocation");
          auto tensor = metadata(shape[0], next_pointer, shape[0]);
          next_pointer += 0x1000;
          return tensor;
        });
    auto comm = owner();
    auto handle = collective_prepare(comm, "all_reduce",
        {metadata(3, 0x1002), metadata(3, 0x2002)}, {}, {}, allocate);
    auto plan = handle->state;
    require(live_bytes == 32 && zeroed == std::vector<size_t>({16, 16}),
        "prepare must own and initialize rounded allocations");
    auto short_allocate = tvm::ffi::Function::FromTyped(
        [&](Shape shape, String, int64_t) {
          return metadata(shape[0]-1, 0x30000, shape[0]-1);
        });
    bool rejected = false;
    try {
      collective_prepare(comm, "all_reduce",
          {metadata(3, 0x3002), metadata(3, 0x4002)}, {}, {}, short_allocate);
    } catch (const std::exception &) { rejected = true; }
    require(rejected && live_bytes == 32 && zeroed.size() == 2,
        "reject insufficient storage before asynchronous initialization");
    plan->warmed = true;
    collective_prepare_capture(handle);
    capturing = true;
    collective_submit(handle);
    require(!comm->state->failed && comm->state->capture_pending &&
        plan->graph->attached && copies.size() == 1 && copies[0].bytes == 6,
        "capture must retain staging and copy only logical input bytes");
    rejected_without_cuda([&] { collective_finish_enqueue(handle); });
    int before = cuda_calls;
    require(collective_poll_enqueue(comm) == "pending",
        "host enqueue is still pending");
    require(collective_poll_enqueue(comm) == "ready" && cuda_calls == before &&
        copies.size() == 1 && comm->state->enqueue == plan,
        "host readiness must not copy or finalize capture");
    rejected_without_cuda([&] { collective_submit(handle); });
    collective_finish_enqueue(handle);
    require(!comm->state->enqueue && copies.size() == 2 && copies[1].bytes == 6 &&
        copies[1].dst == reinterpret_cast<void *>(0x2002) && events == 0,
        "capture finalization must copy the logical output without recording an event");
    collective_submit(handle);
    before = cuda_calls;
    async_error = ncclRemoteError;
    bool reported_error = false;
    try { collective_poll_enqueue(comm); }
    catch (const std::exception &) { reported_error = true; }
    require(reported_error && comm->state->failed && comm->state->enqueue == plan &&
        comm->state->capture_pending && cuda_calls == before &&
        !collective_abort_ready(comm) && live_bytes == 32,
        "a real error must retain private storage without CUDA cleanup");
    rejected_without_cuda([&] { collective_finish_enqueue(handle); });

    pending = 0;
    async_error = ncclSuccess;
    capturing = false;
    {
    auto eager = owner();
    auto eager_handle = collective_prepare(eager, "all_reduce",
        {metadata(3, 0x5002), metadata(3, 0x6002)}, {}, {}, allocate);
    order.clear();
    copies.clear();
    collective_submit(eager_handle);
    before = cuda_calls;
    require(collective_poll(eager) == "pending" &&
        collective_poll(eager) == "pending" &&
        cuda_calls == before, "Eager polling must avoid CUDA while enqueue is pending");
    require(collective_poll(eager) == "pending" &&
        order == std::vector<std::string>({"copy", "launch", "copy", "event"}) &&
        copies[1].bytes == 6, "Eager fence must follow automatic result finalization");
    event_done = true;
    require(collective_poll(eager) == "ready" && eager_handle->state->warmed,
        "Eager readiness must include result-copy completion");
    auto routing = collective_prepare(eager, "routing",
        {metadata(3, 0x7002), metadata(3, 0x8002), metadata(0, 0), metadata(0, 0)},
        Shape({0, 1, 0, 1}), Shape({1, 1, 0, 0}), allocate);
    collective_submit(routing);
    require(routed == std::vector<size_t>({3, 3, 0, 0}),
        "staging must preserve route order and zero wire counts");
    require(collective_poll(eager) == "pending" &&
        collective_poll(eager) == "pending" && collective_poll(eager) == "ready",
        "routing must finalize through Eager polling");
    auto eager_state = eager->state;
    eager_state->comm = nullptr;
    }
    require(live_bytes == 32,
        "completed Eager plans must release their accounted storage");
    {
      auto closing = owner();
      auto unused = collective_prepare(closing, "all_reduce",
          {metadata(3, 0x9002), metadata(3, 0xa002)}, {}, {}, allocate);
      unused->state->warmed = true;
      collective_prepare_capture(unused);
      collective_abort(closing);
      require(collective_poll(closing) == "pending",
          "abort must wait for asynchronous unused graph-lease release");
      collective_graph_done(unused->state->graph);
      require(collective_poll(closing) == "aborted" && !unused->state->graph,
          "abort completes only after caller-thread lease cleanup");
    }
    require(live_bytes == 32, "unused graph reservations must release private owners");
    // The failed graph and enqueue deliberately keep their owners until process exit.
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
"""


def run_native_capture_check(directory):
    import tvm_ffi
    from oh_my_vllm.operators.runtime.native import (
        collectives_vendor_dependencies,
        cuda_include_directories,
    )
    from tvm_ffi.libinfo import find_libtvm_ffi

    root = Path(__file__).resolve().parents[1]
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    vendor = collectives_vendor_dependencies()
    ffi_library = Path(find_libtvm_ffi())
    source, executable = Path(directory) / "check.cpp", Path(directory) / "check"
    source.write_text(CAPTURE_PENDING_CHECK)
    subprocess.run(
        [
            vendor["host_compiler"],
            "-std=c++17",
            f"-I{Path(tvm_ffi.__file__).parent / 'include'}",
            *(f"-I{path}" for path in cuda_include_directories(cuda)),
            f"-I{root / 'python/oh_my_vllm/operators/runtime/cuda'}",
            f"-I{vendor['include_directory']}",
            str(source),
            str(ffi_library),
            f"-Wl,-rpath,{ffi_library.parent}",
            *vendor["ldflags"],
            "-pthread",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False, timeout=10
    )


def metadata_tensor(count, *, pointer, code=2, bits=32, device=0, stride=1):
    import ctypes

    import tvm_ffi

    class Device(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

    class Dtype(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_uint8),
            ("bits", ctypes.c_uint8),
            ("lanes", ctypes.c_uint16),
        ]

    class DLTensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("device", Device),
            ("ndim", ctypes.c_int),
            ("dtype", Dtype),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
            ("byte_offset", ctypes.c_uint64),
        ]

    class ManagedTensor(ctypes.Structure):
        _fields_ = [
            ("tensor", DLTensor),
            ("manager_ctx", ctypes.c_void_p),
            ("deleter", ctypes.c_void_p),
        ]

    shape, strides = (ctypes.c_int64 * 1)(count), (ctypes.c_int64 * 1)(stride)
    managed = ManagedTensor(
        DLTensor(
            pointer, Device(2, device), 1, Dtype(code, bits, 1), shape, strides, 0
        ),
        None,
        None,
    )
    capsule = ctypes.pythonapi.PyCapsule_New
    capsule.restype = ctypes.py_object
    capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    # These descriptors are for metadata checks only. No GPU operation may use them.
    tensor = tvm_ffi.from_dlpack(capsule(ctypes.addressof(managed), b"dltensor", None))
    return tensor, (managed, shape, strides)


class CollectivesFixture:
    def module(self):
        spec = importlib.util.find_spec("oh_my_vllm.operators.runtime.collectives")
        self.assertIsNotNone(spec, "the bounded collective owner API is missing")
        return importlib.import_module(spec.name)
