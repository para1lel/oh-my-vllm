import ctypes
import os
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import tvm_ffi

from . import torch_adapter
from .native import load
from .uploads import UploadPool, Uploads

FUNCTIONS = (
    "allocate",
    "allocate_scratch",
    "reserve_scratch",
    "copy_bytes",
    "copy_batch",
    "upload_bytes",
    "synchronize",
    "view",
    "zero",
    "gather_rows",
    "free_bytes",
    "allocated_bytes",
    "reserved_bytes",
)


@lru_cache(maxsize=128)
def tensor_shape(dimensions):
    return tvm_ffi.Shape(dimensions)


def torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        dtype = dtype.removeprefix("torch.")
        try:
            return getattr(torch, dtype)
        except AttributeError as error:
            raise ValueError(f"unsupported torch dtype: {dtype}") from error
    return torch.from_numpy(np.empty((), dtype=dtype)).dtype


class Device:
    def __init__(self):
        self.module, self.source_id, self.library = load("device", FUNCTIONS)
        cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
        self.runtime = ctypes.CDLL(str(cuda / "lib64/libcudart.so"))
        for name, arguments in {
            "cudaMemGetInfo": [
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ],
            "cudaHostAlloc": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
                ctypes.c_uint,
            ],
            "cudaFreeHost": [ctypes.c_void_p],
            "cudaEventCreateWithFlags": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint,
            ],
            "cudaStreamIsCapturing": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)],
            "cudaEventCreate": [ctypes.POINTER(ctypes.c_void_p)],
            "cudaEventDestroy": [ctypes.c_void_p],
            "cudaEventQuery": [ctypes.c_void_p],
            "cudaEventRecord": [ctypes.c_void_p, ctypes.c_void_p],
            "cudaEventSynchronize": [ctypes.c_void_p],
            "cudaEventElapsedTime": [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.c_void_p,
            ],
            "cudaStreamCreateWithFlags": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint,
            ],
            "cudaStreamDestroy": [ctypes.c_void_p],
            "cudaStreamQuery": [ctypes.c_void_p],
            "cudaStreamSynchronize": [ctypes.c_void_p],
            "cudaStreamWaitEvent": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint],
            "cudaProfilerStart": [],
            "cudaProfilerStop": [],
        }.items():
            function = getattr(self.runtime, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int
        self.runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.runtime.cudaGetErrorString.restype = ctypes.c_char_p
        self.transfer_engines = set()
        self.upload_pool = UploadPool(self)
        self.capture_stream = None
        self.graph_failed = False
        self._failed_graph_resources = []

    def close(self):
        self.upload_pool.close()
        self.capture_stream = None

    def memory_info(self):
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.checked(
            self.runtime.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        )
        return free.value, total.value

    def empty(self, shape, dtype):
        return torch.ops.oh_my_vllm.native_empty(
            tuple(shape),
            torch_dtype(dtype),
            torch.device("cuda:0"),
        )

    def uploads(self):
        return Uploads(self)

    def scratch(self, shape, dtype):
        return torch.ops.oh_my_vllm.native_scratch(
            tuple(shape), torch_dtype(dtype), torch.device("cuda:0")
        )

    def zeros(self, shape, dtype):
        result = self.empty(shape, dtype)
        return torch.ops.oh_my_vllm.native_zero(result)

    def zero(self, value):
        return torch.ops.oh_my_vllm.native_zero(value)

    def copy(self, destination, source):
        return torch.ops.oh_my_vllm.native_copy(source, destination)

    def view(self, value, shape, offset=0):
        return torch.ops.oh_my_vllm.native_view(value, tuple(shape), offset)

    def upload(self, value, dtype=None):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("device upload input must be a CPU tensor")
            if not value.is_contiguous():
                raise ValueError("device upload requires a contiguous tensor")
            source = value
        else:
            source = torch.as_tensor(np.require(value, requirements=["C", "W"]))
        output = self.empty(
            source.shape,
            torch_dtype(dtype) if dtype is not None else source.dtype,
        )
        return torch.ops.oh_my_vllm.native_copy(source, output)

    def download(self, value, dtype):
        result = torch.ops.oh_my_vllm.native_download(value)
        if dtype is None:
            return result
        target = torch_dtype(dtype)
        if target == result.dtype:
            return result
        if target.itemsize != result.element_size():
            raise ValueError("download dtype must keep the source element width")
        # A download preserves raw bytes; a same-width dtype reinterprets them.
        return result.view(target)

    def synchronize(self):
        self.module.synchronize(0)

    def synchronize_compute(self):
        stream = torch_adapter.current_stream("cuda:0")
        self.checked(self.runtime.cudaStreamSynchronize(stream))

    def pinned_quiescent(self):
        stream = torch_adapter.current_stream("cuda:0")
        status = self.runtime.cudaStreamQuery(stream)
        if status == 600:
            return False
        self.checked(status)
        return all(engine.quiescent() for engine in self.transfer_engines)

    def wait_pinned_safe(self):
        self.synchronize_compute()
        for engine in self.transfer_engines:
            for stream in engine.streams.values():
                self.checked(self.runtime.cudaStreamSynchronize(stream))

    def destroy_handles(self, handles, *, suppress=False):
        error = None
        for name, handle in handles:
            if handle:
                try:
                    self.checked(getattr(self.runtime, name)(handle))
                    if isinstance(handle, ctypes.c_void_p):
                        handle.value = None
                except Exception as failure:
                    error = error or failure
        if error is not None and not suppress:
            raise error
        return error

    def _check_graph_health(self):
        if getattr(self, "graph_failed", False):
            raise RuntimeError("CUDA Graph device is unusable after cleanup failure")

    def ensure_healthy(self):
        self._check_graph_health()

    def _capture_stream(self):
        stream = getattr(self, "capture_stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device="cuda:0")
            self.capture_stream = stream
        return stream

    def capture(self, operation):
        self._check_graph_health()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("caller stream is already capturing")
        self.wait_pinned_safe()
        stream = self._capture_stream()
        graph = torch.cuda.CUDAGraph()
        pool = torch.cuda.graph_pool_handle()
        try:
            with (
                torch_adapter.use_stream("cuda:0", stream),
                torch.cuda.graph(graph, pool=pool, stream=stream),
            ):
                operation()
        except BaseException:
            self.graph_failed = False
            raise
        return graph

    def release_graph(self, graph, resources=None):
        try:
            torch.cuda.synchronize()
            graph.reset()
        except BaseException:
            self.graph_failed = True
            retained = resources.pop_all() if resources is not None else None
            self._failed_graph_resources.append((graph, retained))
            raise

    @contextmanager
    def graph(self, operation):
        owner = getattr(operation, "__self__", operation)
        acquire = getattr(owner, "graph_lease", None)
        with ExitStack() as resources:
            # Failed cleanup transfers these input owners with the retained graph.
            resources.callback(lambda owners=(operation, owner): None)
            if acquire is not None:
                resources.enter_context(acquire())
            graph = self.capture(operation)
            active = True

            def replay():
                self._check_graph_health()
                if not active:
                    raise RuntimeError("CUDA Graph is closed")
                stream = torch.cuda.current_stream()
                with torch_adapter.use_stream("cuda:0", stream):
                    graph.replay()

            try:
                yield replay
            except BaseException as failure:
                try:
                    self.release_graph(graph, resources)
                except BaseException as cleanup:
                    raise failure from cleanup
                raise
            else:
                self.release_graph(graph, resources)
            finally:
                active = False

    def checked(self, status):
        if status:
            message = self.runtime.cudaGetErrorString(status).decode("utf-8", "replace")
            raise RuntimeError(f"CUDA runtime error {status}: {message}")
