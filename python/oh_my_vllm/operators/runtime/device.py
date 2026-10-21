import ctypes
import os
from contextlib import ExitStack, contextmanager, suppress
from functools import lru_cache
from pathlib import Path

import numpy as np
import tvm_ffi

from .native import load
from .uploads import Uploads

FUNCTIONS = (
    "allocate",
    "allocate_scratch",
    "reserve_scratch",
    "copy_bytes",
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


class Device:
    def __init__(self):
        self.module, self.source_id, self.library = load("device", FUNCTIONS)
        cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
        self.runtime = ctypes.CDLL(str(cuda / "lib64/libcudart.so"))
        for name, arguments in {
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
            "cudaStreamBeginCapture": [ctypes.c_void_p, ctypes.c_int],
            "cudaStreamEndCapture": [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ],
            "cudaGraphInstantiateWithFlags": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_ulonglong,
            ],
            "cudaGraphLaunch": [ctypes.c_void_p, ctypes.c_void_p],
            "cudaGraphDestroy": [ctypes.c_void_p],
            "cudaGraphExecDestroy": [ctypes.c_void_p],
            "cudaProfilerStart": [],
            "cudaProfilerStop": [],
        }.items():
            function = getattr(self.runtime, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int
        self.runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.runtime.cudaGetErrorString.restype = ctypes.c_char_p

    def empty(self, shape, dtype):
        return self.module.allocate(tensor_shape(tuple(shape)), dtype, 0)

    def uploads(self):
        return Uploads(self)

    def scratch(self, shape, dtype):
        return self.module.allocate_scratch(tensor_shape(tuple(shape)), dtype, 0)

    def zeros(self, shape, dtype):
        result = self.empty(shape, dtype)
        self.module.zero(result)
        return result

    def view(self, value, shape, offset=0):
        return self.module.view(value, tensor_shape(tuple(shape)), offset)

    def upload(self, value, dtype=None):
        value = np.require(value, requirements=["C", "W"])
        output = self.empty(value.shape, dtype or str(value.dtype))
        self.module.copy_bytes(output, tvm_ffi.from_dlpack(value))
        return output

    def download(self, value, dtype):
        output = np.empty(value.shape, dtype=dtype)
        self.module.copy_bytes(tvm_ffi.from_dlpack(output), value)
        return output

    def synchronize(self):
        self.module.synchronize(0)

    def destroy_handles(self, handles, *, suppress=False):
        error = None
        for name, handle in handles:
            if handle:
                try:
                    self.checked(getattr(self.runtime, name)(handle))
                except Exception as failure:
                    error = error or failure
        if error is not None and not suppress:
            raise error

    def capture(self, operation):
        stream, graph, executable = (ctypes.c_void_p() for _ in range(3))
        temporary = (("cudaGraphDestroy", graph), ("cudaStreamDestroy", stream))
        try:
            self.checked(
                self.runtime.cudaStreamCreateWithFlags(ctypes.byref(stream), 1)
            )
            with tvm_ffi.use_raw_stream(tvm_ffi.device("cuda:0"), stream.value):
                self.checked(self.runtime.cudaStreamBeginCapture(stream, 1))
                try:
                    operation()
                except BaseException:
                    self.runtime.cudaStreamEndCapture(stream, ctypes.byref(graph))
                    raise
                self.checked(
                    self.runtime.cudaStreamEndCapture(stream, ctypes.byref(graph))
                )
            self.checked(
                self.runtime.cudaGraphInstantiateWithFlags(
                    ctypes.byref(executable), graph, 0
                )
            )
        except BaseException:
            self.destroy_handles(
                (("cudaGraphExecDestroy", executable), *temporary), suppress=True
            )
            raise
        try:
            self.destroy_handles(temporary)
        except BaseException:
            self.destroy_handles((("cudaGraphExecDestroy", executable),), suppress=True)
            raise
        return executable

    def release_graph(self, executable):
        error = None
        try:
            self.synchronize()
        except BaseException as failure:
            error = failure
        try:
            self.checked(self.runtime.cudaGraphExecDestroy(executable))
        except BaseException:
            if error is None:
                raise
        if error is not None:
            raise error

    @contextmanager
    def graph(self, operation):
        owner = getattr(operation, "__self__", operation)
        acquire = getattr(owner, "graph_lease", None)
        with ExitStack() as resources:
            if acquire is not None:
                resources.enter_context(acquire())
            executable = self.capture(operation)
            active = True

            def replay():
                if not active:
                    raise RuntimeError("CUDA Graph is closed")
                stream = tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))
                try:
                    self.checked(self.runtime.cudaGraphLaunch(executable, stream))
                except BaseException:
                    with suppress(BaseException):
                        self.synchronize()
                    raise

            try:
                yield replay
            except BaseException:
                with suppress(BaseException):
                    self.release_graph(executable)
                raise
            else:
                self.release_graph(executable)
            finally:
                active = False

    def checked(self, status):
        if status:
            message = self.runtime.cudaGetErrorString(status).decode("utf-8", "replace")
            raise RuntimeError(f"CUDA runtime error {status}: {message}")
