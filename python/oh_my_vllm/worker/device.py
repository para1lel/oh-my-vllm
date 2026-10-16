import ctypes
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import tvm_ffi

from ..kernels.native import load

FUNCTIONS = (
    "allocate",
    "copy_bytes",
    "synchronize",
    "evict",
    "l2_bytes",
    "view",
    "zero",
    "gather_rows",
    "final_hidden",
    "free_bytes",
    "allocated_bytes",
)


class Device:
    def __init__(self):
        self.module, self.source_id, self.library = load("device", FUNCTIONS)
        cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
        self.runtime = ctypes.CDLL(str(cuda / "lib64/libcudart.so"))
        for name, arguments in {
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
            "cudaStreamEndCapture": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
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
        return self.module.allocate(tvm_ffi.Shape(shape), dtype, 0)

    def zeros(self, shape, dtype):
        result = self.empty(shape, dtype)
        self.module.zero(result)
        return result

    def view(self, value, shape, offset=0):
        return self.module.view(value, tvm_ffi.Shape(shape), offset)

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

    def checked(self, status):
        if status:
            message = self.runtime.cudaGetErrorString(status).decode("utf-8", "replace")
            raise RuntimeError(f"CUDA runtime error {status}: {message}")

    @contextmanager
    def graph(self, operation):
        stream, graph, executable = (ctypes.c_void_p() for _ in range(3))
        self.checked(self.runtime.cudaStreamCreateWithFlags(ctypes.byref(stream), 1))
        try:
            with tvm_ffi.use_raw_stream(tvm_ffi.device("cuda:0"), stream.value):
                self.checked(self.runtime.cudaStreamBeginCapture(stream, 1))
                try:
                    operation()
                finally:
                    status = self.runtime.cudaStreamEndCapture(
                        stream, ctypes.byref(graph)
                    )
                self.checked(status)
            self.checked(
                self.runtime.cudaGraphInstantiateWithFlags(
                    ctypes.byref(executable), graph, 0
                )
            )
            yield lambda: self.checked(self.runtime.cudaGraphLaunch(executable, None))
        finally:
            if executable.value:
                self.checked(self.runtime.cudaGraphExecDestroy(executable))
            if graph.value:
                self.checked(self.runtime.cudaGraphDestroy(graph))
            self.checked(self.runtime.cudaStreamDestroy(stream))
