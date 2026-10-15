import ctypes
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import tvm_ffi

from .build import build


class Device:
    def __init__(self):
        self.module, self.source_id, self.library = build(
            "device", ("allocate", "copy_bytes", "synchronize", "evict", "l2_bytes")
        )
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
        self.eviction_buffers = None
        self.l2_bytes = self.module.l2_bytes()
        self.eviction_bytes = 1 << (2 * self.l2_bytes).bit_length()

    def empty(self, shape, dtype):
        return self.module.allocate(tvm_ffi.Shape(shape), dtype, 0)

    def upload(self, value, dtype=None):
        value = np.require(value, requirements=["C", "W"])
        output = self.empty(value.shape, dtype or str(value.dtype))
        self.module.copy_bytes(output, tvm_ffi.from_dlpack(value))
        return output

    def download(self, value, dtype):
        output = np.empty(value.shape, dtype=dtype)
        self.module.copy_bytes(tvm_ffi.from_dlpack(output), value)
        return output

    def evict(self):
        if self.eviction_buffers is None:
            self.eviction_buffers = tuple(
                self.empty((self.eviction_bytes // 4,), "uint32") for _ in range(2)
            )
        self.module.evict(*self.eviction_buffers)

    def synchronize(self):
        self.module.synchronize(0)

    def profile(self, operation):
        self.evict()
        self.synchronize()
        self.checked(self.runtime.cudaProfilerStart())
        try:
            operation()
            self.synchronize()
        finally:
            self.checked(self.runtime.cudaProfilerStop())

    def checked(self, status):
        if status:
            raise RuntimeError(f"CUDA runtime error {status}")

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

    def time(self, operation, reset, *, warmup=2, repeats=5):
        events = [ctypes.c_void_p(), ctypes.c_void_p()]
        created = []
        try:
            for event in events:
                self.checked(self.runtime.cudaEventCreate(ctypes.byref(event)))
                created.append(event)
            measurements = []
            for index in range(warmup + repeats):
                reset()
                self.evict()
                self.checked(self.runtime.cudaEventRecord(events[0], None))
                operation()
                self.checked(self.runtime.cudaEventRecord(events[1], None))
                self.checked(self.runtime.cudaEventSynchronize(events[1]))
                duration = ctypes.c_float()
                self.checked(
                    self.runtime.cudaEventElapsedTime(ctypes.byref(duration), *events)
                )
                if index >= warmup:
                    measurements.append(duration.value * 1e-3)
            return measurements
        finally:
            for event in created:
                self.checked(self.runtime.cudaEventDestroy(event))
