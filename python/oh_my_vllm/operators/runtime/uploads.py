import ctypes
from contextlib import suppress

import numpy as np
import tvm_ffi


class UploadBuffer:
    def __init__(self, device, destination, source):
        self.device, self.destination = device, destination
        self.pointer, self.event = ctypes.c_void_p(), ctypes.c_void_p()
        self.pending = self.untracked = False
        self.array = self.tensor = None
        try:
            device.checked(
                device.runtime.cudaHostAlloc(
                    ctypes.byref(self.pointer), max(source.nbytes, 1), 0
                )
            )
            device.checked(
                device.runtime.cudaEventCreateWithFlags(ctypes.byref(self.event), 2)
            )
            memory = (ctypes.c_byte * max(source.nbytes, 1)).from_address(
                self.pointer.value
            )
            self.array = np.ndarray(source.shape, source.dtype, buffer=memory)
            self.tensor = tvm_ffi.from_dlpack(self.array)
        except BaseException:
            self.close()
            raise

    def wait(self):
        if self.pending:
            if self.untracked:
                self.device.synchronize()
            else:
                self.device.checked(
                    self.device.runtime.cudaEventSynchronize(self.event)
                )
            self.pending = self.untracked = False

    def copy(self, source, stream):
        if source.shape != self.array.shape or source.dtype != self.array.dtype:
            raise ValueError("upload shape or dtype changed")
        self.wait()
        np.copyto(self.array, source, casting="no")
        self.pending = self.untracked = True
        try:
            self.device.module.copy_bytes(self.destination, self.tensor)
            self.device.checked(self.device.runtime.cudaEventRecord(self.event, stream))
            self.untracked = False
        except BaseException:
            self.wait()
            raise

    def close(self):
        self.wait()
        self.tensor = self.array = None
        error = None
        if self.pointer:
            try:
                self.device.checked(self.device.runtime.cudaFreeHost(self.pointer))
                self.pointer = ctypes.c_void_p()
            except BaseException as failure:
                error = failure
        if self.event:
            try:
                self.device.checked(self.device.runtime.cudaEventDestroy(self.event))
                self.event = ctypes.c_void_p()
            except BaseException as failure:
                error = error or failure
        if error is not None:
            raise error

    def __del__(self):
        with suppress(BaseException):
            self.close()


class Uploads:
    def __init__(self, device):
        self.device = device
        self.buffers = {}
        self.closed = False

    def copy(self, destination, source):
        if self.closed:
            raise RuntimeError("host uploads are closed")
        if not isinstance(source, np.ndarray):
            raise ValueError("host uploads require a NumPy array")
        stream = tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))
        capture = ctypes.c_int()
        self.device.checked(
            self.device.runtime.cudaStreamIsCapturing(stream, ctypes.byref(capture))
        )
        if capture.value:
            raise RuntimeError("update host inputs before CUDA Graph recording")
        key = id(destination)
        if key not in self.buffers:
            self.buffers[key] = UploadBuffer(self.device, destination, source)
        self.buffers[key].copy(source, stream)

    def close(self):
        self.closed = True
        error = None
        for buffer in self.buffers.values():
            try:
                buffer.close()
            except BaseException as failure:
                error = error or failure
        if error is not None:
            raise error
        self.buffers.clear()
