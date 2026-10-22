import ctypes
from contextlib import suppress

import numpy as np
import tvm_ffi


class HostStorage:
    def __init__(self, device, size):
        self.device, self.size = device, size
        self.pointer = ctypes.c_void_p()
        device.checked(
            device.runtime.cudaHostAlloc(ctypes.byref(self.pointer), size, 0)
        )

    def close(self):
        if self.pointer:
            self.device.checked(self.device.runtime.cudaFreeHost(self.pointer))
            self.pointer = ctypes.c_void_p()

    def __del__(self):
        with suppress(BaseException):
            self.close()


class UploadPool:
    def __init__(self, device, limit=16 << 20):
        self.device, self.limit = device, limit
        self.available = {}
        self.retired = {}
        self.cached_bytes = self.retired_bytes = 0
        self.closed = False

    def acquire(self, size):
        if self.closed:
            raise RuntimeError("host upload pool is closed")
        capacity = 1 << (max(size, 1) - 1).bit_length()
        candidates = [
            (stored, inventory)
            for inventory in (self.available, self.retired)
            for stored, buffers in inventory.items()
            if stored >= capacity and buffers
        ]
        if candidates:
            stored, inventory = min(candidates, key=lambda item: item[0])
            if inventory is self.available:
                self.cached_bytes -= stored
            else:
                self.retired_bytes -= stored
            return inventory[stored].pop()
        self.device.wait_pinned_safe()
        self._reclaim()
        return HostStorage(self.device, capacity)

    def release(self, storage):
        if not self.closed and self.cached_bytes + storage.size <= self.limit:
            self.available.setdefault(storage.size, []).append(storage)
            self.cached_bytes += storage.size
        elif (
            not self.closed
            and self.retired_bytes + storage.size <= self.limit
            and not self.device.pinned_quiescent()
        ):
            self.retired.setdefault(storage.size, []).append(storage)
            self.retired_bytes += storage.size
        else:
            self.device.wait_pinned_safe()
            self._reclaim()
            storage.close()

    def _drain(self, inventory):
        released, error = 0, None
        for size, available in inventory.items():
            pending = []
            for storage in available:
                try:
                    storage.close()
                    released += size
                except BaseException as failure:
                    pending.append(storage)
                    error = error or failure
            inventory[size] = pending
        return released, error

    def _reclaim(self):
        released, error = self._drain(self.retired)
        self.retired_bytes -= released
        if error is not None:
            raise error
        self.retired.clear()

    def close(self):
        self.closed = True
        if not self.cached_bytes and not self.retired_bytes:
            return
        self.device.wait_pinned_safe()
        released, error = self._drain(self.retired)
        self.retired_bytes -= released
        released, failure = self._drain(self.available)
        self.cached_bytes -= released
        error = error or failure
        if error is not None:
            raise error
        self.retired.clear()
        self.available.clear()

    def __del__(self):
        with suppress(BaseException):
            self.close()


class UploadBuffer:
    def __init__(self, device, destination, source):
        self.device, self.destination = device, destination
        self.storage, self.event = None, ctypes.c_void_p()
        self.pending = self.untracked = False
        self.array = self.tensor = None
        try:
            self.storage = device.upload_pool.acquire(source.nbytes)
            device.checked(
                device.runtime.cudaEventCreateWithFlags(ctypes.byref(self.event), 2)
            )
            memory = (ctypes.c_byte * self.storage.size).from_address(
                self.storage.pointer.value
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
            self.device.module.upload_bytes(self.destination, self.tensor)
            self.device.checked(self.device.runtime.cudaEventRecord(self.event, stream))
            self.untracked = False
        except BaseException:
            self.wait()
            raise

    def close(self):
        self.wait()
        self.tensor = self.array = None
        error = None
        if self.storage is not None:
            try:
                self.device.upload_pool.release(self.storage)
                self.storage = None
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
