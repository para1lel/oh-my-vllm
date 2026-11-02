import ctypes
from contextlib import suppress

import torch

from . import torch_adapter


def _eager_stream(device):
    stream = torch_adapter.current_stream("cuda:0")
    capture = ctypes.c_int()
    device.checked(device.runtime.cudaStreamIsCapturing(stream, ctypes.byref(capture)))
    if capture.value:
        raise RuntimeError("update host inputs before CUDA Graph recording")
    return stream


class HostStorage:
    def __init__(self, device, size):
        self.device, self.size = device, size
        self.pointer = ctypes.c_void_p()
        device.checked(
            device.runtime.cudaHostAlloc(ctypes.byref(self.pointer), size, 0)
        )
        self._buffer = (ctypes.c_ubyte * max(size, 1)).from_address(self.pointer.value)
        self.tensor = torch.frombuffer(self._buffer, dtype=torch.uint8, count=size)

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

    def acquire(self, size, *, exact=False):
        if self.closed:
            raise RuntimeError("host upload pool is closed")
        capacity = 1 << (max(size, 1) - 1).bit_length()
        candidates = [
            (stored, inventory)
            for inventory in (self.available, self.retired)
            for stored, buffers in inventory.items()
            if (stored == capacity if exact else stored >= capacity) and buffers
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
    def __init__(self, device, destination, source, *, capacity_bytes=None):
        self.device, self.destination = device, destination
        self.storage, self.event = None, ctypes.c_void_p()
        self.pending = self.untracked = False
        self.tensor = None
        try:
            ranged = capacity_bytes is not None
            size = capacity_bytes if ranged else source.nbytes
            shape, dtype = (
                ((size,), torch.uint8) if ranged else (source.shape, source.dtype)
            )
            self.storage = device.upload_pool.acquire(size, exact=ranged)
            device.checked(
                device.runtime.cudaEventCreateWithFlags(ctypes.byref(self.event), 2)
            )
            self.tensor = (
                self.storage.tensor if ranged else self.storage.tensor.view(dtype)
            )
            elements = size if ranged else source.numel()
            self.tensor = self.tensor.reshape(-1)[:elements].reshape(shape)
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
        if (
            not isinstance(source, torch.Tensor)
            or source.device.type != "cpu"
            or not source.is_contiguous()
            or source.shape != self.tensor.shape
            or source.dtype != self.tensor.dtype
        ):
            raise ValueError("upload shape or dtype changed")
        self.wait()
        self.tensor.copy_(source)
        self._submit(self.destination, self.tensor, stream)

    def _submit(self, destination, source, stream):
        self.pending = self.untracked = True
        try:
            torch.ops.oh_my_vllm.native_upload(source, destination)
            self.device.checked(self.device.runtime.cudaEventRecord(self.event, stream))
            self.untracked = False
        except BaseException:
            self.wait()
            raise

    def close(self):
        self.wait()
        self.tensor = None
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


class RangeUpload(UploadBuffer):
    def __init__(self, device, destination, capacity_bytes):
        self.closed = False
        self.capacity_bytes = capacity_bytes
        self._views = None
        super().__init__(device, destination, None, capacity_bytes=capacity_bytes)
        self.staging_bytes = self.storage.size
        self.dtype = destination.dtype

    def copy(self, source, *, offset=0):
        if self.closed:
            raise RuntimeError("host upload range is closed")
        stream = _eager_stream(self.device)
        if (
            not isinstance(source, torch.Tensor)
            or source.device.type != "cpu"
            or not source.is_contiguous()
            or source.dtype != self.dtype
            or type(offset) is not int
            or not 0 <= offset <= self.destination.numel()
            or source.numel() > self.destination.numel() - offset
            or source.nbytes > self.capacity_bytes
        ):
            raise ValueError("upload range exceeds its prepared bounds or dtype")
        if not source.nbytes:
            return
        self.wait()
        snapshot = (
            self.storage.tensor.view(source.dtype)
            .reshape(-1)[: source.numel()]
            .reshape(source.shape)
        )
        snapshot.copy_(source)
        destination = self.device.view(self.destination, source.shape, offset)
        self._views = destination, snapshot
        self._submit(destination, snapshot, stream)

    def close(self):
        self.closed = True
        self.wait()
        self._views = None
        super().close()


class Uploads:
    def __init__(self, device):
        self.device = device
        self.buffers = {}
        self.ranges = {}
        self.closed = False

    def copy(self, destination, source):
        if self.closed:
            raise RuntimeError("host uploads are closed")
        if (
            not isinstance(source, torch.Tensor)
            or source.device.type != "cpu"
            or not source.is_contiguous()
        ):
            raise ValueError("host uploads require a contiguous CPU Torch tensor")
        stream = _eager_stream(self.device)
        key = id(destination)
        if key not in self.buffers:
            self.buffers[key] = UploadBuffer(self.device, destination, source)
        self.buffers[key].copy(source, stream)

    def prepare_range(self, destination, *, capacity_bytes):
        if self.closed:
            raise RuntimeError("host uploads are closed")
        _eager_stream(self.device)
        if (
            type(capacity_bytes) is not int
            or capacity_bytes <= 0
            or not isinstance(destination, torch.Tensor)
            or destination.device.type != "cuda"
            or not destination.is_contiguous()
        ):
            raise ValueError(
                "upload range requires positive capacity and contiguous CUDA storage"
            )
        if destination.element_size() <= 0:
            raise ValueError("upload range requires a byte-addressable scalar dtype")
        key = id(destination)
        if key in self.ranges:
            if self.ranges[key].capacity_bytes != capacity_bytes:
                raise ValueError("upload range capacity changed")
        else:
            self.ranges[key] = RangeUpload(self.device, destination, capacity_bytes)
        return self.ranges[key]

    def close(self):
        self.closed = True
        error = None
        for buffer in (*self.buffers.values(), *self.ranges.values()):
            try:
                buffer.close()
            except BaseException as failure:
                error = error or failure
        if error is not None:
            raise error
        self.buffers.clear()
        self.ranges.clear()
