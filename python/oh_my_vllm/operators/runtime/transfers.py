import ctypes
from dataclasses import dataclass

import numpy as np
import tvm_ffi

from .uploads import HostStorage


class HostSlice:
    def __init__(self, pool, slab, offset, size, reserved):
        self.pool, self.slab = pool, slab
        self.offset, self.size, self.reserved = offset, size, reserved
        memory = (ctypes.c_uint8 * size).from_address(slab.pointer.value + offset)
        self.array = np.ndarray((size,), np.uint8, buffer=memory)
        self.tensor = tvm_ffi.from_dlpack(self.array)

    def close(self):
        if self.pool is not None:
            self.pool.release(self)


class PinnedPool:
    def __init__(self, device, limit, quiescent):
        if type(limit) is not int or limit < 0:
            raise ValueError("invalid pinned pool limit")
        self.device, self.limit, self.quiescent = device, limit, quiescent
        self.slabs = {}
        self.allocated_bytes = self.used_bytes = 0
        self.closed = False

    def acquire(self, size):
        if self.closed:
            raise RuntimeError("pinned pool is closed")
        if type(size) is not int or size <= 0:
            raise ValueError("pinned slice size must be positive")
        reserved = (size + 255) // 256 * 256
        if self.used_bytes + reserved > self.limit:
            return None
        candidates = [
            (length, index, slab)
            for slab, free in self.slabs.items()
            for index, (_, length) in enumerate(free)
            if length >= reserved
        ]
        if candidates:
            _, index, slab = min(candidates, key=lambda item: item[0])
        else:
            if not self.quiescent():
                return None
            if self.allocated_bytes + reserved > self.limit:
                self.trim()
            remaining = self.limit - self.allocated_bytes
            if remaining < reserved:
                return None
            capacity = min(remaining, max(64 << 20, reserved))
            slab = HostStorage(self.device, capacity)
            self.slabs[slab] = [(0, capacity)]
            self.allocated_bytes += capacity
            index = 0
        offset, length = self.slabs[slab][index]
        result = HostSlice(self, slab, offset, size, reserved)
        self.slabs[slab].pop(index)
        if length > reserved:
            self.slabs[slab].insert(index, (offset + reserved, length - reserved))
        self.used_bytes += reserved
        return result

    def release(self, value):
        if value.pool is not self or value.slab not in self.slabs:
            raise ValueError("pinned slice belongs to a different pool")
        free = self.slabs[value.slab]
        free.append((value.offset, value.reserved))
        free.sort()
        merged = []
        for offset, size in free:
            if merged and merged[-1][0] + merged[-1][1] == offset:
                start, previous = merged.pop()
                merged.append((start, previous + size))
            else:
                merged.append((offset, size))
        self.slabs[value.slab] = merged
        self.used_bytes -= value.reserved
        value.pool = value.slab = value.tensor = value.array = None

    def trim(self):
        if not self.quiescent():
            raise RuntimeError("pinned storage release requires idle CUDA streams")
        for slab, free in list(self.slabs.items()):
            if free == [(0, slab.size)]:
                slab.close()
                self.allocated_bytes -= slab.size
                del self.slabs[slab]

    def close(self):
        if self.used_bytes:
            raise RuntimeError("pinned pool still has live slices")
        self.trim()
        self.closed = True


@dataclass
class Transfer:
    producer: ctypes.c_void_p
    complete: ctypes.c_void_p
    destinations: tuple
    sources: tuple


class TransferEngine:
    def __init__(self, device):
        self.device, self.streams, self.pending, self.events = device, {}, {}, []
        self.closed = False
        try:
            for direction in ("h2d", "d2h"):
                stream = ctypes.c_void_p()
                device.checked(
                    device.runtime.cudaStreamCreateWithFlags(ctypes.byref(stream), 1)
                )
                self.streams[direction] = stream
            device.transfer_engines.add(self)
        except BaseException:
            self.close()
            raise

    def event(self):
        if self.events:
            return self.events.pop()
        event = ctypes.c_void_p()
        self.device.checked(
            self.device.runtime.cudaEventCreateWithFlags(ctypes.byref(event), 2)
        )
        return event

    def submit(self, job_id, destinations, sources, direction):
        if self.closed or job_id in self.pending:
            raise ValueError("transfer engine closed or transfer identifier repeated")
        if direction not in self.streams:
            raise ValueError("invalid transfer direction")
        if not destinations or len(destinations) != len(sources):
            raise ValueError("transfer requires matching nonempty tensor lists")
        producer = self.event()
        try:
            complete = self.event()
        except BaseException:
            self.events.append(producer)
            raise
        transfer = Transfer(producer, complete, tuple(destinations), tuple(sources))
        self.pending[job_id] = transfer
        stream = self.streams[direction]
        runtime, checked = self.device.runtime, self.device.checked
        try:
            compute = tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))
            checked(runtime.cudaEventRecord(producer, compute))
            checked(runtime.cudaStreamWaitEvent(stream, producer, 0))
            self.device.module.copy_batch(
                transfer.destinations, transfer.sources, stream.value
            )
            checked(runtime.cudaEventRecord(complete, stream))
        except BaseException:
            # A failed batch can already contain submitted copies.
            checked(runtime.cudaStreamSynchronize(stream))
            self.pending.pop(job_id)
            self.events.extend((producer, complete))
            raise

    def ready(self, status):
        if status == 600:
            return False
        self.device.checked(status)
        return True

    def poll(self):
        completed = []
        for job_id, transfer in list(self.pending.items()):
            if self.ready(self.device.runtime.cudaEventQuery(transfer.complete)):
                completed.append(job_id)
                self.pending.pop(job_id)
                self.events.extend((transfer.producer, transfer.complete))
        return completed

    def quiescent(self):
        compute = tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))
        return all(
            self.ready(self.device.runtime.cudaStreamQuery(stream))
            for stream in (compute, *self.streams.values())
        )

    def close(self):
        if self.closed:
            return
        for stream in self.streams.values():
            self.device.checked(self.device.runtime.cudaStreamSynchronize(stream))
        for transfer in self.pending.values():
            self.events.extend((transfer.producer, transfer.complete))
        self.pending.clear()
        self.device.destroy_handles(
            [("cudaEventDestroy", event) for event in self.events]
            + [("cudaStreamDestroy", stream) for stream in self.streams.values()]
        )
        self.events.clear()
        self.streams.clear()
        self.device.transfer_engines.discard(self)
        self.closed = True
