import ctypes
from dataclasses import dataclass

import torch

from . import torch_adapter


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
            compute = torch_adapter.current_stream("cuda:0")
            checked(runtime.cudaEventRecord(producer, compute))
            checked(runtime.cudaStreamWaitEvent(stream, producer, 0))
            torch.ops.oh_my_vllm.native_copy_batch(
                transfer.sources, transfer.destinations, stream.value
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
        compute = torch_adapter.current_stream("cuda:0")
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
