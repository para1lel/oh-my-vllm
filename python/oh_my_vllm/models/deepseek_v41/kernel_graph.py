from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import torch

_CAPTURED_TENSORS = ContextVar("deepseek_graph_cached_tensors", default=None)


@contextmanager
def capture_cached_tensors():
    owners = {}
    token = _CAPTURED_TENSORS.set(owners)
    try:
        yield owners
    finally:
        _CAPTURED_TENSORS.reset(token)


def retain_graph_tensors(value):
    owners = _CAPTURED_TENSORS.get()
    if owners is not None:
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, torch.Tensor):
                owners[id(item)] = item
            elif isinstance(item, (list, tuple)):
                pending.extend(item)
            elif isinstance(item, dict):
                pending.extend(item.values())
    return value


def retain_cached_tensor(operation):
    @wraps(operation)
    def selected(*args, **kwargs):
        return retain_graph_tensors(operation(*args, **kwargs))

    for name in ("cache_clear", "cache_info", "cache_parameters"):
        if hasattr(operation, name):
            setattr(selected, name, getattr(operation, name))
    return selected


class _PinnedMetadataUploader:
    """Reusable pinned hosts for stream-ordered metadata uploads."""

    def __init__(self, device, dtype=torch.int32, capacity=384):
        self.device = device
        self.dtype = dtype
        self.capacity = capacity
        self.slots = []
        self.cursor = 0

    def _new_slot(self):
        # Pin on the CPU explicitly: the caller can set a CUDA default device.
        host = torch.empty(
            (self.capacity,), dtype=self.dtype, device="cpu", pin_memory=True
        )
        device = torch.empty((self.capacity,), dtype=self.dtype, device=self.device)
        return [host, device, torch.cuda.Event()]

    def _slot(self):
        for offset in range(len(self.slots)):
            index = (self.cursor + offset) % len(self.slots)
            slot = self.slots[index]
            if slot[2].query():
                self.cursor = index + 1
                return slot
        if len(self.slots) < 64:
            slot = self._new_slot()
            self.slots.append(slot)
            self.cursor = len(self.slots)
            return slot
        slot = self.slots[self.cursor % len(self.slots)]
        slot[2].synchronize()
        self.cursor += 1
        return slot

    def upload(self, values):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("captured metadata needs a capture-owned host buffer")
        slot = self._slot()
        count = len(values)
        slot[0][:count].copy_(torch.tensor(values, dtype=self.dtype, device="cpu"))
        slot[1][:count].copy_(slot[0][:count], non_blocking=True)
        slot[2].record(torch.cuda.current_stream(self.device))
        return slot[1][:count]


_POINTER_UPLOADS = {}


_CAPTURE_POINTER_HOSTS = {}


@retain_cached_tensor
def _upload_pointer_array(values, role, device):
    """Upload one expert pointer array without a blocking host copy."""
    if torch.cuda.is_current_stream_capturing():
        # A captured copy must read pinned memory that outlives the graph.
        key = (role, tuple(values))
        host = _CAPTURE_POINTER_HOSTS.get(key)
        if host is None:
            host = torch.empty((len(values),), dtype=torch.int64, pin_memory=True)
            host.copy_(torch.tensor(values, dtype=torch.int64, device="cpu"))
            _CAPTURE_POINTER_HOSTS[key] = host
        retain_graph_tensors(host)
        result = torch.empty((len(values),), dtype=torch.int64, device=device)
        result.copy_(host, non_blocking=True)
        return result
    key = (role, str(device))
    uploader = _POINTER_UPLOADS.get(key)
    if uploader is None:
        uploader = _PinnedMetadataUploader(device, dtype=torch.int64)
        _POINTER_UPLOADS[key] = uploader
    return uploader.upload(values)
