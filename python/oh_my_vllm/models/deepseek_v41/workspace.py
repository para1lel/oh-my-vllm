"""Bounded storage and graph ownership for rank-local model computation."""

import math
from bisect import bisect_left
from contextlib import ExitStack, contextmanager

from ...operators.projection.workspace import require_eager

_FAILED_PREPARATIONS = []
_ELEMENT_BYTES = {"uint8": 1, "bfloat16": 2, "float32": 4, "int32": 4, "uint64": 8}


def _span(value):
    first = value.data_ptr()
    return first, first + value.numel() * _ELEMENT_BYTES[str(value.dtype)]


def _disjoint(span, intervals):
    first, last = span
    if first == last:
        return
    index = bisect_left(intervals, (first,))
    if (index and intervals[index - 1][1] > first) or (
        index < len(intervals) and intervals[index][0] < last
    ):
        raise ValueError("model writable storage overlaps another tensor")


class Workspace:
    def __init__(self, owner, reads, allocate, byte_limit, *, extra_bytes=0):
        self.owner, self.device = owner, owner.device
        self.closed, self.parts = False, []
        self.check_open()
        if type(byte_limit) is not int or byte_limit <= 0:
            raise ValueError("model workspace requires a positive byte limit")
        require_eager(self.device)
        self.allocated_bytes, self.byte_limit = extra_bytes, byte_limit
        self._allocate = self.device.empty if allocate is None else allocate
        self._writes, self._allocations = [], []
        self._tensor_device = reads[0].device
        if any(
            tensor.device != self._tensor_device or not tensor.is_contiguous()
            for tensor in reads
        ):
            raise ValueError("model weight device or layout differs")
        # Read-only aliases are legal. Merge them before checking allocations.
        self._reads = []
        for first, last in sorted(set(map(_span, reads))):
            if self._reads and first <= self._reads[-1][1]:
                self._reads[-1] = (self._reads[-1][0], max(last, self._reads[-1][1]))
            else:
                self._reads.append((first, last))

    def reserve(self, shape, dtype):
        size = math.prod(shape) * _ELEMENT_BYTES[dtype]
        if self.allocated_bytes + size > self.byte_limit:
            raise ValueError("model workspace exceeds its byte limit")
        value = self._allocate(shape, dtype)
        if (
            tuple(value.shape) != tuple(shape)
            or str(value.dtype) != dtype
            or not value.is_contiguous()
            or value.device != self._tensor_device
        ):
            raise ValueError("model allocator returned invalid storage")
        self._allocations.append(value)
        span = _span(value)
        _disjoint(span, self._reads)
        _disjoint(span, self._writes)
        if span[0] != span[1]:
            self._writes.insert(bisect_left(self._writes, span), span)
        self.allocated_bytes += size
        return value

    def part(self, factory, *args, **kwargs):
        workspace = factory(*args, **kwargs, allocate=self.reserve)
        self.parts.append(workspace)
        return workspace

    @contextmanager
    def preparing(self):
        try:
            yield
        except BaseException as error:
            try:
                self.device.synchronize_compute()
            except BaseException as completion:
                _FAILED_PREPARATIONS.append(self)
                error.add_note(
                    f"model preparation completion also failed: {completion}"
                )
            self.close()
            raise

    def check_open(self):
        if self.closed or self.owner.closed or self.owner._retained:
            raise RuntimeError("model workspace or rank weights are closed")

    def check_input(self, value, shape, dtype, *, writes=None):
        if (
            tuple(value.shape) != shape
            or str(value.dtype) != dtype
            or not value.is_contiguous()
            or value.device != self._tensor_device
        ):
            raise ValueError("model input shape, dtype, layout, or device differs")
        _disjoint(_span(value), self._writes if writes is None else writes)

    def writes_since(self, index):
        return sorted(
            _span(value) for value in self._allocations[index:] if value.numel()
        )

    @contextmanager
    def graph_lease(self):
        self.check_open()
        with ExitStack() as resources:
            for part in self.parts:
                resources.enter_context(part.graph_lease())
            yield

    def close(self):
        self.closed = True
        for part in reversed(self.parts):
            part.close()
