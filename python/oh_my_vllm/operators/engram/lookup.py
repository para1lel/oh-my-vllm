import ctypes
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple

import tvm_ffi

from ..runtime.native import load
from .api import FUNCTIONS

# Keep owners until process teardown if CUDA cannot establish completion.
_failed_workspaces = []


def register(shared_file, *, limit):
    if not shared_file.sealed:
        raise RuntimeError("Shared weights must be sealed before registration")
    if type(limit) is not int or limit < shared_file.mapped_bytes:
        raise ValueError("Shared weight mapping exceeds the pinned byte limit")
    descriptor = shared_file.fileno()
    module, _, _ = load("engram", FUNCTIONS)
    return module.engram_register(descriptor, shared_file.size, limit)


def pinned_bytes():
    module, _, _ = load("engram", FUNCTIONS)
    return module.engram_pinned_bytes()


@dataclass(frozen=True)
class Plan:
    rows: int
    heads: int
    table_rows: int
    table_offset: int
    scale_offset: int

    @property
    def storage(self):
        return (
            ((self.rows, self.heads, 256), "bfloat16"),
            ((self.rows, self.heads), "uint8"),
        )

    @property
    def reuse_key(self):
        return (
            self.rows,
            self.heads,
            self.table_rows,
            self.table_offset,
            self.scale_offset,
        )

    @property
    def table_bytes(self):
        return max(
            self.table_offset + self.table_rows * 256,
            self.scale_offset + self.table_rows * 8,
        )


def make_plan(*, rows, heads, table_rows, scale_offset, table_offset=0):
    for value, lower, upper in (
        (rows, 0, 384),
        (heads, 1, 24),
        (table_rows, 1, (1 << 31) - 1),
        (table_offset, 0, (1 << 63) - 1),
        (scale_offset, 0, (1 << 63) - 1),
    ):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError("Engram lookup dimensions exceed supported bounds")
    plan = Plan(rows, heads, table_rows, table_offset, scale_offset)
    if plan.table_bytes > (1 << 63) - 1 or not (
        table_offset + table_rows * 256 <= scale_offset
        or scale_offset + table_rows * 8 <= table_offset
    ):
        raise ValueError("Engram table and scale ranges must fit and be disjoint")
    return plan


class Outputs(NamedTuple):
    value: object
    status: object


class Workspace:
    """Use one caller stream for Eager calls and graph replays.

    Enter each lease before capture and exit after graph destruction. Capture
    can use a private stream. Close must occur outside capture. Use one caller.
    """

    max_pending = 8

    def __init__(self, plan, device, table, allocate):
        if len(table.shape) != 1 or table.shape[0] < plan.table_bytes:
            raise ValueError("Engram shared table is smaller than the planned ranges")
        self.module, self.source_id, self.library = load("engram", FUNCTIONS)
        self.plan, self.device, self.table = plan, device, table
        self.outputs = Outputs(*(allocate(*field) for field in plan.storage))
        self.inputs = None
        self.pending, self.leases = deque(), {}
        self.stream = None
        self.closed = False

    def check_open(self):
        if self.closed:
            raise RuntimeError("Engram lookup workspace is closed")

    @contextmanager
    def graph_lease(self):
        self.check_open()
        stream, capture = self.stream_status()
        if capture:
            raise RuntimeError("Enter the Engram lookup lease before capture")
        self.check_stream(stream)
        self.retire(wait=True)
        key = object()
        owners = (self.table, self.outputs, self.module, {})
        self.leases[key] = owners[3]
        try:
            yield
        finally:
            del self.leases[key]

    def stream_status(self):
        stream = tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))
        capture = ctypes.c_int()
        self.device.checked(
            self.device.runtime.cudaStreamIsCapturing(stream, ctypes.byref(capture))
        )
        return stream, capture.value

    def check_stream(self, stream):
        if self.stream is not None and stream != self.stream:
            raise RuntimeError("Engram lookup requires one caller stream")
        self.stream = stream

    def retire(self, *, wait=False):
        runtime, checked = self.device.runtime, self.device.checked
        for _ in range(len(self.pending)):
            event, _inputs = self.pending[0]
            result = (
                runtime.cudaEventSynchronize(event)
                if wait
                else runtime.cudaEventQuery(event)
            )
            if not wait and result == 600:
                self.pending.rotate(-1)
                continue
            checked(result)
            checked(runtime.cudaEventDestroy(event))
            self.pending.popleft()

    def close(self):
        if self.closed:
            return
        _, capture = self.stream_status()
        if capture:
            raise RuntimeError("Close the Engram lookup workspace outside capture")
        try:
            self.retire(wait=True)
        except BaseException:
            self.retain_failed()
            raise
        self.closed = True
        self.table = self.inputs = None

    def retain_failed(self):
        self.closed = True
        _failed_workspaces.append(self)

    def run(self, indices):
        self.check_open()
        if tuple(indices.shape) != (self.plan.rows, self.plan.heads):
            raise ValueError("Engram lookup indices differ from the workspace plan")
        stream, capture = self.stream_status()
        runtime, checked = self.device.runtime, self.device.checked
        if capture not in (0, 1) or (capture == 1 and not self.leases):
            raise RuntimeError("Engram lookup capture requires an active graph lease")
        if capture == 1:
            owners = next(reversed(self.leases.values()))
            owners.setdefault(id(indices), indices)
            self.launch(indices)
        else:
            self.check_stream(stream)
            self.retire()
            if len(self.pending) >= self.max_pending:
                self.retire(wait=True)
            event = ctypes.c_void_p()
            checked(runtime.cudaEventCreateWithFlags(ctypes.byref(event), 2))
            self.pending.append((event, indices))
            try:
                self.launch(indices)
                checked(runtime.cudaEventRecord(event, stream))
            except BaseException:
                try:
                    self.device.synchronize()
                    self.retire(wait=True)
                except BaseException:
                    self.retain_failed()
                raise
        self.inputs = indices
        return self.outputs

    def launch(self, indices):
        self.module.engram_lookup(
            self.table,
            indices,
            *self.outputs,
            self.plan.table_rows,
            self.plan.table_offset,
            self.plan.scale_offset,
        )

    def __del__(self):
        if not getattr(self, "closed", True):
            try:
                self.close()
            except BaseException:
                if not self.closed:
                    self.retain_failed()


def prepare(plan, device, table, *, allocate=None):
    return Workspace(
        plan, device, table, device.empty if allocate is None else allocate
    )


def run(workspace, indices):
    return workspace.run(indices)
