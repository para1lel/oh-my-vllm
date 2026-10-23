import ctypes
from contextlib import contextmanager

import tvm_ffi

from ..runtime.native import load
from .api import FUNCTIONS, Outputs


class Workspace:
    def __init__(self, plan, device, allocate):
        self.module, self.source_id, self.library = load("engram", FUNCTIONS)
        self.plan, self.device = plan, device
        self.outputs = Outputs(*(allocate(*field) for field in plan.storage))
        self.inputs = None
        self.leases = {}
        self.closed = False

    def check_open(self):
        if self.closed:
            raise RuntimeError("Engram workspace is closed")

    @contextmanager
    def graph_lease(self):
        self.check_open()
        key = object()
        owners = (self.outputs, self.module, {})
        self.leases[key] = owners[2]
        try:
            yield
        finally:
            del self.leases[key]

    def close(self):
        self.closed = True

    def run(self, inputs):
        self.check_open()
        tensors = (
            inputs.tokens,
            inputs.history,
            inputs.multipliers,
            inputs.primes,
            inputs.offsets,
        )
        if any(
            tuple(tensor.shape) != shape
            for tensor, shape in zip(tensors, self.plan.input_shapes, strict=True)
        ):
            raise ValueError("Engram inputs differ from the workspace plan")
        self.inputs = inputs
        if self.leases:
            stream = tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))
            capture = ctypes.c_int()
            self.device.checked(
                self.device.runtime.cudaStreamIsCapturing(stream, ctypes.byref(capture))
            )
            if capture.value == 1:
                owners = next(reversed(self.leases.values()))
                owners.setdefault(tuple(map(id, tensors)), inputs)
        self.module.engram_hash(
            *tensors,
            *self.outputs,
            self.plan.pad_id,
            self.plan.vocabulary_size,
        )
        return self.outputs


def prepare(plan, device, *, allocate=None):
    return Workspace(plan, device, device.empty if allocate is None else allocate)
