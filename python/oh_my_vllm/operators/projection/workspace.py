import ctypes
from contextlib import contextmanager
from dataclasses import dataclass

import tvm_ffi

from ..runtime.native import load
from .api import FUNCTIONS, Outputs
from .plan import make_plan, scale_bytes


@dataclass(frozen=True)
class PackedWeights:
    encoded: object
    scales: object
    width: int
    bits: int
    outputs: int


def require_eager(device):
    stream = tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))
    capture = ctypes.c_int()
    device.checked(device.runtime.cudaStreamIsCapturing(stream, ctypes.byref(capture)))
    if capture.value:
        raise RuntimeError("prepare projection weights outside capture")


def prepare_weights(device, weights, *, bits, allocate=None):
    require_eager(device)
    if type(bits) is not int or bits not in (4, 8):
        raise ValueError("projection weight bits must be four or eight")
    if len(weights.encoded.shape) != 2:
        raise ValueError("projection weights must be a matrix")
    outputs, columns = weights.encoded.shape
    plan = make_plan(rows=0, outputs=outputs, width=columns * (8 // bits), bits=bits)
    shape = (outputs if bits == 4 else outputs // 32, plan.width // 32)
    if (
        str(weights.encoded.dtype) != "uint8"
        or str(weights.scales.dtype) != "uint8"
        or tuple(weights.scales.shape) != shape
    ):
        raise ValueError("projection raw weight type or scale shape differs")
    allocate = device.empty if allocate is None else allocate
    encoded = weights.encoded
    if plan.padded_width != plan.width:
        encoded = allocate((outputs, plan.padded_width // 2), "uint8")
    scales = allocate((scale_bytes(outputs, plan.width),), "uint8")
    module, _, _ = load("projection", FUNCTIONS)
    # Both uploads remain owned until packing on the caller stream completes.
    try:
        module.pack_weights(*weights, encoded, scales, bits)
    finally:
        device.synchronize_compute()
    return PackedWeights(encoded, scales, plan.width, bits, outputs)


class Workspace:
    def __init__(self, plan, device, weights, allocate):
        if (weights.width, weights.bits, weights.outputs) != (
            plan.width,
            plan.bits,
            plan.outputs,
        ):
            raise ValueError("prepared projection weights differ from the plan")
        self.module, self.source_id, self.library = load("projection", FUNCTIONS)
        self.plan, self.device, self.weights = plan, device, weights
        self.outputs = Outputs(allocate((plan.rows, plan.outputs), "bfloat16"))
        padding = plan.padded_width if plan.padded_width != plan.width else 0
        self.padded_input = allocate((plan.rows, padding), "uint8")
        self.packed_input_scales = allocate(
            (scale_bytes(plan.rows, plan.width),), "uint8"
        )
        size = self.module.workspace_size(*plan.reuse_key)
        self.scratch = allocate((size,), "uint8")
        self.closed = False

    @contextmanager
    def graph_lease(self):
        self.check_open()
        owners = (
            self.weights,
            self.outputs,
            self.padded_input,
            self.packed_input_scales,
            self.scratch,
            self.module,
        )
        try:
            yield
        finally:
            del owners

    def check_open(self):
        if self.closed:
            raise RuntimeError("projection workspace is closed")

    def close(self):
        self.closed = True

    def run(self, inputs):
        self.check_open()
        self.module.project(
            *inputs,
            self.weights.encoded,
            self.weights.scales,
            self.outputs.value,
            self.padded_input,
            self.packed_input_scales,
            self.scratch,
            self.plan.width,
            self.plan.bits,
        )
        return self.outputs


def prepare(plan, device, packed_weights, *, allocate=None):
    return Workspace(
        plan, device, packed_weights, device.empty if allocate is None else allocate
    )
