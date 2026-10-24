"""Device-counted A8W4 projections with a uniform output width and K.

Counts are in [0,384]; offsets start at zero and their differences equal counts.
Their final value must not exceed capacity. Invalid metadata empties all groups,
zeros all output rows, and sets every status byte. Reject status before commit.
Inputs stay owned by the caller until Eager completion or graph destruction.
"""

from contextlib import contextmanager

import numpy as np

from ..runtime.native import load
from .api import FUNCTIONS, GroupedInputs, GroupedOutputs, PreparedGroupedInputs
from .plan import scale_bytes
from .workspace import require_eager


def span(value):
    first = value.data_ptr()
    size = {"uint8": 1, "bfloat16": 2, "int32": 4, "uint64": 8}.get(str(value.dtype))
    if size is None:
        raise ValueError("unsupported grouped projection tensor type")
    return first, first + value.numel() * size


def disjoint(left, right):
    first, last = span(left)
    other_first, other_last = span(right)
    if max(first, other_first) < min(last, other_last):
        raise ValueError("grouped projection tensors overlap")


class Workspace:
    def __init__(self, plan, device, weights, allocate, prepared):
        require_eager(device)
        if type(prepared) is not bool:
            raise ValueError("prepared mode must be a boolean")
        if len(weights) != plan.groups:
            raise ValueError("grouped weight count differs")
        self.weights = tuple(weights)
        for weight in self.weights:
            if (weight.bits, weight.width, weight.outputs) != (
                4,
                plan.width,
                plan.outputs,
            ):
                raise ValueError("grouped weights differ from the plan")
            if (
                str(weight.encoded.dtype) != "uint8"
                or str(weight.scales.dtype) != "uint8"
                or tuple(weight.encoded.shape) != (plan.outputs, plan.padded_width // 2)
                or tuple(weight.scales.shape)
                != (scale_bytes(plan.outputs, plan.width),)
                or not weight.encoded.is_contiguous()
                or not weight.scales.is_contiguous()
                or weight.encoded.device != weight.scales.device
            ):
                raise ValueError("invalid grouped weight storage")
        self.module, self.source_id, self.library = load("projection", FUNCTIONS)
        self.plan, self.device, self.prepared = plan, device, prepared
        metadata, scratch, self.sm_count = self.module.grouped_sizes(plan.groups)
        self.outputs = GroupedOutputs(
            allocate((plan.capacity, plan.outputs), "bfloat16"),
            allocate((plan.groups,), "uint8"),
        )
        for weight in weights:
            for value in (weight.encoded, weight.scales):
                if value.device != self.outputs.value.device or value.data_ptr() % 16:
                    raise ValueError("grouped weight device or alignment differs")
        self.addresses = device.upload(
            np.array(
                [[w.encoded.data_ptr(), w.scales.data_ptr()] for w in weights],
                np.uint64,
            )
        )
        self.padded_input = (
            allocate((plan.capacity, plan.padded_width), "uint8")
            if not prepared and plan.width != plan.padded_width
            else None
        )
        self.packed_scales = (
            None if prepared else allocate((plan.scale_capacity,), "uint8")
        )
        self.metadata = allocate((metadata,), "uint8")
        self.scratch = allocate((scratch,), "uint8")
        self._writes = tuple(
            t
            for t in (
                *self.outputs,
                self.padded_input,
                self.packed_scales,
                self.metadata,
                self.scratch,
            )
            if t is not None
        )
        self._reads = (
            self.addresses,
            *(t for w in weights for t in (w.encoded, w.scales)),
        )
        for i, left in enumerate(self._writes):
            for right in (*self._writes[i + 1 :], *self._reads):
                disjoint(left, right)
        self.closed = False

    def check_open(self):
        if self.closed:
            raise RuntimeError("grouped projection workspace is closed")

    @contextmanager
    def graph_lease(self, inputs=None):
        self.check_open()
        owners = self.weights, self._reads, self._writes, self.module, inputs
        try:
            yield
        finally:
            del owners

    def close(self):
        self.closed = True

    def run(self, inputs):
        self.check_open()
        expected = PreparedGroupedInputs if self.prepared else GroupedInputs
        if not isinstance(inputs, expected):
            raise ValueError("grouped input layout differs from the workspace")
        for value in inputs:
            for other in self._writes:
                disjoint(value, other)
        operation = (
            self.module.project_grouped_prepared
            if self.prepared
            else self.module.project_grouped
        )
        conversion = (
            ()
            if self.prepared
            else (
                inputs.encoded if self.padded_input is None else self.padded_input,
                self.packed_scales,
            )
        )
        operation(
            *inputs,
            self.addresses,
            *self.outputs,
            *conversion,
            self.metadata,
            self.scratch,
            self.plan.width,
            self.sm_count,
        )
        return self.outputs


def prepare(plan, device, packed_weights, *, allocate=None, prepared=False):
    return Workspace(
        plan,
        device,
        packed_weights,
        device.empty if allocate is None else allocate,
        prepared,
    )
