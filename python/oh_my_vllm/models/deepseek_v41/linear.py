"""BF16 activation quantization and encoded rank-weight projection.

The caller retains inputs through Eager completion or graph destruction.
Every status entry must be zero before the projected value can be accepted.
"""

from contextlib import ExitStack, contextmanager
from typing import NamedTuple

from ...operators.projection.api import Inputs as ProjectionInputs
from ...operators.projection.plan import make_plan as projection_plan
from ...operators.projection.workspace import prepare as prepare_projection
from ...operators.quantization.api import Inputs as QuantizationInputs
from ...operators.quantization.plan import make_plan as quantization_plan
from ...operators.quantization.workspace import prepare as prepare_quantization


class Outputs(NamedTuple):
    value: object
    status: object


def _span(tensor):
    size = {"uint8": 1, "bfloat16": 2}.get(str(tensor.dtype))
    if size is None:
        raise ValueError("encoded composition requires uint8 or BF16 storage")
    first = tensor.data_ptr()
    return first, first + tensor.numel() * size


def _disjoint(left, right):
    first, last = _span(left)
    other_first, other_last = _span(right)
    if max(first, other_first) < min(last, other_last):
        raise ValueError("encoded composition tensors overlap")


class Workspace:
    def __init__(self, owner, weights, rows, allocate):
        self.owner, self.weights, self.device = owner, weights, owner.device
        self.closed = False
        self.plan = projection_plan(
            rows=rows, outputs=weights.outputs, width=weights.width, bits=weights.bits
        )
        self.quantization = prepare_quantization(
            quantization_plan(rows=rows, width=weights.width, mode="fp8_ue8m0"),
            self.device,
            allocate=allocate,
        )
        self.projection = prepare_projection(
            self.plan, self.device, weights, allocate=allocate
        )
        self.outputs = Outputs(
            self.projection.outputs.value, self.quantization.outputs.status
        )
        self._writes = (
            *self.quantization.outputs,
            self.outputs.value,
            self.projection.padded_input,
            self.projection.packed_input_scales,
            self.projection.scratch,
        )
        self._weights = weights.encoded, weights.scales
        for index, left in enumerate(self._writes):
            for right in (*self._writes[index + 1 :], *self._weights):
                _disjoint(left, right)

    def check_open(self):
        if self.closed or self.owner.closed or self.owner._retained:
            raise RuntimeError(
                "encoded projection workspace or rank weights are closed"
            )

    @contextmanager
    def graph_lease(self):
        self.check_open()
        owners = self.owner, self.weights, tuple(self.owner._arenas.values())
        with ExitStack() as resources:
            resources.enter_context(self.quantization.graph_lease())
            resources.enter_context(self.projection.graph_lease())
            try:
                yield
            finally:
                del owners

    def close(self):
        self.closed = True
        self.quantization.close()
        self.projection.close()

    def run(self, value):
        self.check_open()
        if str(value.dtype) != "bfloat16" or tuple(value.shape) != (
            self.plan.rows,
            self.plan.width,
        ):
            raise ValueError("encoded projection input shape or dtype differs")
        for tensor in (*self._writes, *self._weights):
            _disjoint(value, tensor)
        quantized = self.quantization.run(QuantizationInputs(value))
        self.projection.run(ProjectionInputs(quantized.encoded, quantized.scales))
        return self.outputs


def prepare(rank_weights, name, *, rows, allocate=None):
    if rank_weights.closed or rank_weights._retained:
        raise RuntimeError("rank weights are closed or failed")
    if not isinstance(name, str) or name not in rank_weights.packed:
        raise ValueError("rank has no prepared encoded weight with this name")
    return Workspace(rank_weights, rank_weights.packed[name], rows, allocate)
