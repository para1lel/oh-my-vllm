"""Rank-local MoE; collectives and state acceptance belong to the caller.

Routed output is FP32. Shared output is BF16 and follows dense placement.
Reject every nonzero status before committing state. Inputs stay owned through
Eager completion or graph destruction. Preparation and closure are Eager-only;
close invalidates Python execution, not existing graph leases.
"""

import math
from bisect import bisect_left
from contextlib import ExitStack, contextmanager
from typing import NamedTuple

from ...operators.activation import api as activation
from ...operators.activation.plan import make_plan as activation_plan
from ...operators.activation.workspace import prepare as prepare_activation
from ...operators.linear import api as linear
from ...operators.linear.plan import make_plan as linear_plan
from ...operators.linear.workspace import prepare as prepare_linear
from ...operators.projection import api as projection
from ...operators.projection.grouped import prepare as prepare_grouped
from ...operators.projection.plan import make_grouped_plan
from ...operators.projection.plan import make_plan as projection_plan
from ...operators.projection.workspace import (
    prepare as prepare_projection,
)
from ...operators.projection.workspace import (
    require_eager,
)
from ...operators.quantization import api as quantization
from ...operators.quantization.plan import make_plan as quantization_plan
from ...operators.quantization.workspace import prepare as prepare_quantization
from ...operators.routing import api as routing
from ...operators.routing.plan import make_dispatch_plan
from ...operators.routing.plan import make_plan as routing_plan
from ...operators.routing.workspace import prepare as prepare_routing
from ...operators.routing.workspace import prepare_dispatch

_FAILED_PREPARATIONS = []
_ELEMENT_BYTES = {"uint8": 1, "bfloat16": 2, "float32": 4, "int32": 4, "uint64": 8}


class Outputs(NamedTuple):
    routed: object
    shared: object
    status: tuple


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
        raise ValueError("MoE writable storage overlaps another tensor")


def _configuration(owner, name, rows):
    if owner.closed or owner._retained:
        raise RuntimeError("rank weights are closed or failed")
    if not isinstance(name, str):
        raise ValueError("MoE requires a target or prediction layer name")
    family, _, index = name.partition(".")
    if family not in ("layers", "mtp") or index not in map(
        str, range(40 if family == "layers" else 3)
    ):
        raise ValueError("MoE requires a target or prediction layer name")
    experts = 384 if family == "layers" else 128
    plan = routing_plan(rows=rows, experts=experts)
    strategy, rank = owner.plan.strategy, owner.plan.rank
    if (
        strategy not in ("TP4", "TEP4", "DEP4")
        or type(rank) is not int
        or rank not in range(4)
    ):
        raise ValueError("MoE requires a four-rank placement")
    count = experts if strategy == "TP4" else experts // 4
    return plan, count, 0 if strategy == "TP4" else rank * count


def _weights(owner, name, count, start):
    prefix = f"{name}.ffn"
    dense = [
        owner.tensors[f"{prefix}.gate.{suffix}"]
        for suffix in ("weight", "bias", "bias_vl")
    ]
    routed = {
        part: tuple(
            owner.packed[f"{prefix}.experts.{expert}.{part}.weight"]
            for expert in range(start, start + count)
        )
        for part in ("w1", "w3", "w2")
    }
    shared = {
        part: owner.packed[f"{prefix}.shared_experts.{part}.weight"] for part in routed
    }
    width = 576 if owner.plan.strategy == "TP4" else 2304
    shared_width = 2304 if owner.plan.strategy == "DEP4" else 576
    for weights, intermediate, bits in (
        (routed, width, 4),
        (shared, shared_width, 8),
    ):
        for part, values in weights.items():
            outputs, columns = (
                (5120, intermediate) if part == "w2" else (intermediate, 5120)
            )
            for value in values if isinstance(values, tuple) else (values,):
                if (value.outputs, value.width, value.bits) != (
                    outputs,
                    columns,
                    bits,
                ):
                    raise ValueError("MoE weight placement differs from the rank plan")
    reads = [
        *dense,
        *(
            tensor
            for values in routed.values()
            for weight in values
            for tensor in (weight.encoded, weight.scales)
        ),
        *(
            tensor
            for weight in shared.values()
            for tensor in (weight.encoded, weight.scales)
        ),
    ]
    return dense, routed, shared, width, shared_width, reads


class Workspace:
    def __init__(self, owner, name, rows, allocate, byte_limit):
        self.owner, self.device = owner, owner.device
        self.plan, count, start = _configuration(owner, name, rows)
        if type(byte_limit) is not int or byte_limit <= 0:
            raise ValueError("MoE requires a positive workspace byte limit")
        require_eager(self.device)
        self.closed, self.parts = False, []
        self.allocated_bytes, self.byte_limit = count * 16 * 3, byte_limit
        self._writes = []
        dense, routed, shared, width, shared_width, reads = _weights(
            owner, name, count, start
        )
        self._tensor_device = dense[0].device
        if any(
            tensor.device != self._tensor_device or not tensor.is_contiguous()
            for tensor in reads
        ):
            raise ValueError("MoE weight device or layout differs")
        self._allocations = []
        # Read-only aliases are legal. Merge them before checking allocations.
        self._reads = []
        for first, last in sorted(set(map(_span, reads))):
            if self._reads and first <= self._reads[-1][1]:
                self._reads[-1] = (self._reads[-1][0], max(last, self._reads[-1][1]))
            else:
                self._reads.append((first, last))

        def reserve(shape, dtype):
            size = math.prod(shape) * _ELEMENT_BYTES[dtype]
            if self.allocated_bytes + size > byte_limit:
                raise ValueError("MoE workspace exceeds its byte limit")
            value = allocate(shape, dtype)
            if (
                tuple(value.shape) != tuple(shape)
                or str(value.dtype) != dtype
                or not value.is_contiguous()
                or value.device != self._tensor_device
            ):
                raise ValueError("MoE allocator returned invalid storage")
            self._allocations.append(value)
            span = _span(value)
            _disjoint(span, self._reads)
            _disjoint(span, self._writes)
            if span[0] != span[1]:
                self._writes.insert(bisect_left(self._writes, span), span)
            self.allocated_bytes += size
            return value

        def part(factory, *args, **kwargs):
            workspace = factory(*args, **kwargs, allocate=reserve)
            self.parts.append(workspace)
            return workspace

        capacity = rows * self.plan.topk
        try:
            self.gate = part(
                prepare_linear,
                linear_plan(
                    rows=rows,
                    outputs=self.plan.experts,
                    width=5120,
                    output_dtype="float32",
                ),
                self.device,
                linear.Weights(dense[0]),
            )
            self.route = part(
                prepare_routing, self.plan, self.device, routing.Weights(*dense[1:])
            )
            self.input_quantization = part(
                prepare_quantization,
                quantization_plan(rows=rows, width=5120, mode="fp8_ue8m0"),
                self.device,
            )
            self.dispatch = part(
                prepare_dispatch,
                make_dispatch_plan(
                    rows=rows,
                    experts=self.plan.experts,
                    expert_start=start,
                    expert_count=count,
                    encoded=True,
                ),
                self.device,
            )
            self.routed = {
                key: part(
                    prepare_grouped,
                    make_grouped_plan(
                        capacity=capacity,
                        groups=count,
                        outputs=5120 if key == "w2" else width,
                        width=width if key == "w2" else 5120,
                    ),
                    self.device,
                    weights,
                    prepared=key != "w2",
                )
                for key, weights in routed.items()
            }
            self.shared = {
                key: part(
                    prepare_projection,
                    projection_plan(
                        rows=rows, outputs=weights.outputs, width=weights.width, bits=8
                    ),
                    self.device,
                    weights,
                )
                for key, weights in shared.items()
            }
            self.routed_activation = part(
                prepare_activation,
                activation_plan(rows=capacity, width=width, routed=True),
                self.device,
            )
            self.shared_activation = part(
                prepare_activation,
                activation_plan(rows=rows, width=shared_width),
                self.device,
            )
            self.routed_quantization = part(
                prepare_quantization,
                quantization_plan(rows=capacity, width=width, mode="fp8_ue8m0"),
                self.device,
            )
            self.shared_quantization = part(
                prepare_quantization,
                quantization_plan(rows=rows, width=shared_width, mode="fp8_ue8m0"),
                self.device,
            )
            self.outputs = Outputs(
                self.dispatch.combined.value,
                self.shared["w2"].outputs.value,
                (
                    self.input_quantization.outputs.status,
                    self.dispatch.outputs.status,
                    *(workspace.outputs.status for workspace in self.routed.values()),
                    self.routed_quantization.outputs.status,
                    self.shared_quantization.outputs.status,
                    self.dispatch.combined.status,
                ),
            )
        except BaseException as error:
            try:
                self.device.synchronize_compute()
            except BaseException as completion:
                _FAILED_PREPARATIONS.append(self)
                error.add_note(f"MoE preparation completion also failed: {completion}")
            self.close()
            raise

    def check_open(self):
        if self.closed or self.owner.closed or self.owner._retained:
            raise RuntimeError("MoE workspace or rank weights are closed")

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

    def run(self, value, image_mask):
        self.check_open()
        for tensor, shape, dtype in (
            (value, (self.plan.rows, 5120), "bfloat16"),
            (image_mask, (self.plan.rows,), "uint8"),
        ):
            if (
                tuple(tensor.shape) != shape
                or str(tensor.dtype) != dtype
                or not tensor.is_contiguous()
                or tensor.device != self.outputs.routed.device
            ):
                raise ValueError("MoE input shape, dtype, layout, or device differs")
            _disjoint(_span(tensor), self._writes)
        logits = self.gate.run(linear.Inputs(value)).value
        routes = self.route.run(routing.Inputs(logits, image_mask))
        quantized = self.input_quantization.run(quantization.Inputs(value))
        dispatched = self.dispatch.run(
            routing.EncodedDispatchInputs(
                quantized.encoded, quantized.scales, quantized.status, *routes
            )
        )
        gate = self.routed["w1"].run(dispatched.grouped).value
        up = self.routed["w3"].run(dispatched.grouped).value
        activated = self.routed_activation.run(
            activation.Inputs(gate, up, dispatched.coefficients)
        ).value
        down_input = self.routed_quantization.run(quantization.Inputs(activated))
        down = (
            self.routed["w2"]
            .run(
                projection.GroupedInputs(
                    down_input.encoded,
                    down_input.scales,
                    dispatched.counts,
                    dispatched.offsets,
                )
            )
            .value
        )
        self.dispatch.combine(down)
        shared_input = projection.Inputs(quantized.encoded, quantized.scales)
        gate = self.shared["w1"].run(shared_input).value
        up = self.shared["w3"].run(shared_input).value
        activated = self.shared_activation.run(activation.Inputs(gate, up)).value
        down_input = self.shared_quantization.run(quantization.Inputs(activated))
        self.shared["w2"].run(projection.Inputs(down_input.encoded, down_input.scales))
        return self.outputs


def prepare(rank_weights, name, *, rows, allocate=None, byte_limit=256 << 20):
    """Bound workspace tensor payloads, including grouped pointer tables.

    Device allocator/context overhead and weights need separate caller reserves.
    A custom allocator must return disjoint storage on the rank device.
    """
    return Workspace(
        rank_weights,
        name,
        rows,
        rank_weights.device.empty if allocate is None else allocate,
        byte_limit,
    )
