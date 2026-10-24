"""Rank-local MoE; collectives and state acceptance belong to the caller.

Routed output is global-row FP32. Shared output is local-row BF16. In DEP4,
encode only owner-local rows and exchange the encoded packets, not dense work.
Reject every nonzero status before committing state. Inputs stay owned through
Eager completion or graph destruction. Preparation and closure are Eager-only;
close invalidates Python execution, not existing graph leases.
Packets borrow reusable producer buffers. Consume them before encoding again;
retain external packets and their producer graph leases through completion.
"""

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
from ...operators.quantization import api as quantization
from ...operators.quantization.plan import make_plan as quantization_plan
from ...operators.quantization.workspace import prepare as prepare_quantization
from ...operators.routing import api as routing
from ...operators.routing.plan import make_dispatch_plan
from ...operators.routing.plan import make_plan as routing_plan
from ...operators.routing.workspace import prepare as prepare_routing
from ...operators.routing.workspace import prepare_dispatch
from .workspace import Workspace as RankWorkspace


class Outputs(NamedTuple):
    routed: object
    shared: object
    status: tuple


class Branch(NamedTuple):
    value: object
    status: tuple


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


class Workspace(RankWorkspace):
    def __init__(self, owner, name, rows, routed_rows, allocate, byte_limit):
        self.plan, count, start = _configuration(owner, name, rows)
        routed_rows = rows if routed_rows is None else routed_rows
        dispatch = make_dispatch_plan(
            rows=routed_rows,
            experts=self.plan.experts,
            expert_start=start,
            expert_count=count,
            encoded=True,
        )
        if rows > routed_rows or (
            owner.plan.strategy != "DEP4" and rows != routed_rows
        ):
            raise ValueError("MoE row domains differ from the rank placement")
        dense, routed, shared, width, shared_width, reads = _weights(
            owner, name, count, start
        )
        super().__init__(owner, reads, allocate, byte_limit, extra_bytes=count * 48)
        with self.preparing():
            self._prepare_encode(dense)
            first = len(self._allocations)
            self._prepare_shared(shared, shared_width)
            self._shared_writes = self.writes_since(first)
            first = len(self._allocations)
            self._prepare_routed(routed, width, dispatch)
            self._routed_writes = self.writes_since(first)
            self._routed_status = (
                self.dispatch.outputs.status,
                *(workspace.outputs.status for workspace in self.routed.values()),
                self.routed_quantization.outputs.status,
                self.dispatch.combined.status,
            )
            self.outputs = Outputs(
                self.dispatch.combined.value,
                self.shared["w2"].outputs.value,
                (
                    self.input_quantization.outputs.status,
                    *self._routed_status,
                    self.shared_quantization.outputs.status,
                ),
            )

    def _quantize(self, rows, width):
        return self.part(
            prepare_quantization,
            quantization_plan(rows=rows, width=width, mode="fp8_ue8m0"),
            self.device,
        )

    def _prepare_encode(self, dense):
        self.gate = self.part(
            prepare_linear,
            linear_plan(
                rows=self.plan.rows,
                outputs=self.plan.experts,
                width=5120,
                output_dtype="float32",
            ),
            self.device,
            linear.Weights(dense[0]),
        )
        self.route = self.part(
            prepare_routing, self.plan, self.device, routing.Weights(*dense[1:])
        )
        self.input_quantization = self._quantize(self.plan.rows, 5120)

    def _prepare_shared(self, weights, width):
        self.shared = {
            key: self.part(
                prepare_projection,
                projection_plan(
                    rows=self.plan.rows,
                    outputs=weight.outputs,
                    width=weight.width,
                    bits=8,
                ),
                self.device,
                weight,
            )
            for key, weight in weights.items()
        }
        self.shared_activation = self.part(
            prepare_activation,
            activation_plan(rows=self.plan.rows, width=width),
            self.device,
        )
        self.shared_quantization = self._quantize(self.plan.rows, width)

    def _prepare_routed(self, weights, width, plan):
        capacity = plan.rows * plan.topk
        self.dispatch = self.part(prepare_dispatch, plan, self.device)
        self.routed = {
            key: self.part(
                prepare_grouped,
                make_grouped_plan(
                    capacity=capacity,
                    groups=plan.expert_count,
                    outputs=5120 if key == "w2" else width,
                    width=width if key == "w2" else 5120,
                ),
                self.device,
                values,
                prepared=key != "w2",
            )
            for key, values in weights.items()
        }
        self.routed_activation = self.part(
            prepare_activation,
            activation_plan(rows=capacity, width=width, routed=True),
            self.device,
        )
        self.routed_quantization = self._quantize(capacity, width)

    def _check_packet(self, packet, rows, writes):
        self.check_open()
        if not isinstance(packet, routing.EncodedDispatchInputs):
            raise ValueError("MoE requires a quantized routing packet")
        for value, shape, dtype in zip(
            packet,
            ((rows, 5120), (rows, 160), (rows, 160), *((rows, self.plan.topk),) * 2),
            ("uint8", "uint8", "uint8", "int32", "float32"),
            strict=True,
        ):
            self.check_input(value, shape, dtype, writes=writes)

    def encode_routes(self, value, image_mask):
        self.check_open()
        for tensor, shape, dtype in (
            (value, (self.plan.rows, 5120), "bfloat16"),
            (image_mask, (self.plan.rows,), "uint8"),
        ):
            self.check_input(tensor, shape, dtype)
        logits = self.gate.run(linear.Inputs(value)).value
        routes = self.route.run(routing.Inputs(logits, image_mask))
        quantized = self.input_quantization.run(quantization.Inputs(value))
        return routing.EncodedDispatchInputs(
            quantized.encoded, quantized.scales, quantized.status, *routes
        )

    def run_routed(self, packet):
        self._check_packet(packet, self.dispatch.plan.rows, self._routed_writes)
        dispatched = self.dispatch.run(packet)
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
        return Branch(
            self.dispatch.combined.value, (packet.status, *self._routed_status)
        )

    def run_shared(self, packet):
        self._check_packet(packet, self.plan.rows, self._shared_writes)
        shared_input = projection.Inputs(packet.encoded, packet.scales)
        gate = self.shared["w1"].run(shared_input).value
        up = self.shared["w3"].run(shared_input).value
        activated = self.shared_activation.run(activation.Inputs(gate, up)).value
        down_input = self.shared_quantization.run(quantization.Inputs(activated))
        self.shared["w2"].run(projection.Inputs(down_input.encoded, down_input.scales))
        return Branch(
            self.shared["w2"].outputs.value,
            (packet.status, self.shared_quantization.outputs.status),
        )

    def run(self, value, image_mask):
        if self.plan.rows != self.dispatch.plan.rows:
            raise ValueError("MoE run requires equal local and routed row domains")
        packet = self.encode_routes(value, image_mask)
        self.run_routed(packet)
        self.run_shared(packet)
        return self.outputs


def prepare(
    rank_weights, name, *, rows, routed_rows=None, allocate=None, byte_limit=256 << 20
):
    """Bound workspace tensor payloads, including grouped pointer tables.

    Device allocator/context overhead and weights need separate caller reserves.
    A custom allocator must return disjoint storage on the rank device.
    """
    return Workspace(
        rank_weights,
        name,
        rows,
        routed_rows,
        allocate,
        byte_limit,
    )
