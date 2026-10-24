"""Rank-local source compression with caller-owned pair and cache state.

Encode normalized BF16 attention inputs, stage complete pairs externally, then
finish before packing. Finish returns normalized BF16 latents before RoPE so
an index-key consumer can read them first. Pack rotates only a separate copy.
Finite projection/RMS inputs and the pooling operator's numerical domain are
preconditions. Reject every returned status before publishing state.
Check encode status before retaining a partial group; the next encode resets it.

All results borrow reusable storage. Retain inputs, staged pairs, coefficients,
weights and graph leases through Eager completion or graph destruction.
"""

from dataclasses import dataclass
from typing import NamedTuple

from ...operators.compression import api as compression
from ...operators.compression.plan import make_plan as pooling_plan
from ...operators.compression.workspace import prepare as prepare_pooling
from ...operators.linear import api as linear
from ...operators.linear.plan import make_plan as linear_plan
from ...operators.linear.workspace import prepare as prepare_linear
from ...operators.quantization import api as quantization
from ...operators.quantization.plan import make_plan as quantization_plan
from ...operators.quantization.workspace import prepare as prepare_quantization
from ...operators.transform import api as transform
from ...operators.transform.plan import make_plan as transform_plan
from ...operators.transform.workspace import prepare as prepare_transform
from .workspace import Workspace as RankWorkspace


@dataclass(frozen=True)
class Plan:
    rows: int
    completed_rows: int
    ratio: int


class Encoded(NamedTuple):
    kv: object
    gates: object | None
    status: tuple


class Outputs(NamedTuple):
    value: object
    status: tuple


class Packed(NamedTuple):
    encoded: object
    scales: object
    status: tuple


def _weights(owner, layer):
    if owner.closed or owner._retained:
        raise RuntimeError("rank weights are closed or failed")
    if (
        not isinstance(layer, str)
        or layer not in ("layers.2", "layers.8", "layers.14", "layers.20")
        or owner.plan.strategy not in ("TP4", "TEP4", "DEP4")
        or type(owner.plan.rank) is not int
        or owner.plan.rank not in range(4)
    ):
        raise ValueError("compressor requires a four-rank source layer")
    ratio = 1 if layer == "layers.20" else 2
    dtype = "float32" if ratio == 2 else "bfloat16"
    shapes = {"wkv.weight": ((512, 5120), dtype), "norm.weight": ((512,), "bfloat16")}
    if ratio == 2:
        shapes["wgate.weight"] = ((512, 5120), dtype)
    weights = {}
    for name, (shape, kind) in shapes.items():
        value = owner.tensors.get(f"{layer}.attn.compressor.{name}")
        if value is None or tuple(value.shape) != shape or str(value.dtype) != kind:
            raise ValueError("compressor weight differs from the source rank plan")
        weights[name] = value
    return ratio, weights


class Workspace(RankWorkspace):
    def __init__(self, owner, layer, rows, completed_rows, allocate, byte_limit):
        ratio, self.weights = _weights(owner, layer)
        if (
            type(rows) is not int
            or type(completed_rows) is not int
            or not 0 <= completed_rows <= rows <= 384
            or (ratio == 1 and completed_rows != rows)
        ):
            raise ValueError("compressor input and complete-group row counts differ")
        self.plan = Plan(rows, completed_rows, ratio)
        super().__init__(owner, list(self.weights.values()), allocate, byte_limit)
        self._phase = "idle"
        with self.preparing():
            status = ()
            if ratio == 2:
                self.widen = self._transform("widen", rows, 5120)
                status = (self.widen.outputs.status,)
            self.kv = self._linear(self.weights["wkv.weight"])
            self.gate = (
                self._linear(self.weights["wgate.weight"]) if ratio == 2 else None
            )
            self.encoded = Encoded(
                self.kv.outputs.value,
                self.gate.outputs.value if self.gate is not None else None,
                status,
            )
            first = len(self._allocations)
            if ratio == 2:
                self.pool = self.part(
                    prepare_pooling, pooling_plan(rows=completed_rows), self.device
                )
            self.norm = self._transform("rms", completed_rows, 512, epsilon=1e-20)
            self.outputs = Outputs(self.norm.outputs.value, status)
            self.rope = self._transform("rope", completed_rows, 512)
            self._latent_head = self.device.view(
                self.outputs.value, (completed_rows, 1, 512)
            )
            self._rotated = self.device.view(
                self.rope.outputs.value, (completed_rows, 512)
            )
            self.quantization = self.part(
                prepare_quantization,
                quantization_plan(rows=completed_rows, width=512, mode="fp4_e4m3"),
                self.device,
            )
            self.packed = Packed(
                self.quantization.outputs.encoded,
                self.quantization.outputs.scales,
                (*status, self.quantization.outputs.status),
            )
            self._finish_writes = self.writes_since(first)

    def _linear(self, weight):
        dtype = str(weight.dtype)
        return self.part(
            prepare_linear,
            linear_plan(
                rows=self.plan.rows,
                outputs=512,
                width=5120,
                input_dtype=dtype,
                output_dtype=dtype,
            ),
            self.device,
            linear.Weights(weight),
        )

    def _transform(self, kind, rows, width, **options):
        return self.part(
            prepare_transform,
            transform_plan(kind=kind, rows=rows, width=width, **options),
            self.device,
        )

    def _enter(self, phase):
        self.check_open()
        if self._phase != phase:
            raise RuntimeError(f"compressor requires {phase} computation first")
        self._phase = "failed"

    def encode(self, value):
        self.check_open()
        self._phase = "failed"
        self.check_input(value, (self.plan.rows, 5120), "bfloat16")
        if self.plan.ratio == 2:
            value = self.widen.run(transform.Inputs(value, None)).value
            self.gate.run(linear.Inputs(value))
        self.kv.run(linear.Inputs(value))
        self._phase = "encoded"
        return self.encoded

    def finish(self, kv, gates=None):
        self._enter("encoded")
        rows = self.plan.completed_rows
        if kv is None:
            raise ValueError("compressor requires projected KV values")
        if self.plan.ratio == 2:
            if gates is None:
                raise ValueError("ratio2 compressor requires KV and gate pairs")
            for value in (kv, gates):
                self.check_input(
                    value, (rows, 2, 512), "float32", writes=self._finish_writes
                )
            value = self.pool.run(compression.Inputs(kv, gates)).value
        else:
            if gates is not None:
                raise ValueError("ratio1 compressor does not take gates")
            self.check_input(kv, (rows, 512), "bfloat16", writes=self._finish_writes)
            value = kv
        self.norm.run(transform.Inputs(value, self.weights["norm.weight"]))
        self._phase = "finished"
        return self.outputs

    def pack(self, coefficients):
        self._enter("finished")
        self.check_input(coefficients, (self.plan.completed_rows, 32, 2), "float32")
        self.rope.run(transform.Inputs(self._latent_head, coefficients))
        self.quantization.run(quantization.Inputs(self._rotated))
        self._phase = "packed"
        return self.packed


def prepare(
    owner,
    layer,
    *,
    rows,
    completed_rows,
    allocate=None,
    byte_limit=128 << 20,
):
    """Prepare fixed row domains without allocating pair state or publishing KV."""
    return Workspace(owner, layer, rows, completed_rows, allocate, byte_limit)
