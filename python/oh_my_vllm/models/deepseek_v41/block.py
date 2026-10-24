"""Staged rank-local HC, attention, and MoE computation.

The caller prepares Engram or prediction-layer inputs, selects and stages KV,
completes collectives, and aligns completed branches to owner-local rows.
Prediction prefill only seeds KV and does not execute this block.
This component neither selects physical state nor commits logical state.
Reject every returned status and collective error before accepting state.
Inputs, external packets, and their producer leases remain caller-owned through
Eager completion or graph destruction. Returned tensors borrow reusable storage.
"""

from typing import NamedTuple

from ...operators.hyperconnection import api as hyperconnection
from ...operators.hyperconnection.plan import make_plan as hc_plan
from ...operators.hyperconnection.workspace import prepare as prepare_hc
from ...operators.routing.api import EncodedDispatchInputs
from ...operators.transform import api as transform
from ...operators.transform.plan import make_plan as transform_plan
from ...operators.transform.workspace import prepare as prepare_transform
from . import attention, moe
from .workspace import Workspace as RankWorkspace
from .workspace import _span


class Outputs(NamedTuple):
    residual: object
    next_pre: object
    status: tuple


def _weights(owner, name):
    shapes = {
        "attn_norm.weight": ((5120,), "bfloat16"),
        "ffn_norm.weight": ((5120,), "bfloat16"),
        **{
            f"hc_{sublayer}_{part}": (shape, "float32")
            for sublayer in ("attn", "ffn")
            for part, shape in (("fn", (24, 20480)), ("scale", (3,)), ("base", (24,)))
        },
    }
    weights = {key: owner.tensors[f"{name}.{key}"] for key in shapes}
    for key, (shape, dtype) in shapes.items():
        value = weights[key]
        if tuple(value.shape) != shape or str(value.dtype) != dtype:
            raise ValueError("block normalization or HC weight differs")
    prefix = name + "."
    reads = [value for key, value in owner.tensors.items() if key.startswith(prefix)]
    reads.extend(
        value
        for key, weight in owner.packed.items()
        if key.startswith(prefix)
        for value in (weight.encoded, weight.scales)
    )
    return weights, reads


class Workspace(RankWorkspace):
    def __init__(self, owner, name, rows, routed_rows, slots, allocate, byte_limit):
        _, experts, _ = moe._configuration(owner, name, rows)
        routed_rows = rows if routed_rows is None else routed_rows
        if (
            type(routed_rows) is not int
            or not rows <= routed_rows <= 384
            or (owner.plan.strategy != "DEP4" and rows != routed_rows)
        ):
            raise ValueError("block row domains differ from the rank placement")
        self.plan = hc_plan(rows=rows)
        self.weights, reads = _weights(owner, name)
        super().__init__(owner, reads, allocate, byte_limit, extra_bytes=experts * 48)
        self._phase, self._residual = "idle", None
        self._shared = self._routed = None
        with self.preparing():
            self.hc_attn = self._hc("attn")
            self.attn_norm = self._norm()
            self.attention = self.part(
                attention.prepare,
                owner,
                name,
                rows=rows,
                slots=slots,
                byte_limit=byte_limit,
            )
            self._attention_writes = self.writes_since(0)
            first = len(self._allocations)
            self.hc_ffn = self._hc("ffn")
            self.ffn_norm = self._norm()
            self.moe = self.part(
                moe.prepare,
                owner,
                name,
                rows=rows,
                routed_rows=routed_rows,
                byte_limit=byte_limit,
            )
            self._ffn_writes = sorted(
                [_span(self.hc_attn.outputs.expanded), *self.writes_since(first)]
            )
            self._finish_writes = [_span(self.hc_ffn.outputs.expanded)]
            self.outputs = Outputs(
                self.hc_ffn.outputs.expanded, self.hc_ffn.outputs.next_pre, ()
            )

    def _hc(self, sublayer):
        return self.part(
            prepare_hc,
            self.plan,
            self.device,
            hyperconnection.Weights(
                *(
                    self.weights[f"hc_{sublayer}_{part}"]
                    for part in ("fn", "scale", "base")
                )
            ),
        )

    def _norm(self):
        return self.part(
            prepare_transform,
            transform_plan(kind="rms", rows=self.plan.rows, width=5120, epsilon=1e-20),
            self.device,
        )

    def _enter(self, phase):
        self.check_open()
        if self._phase != phase:
            raise RuntimeError(f"block requires {phase} computation first")
        self._phase = "failed"

    def begin(self, residual, incoming_pre, coefficients):
        self.check_open()
        self._phase, self._residual = "failed", None
        self._shared = self._routed = None
        rows = self.plan.rows
        for value, shape, dtype in (
            (residual, (rows, 4, 5120), "bfloat16"),
            (incoming_pre, (rows, 4), "float32"),
            (coefficients, (rows, 32, 2), "float32"),
        ):
            self.check_input(value, shape, dtype, writes=self._attention_writes)
        collapsed, _ = self.hc_attn.begin(residual, incoming_pre)
        normalized = self.attn_norm.run(
            transform.Inputs(collapsed, self.weights["attn_norm.weight"])
        ).value
        encoded = self.attention.encode(normalized, coefficients)
        self._residual, self._phase = residual, "encoded"
        return encoded

    def attend(self, indices, window, window_scales, compressed, compressed_scales):
        self._enter("encoded")
        result = self.attention.attend(
            indices, window, window_scales, compressed, compressed_scales
        )
        self._phase = "attended"
        return result

    def ffn(self, attention_complete, image_mask):
        self._enter("attended")
        dtype = str(attention_complete.dtype)
        if dtype not in ("bfloat16", "float32"):
            raise ValueError("completed attention requires BF16 or FP32 values")
        self.check_input(
            attention_complete,
            (self.plan.rows, 5120),
            dtype,
            writes=self._ffn_writes,
        )
        self.check_input(
            image_mask, (self.plan.rows,), "uint8", writes=self._ffn_writes
        )
        residual = self.hc_attn.finish(self._residual, attention_complete)
        collapsed, _ = self.hc_ffn.begin(residual, self.hc_attn.outputs.next_pre)
        normalized = self.ffn_norm.run(
            transform.Inputs(collapsed, self.weights["ffn_norm.weight"])
        ).value
        packet = self.moe.encode_routes(normalized, image_mask)
        self._shared = self.moe.run_shared(packet)
        self._phase = "ffn"
        return packet, self._shared

    def run_routed(self, packet):
        self._enter("ffn")
        if not isinstance(packet, EncodedDispatchInputs):
            raise ValueError("block requires a quantized routing packet")
        self.check_input(
            packet.status,
            (self.moe.dispatch.plan.rows, 160),
            "uint8",
            writes=self._finish_writes,
        )
        self._routed = self.moe.run_routed(packet)
        self._phase = "routed"
        return self._routed

    def finish(self, routed_complete, shared_complete):
        self._enter("routed")
        residual = self.hc_ffn.finish_moe(
            self.hc_attn.outputs.expanded, routed_complete, shared_complete
        )
        self.outputs = Outputs(
            residual,
            self.hc_ffn.outputs.next_pre,
            (
                *self.attention.outputs.status,
                *self._shared.status,
                *self._routed.status,
            ),
        )
        self._phase = "finished"
        return self.outputs


def prepare(
    rank_weights,
    name,
    *,
    rows,
    slots,
    routed_rows=None,
    allocate=None,
    byte_limit=512 << 20,
):
    """Prepare bounded computation with caller-owned collective and state buffers."""
    return Workspace(rank_weights, name, rows, routed_rows, slots, allocate, byte_limit)
