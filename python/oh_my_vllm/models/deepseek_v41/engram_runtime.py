"""Pinned-host lookup, encoded projection, and gated residual update.

The caller supplies a registered shared table and the selected layer's layout.
Hash IDs include per-head row offsets. Token mask zero disables the contribution.
Hashes, history, row identities, and state acceptance stay with the caller.
Reject every returned status before state commit. Returned tensors borrow reusable
storage; retain inputs through Eager completion or graph destruction.
"""

from typing import NamedTuple

from ...operators.engram import blend, lookup
from . import linear
from .engram import Table
from .workspace import Workspace as RankWorkspace


class Outputs(NamedTuple):
    value: object
    gate: object
    status: tuple


class Workspace(RankWorkspace):
    def __init__(self, owner, layer, table, layout, rows, allocate, byte_limit):
        if owner.closed or owner._retained:
            raise RuntimeError("rank weights are closed or failed")
        if (
            type(layer) is not int
            or layer not in (1, 14)
            or not isinstance(layout, Table)
        ):
            raise ValueError("Engram requires a target layer and table layout")
        prefix = f"layers.{layer}.engram"
        weight = owner.packed[prefix + ".wkv.weight"]
        self.q_weight, self.k_weight = (
            owner.tensors[prefix + "." + name] for name in ("q_weight", "k_weight")
        )
        if (weight.outputs, weight.width, weight.bits) != (25600, 6144, 8) or any(
            tuple(value.shape) != (4, 5120) or str(value.dtype) != "bfloat16"
            for value in (self.q_weight, self.k_weight)
        ):
            raise ValueError("Engram projection or gate weights differ")
        self.plan = blend.make_plan(rows=rows)
        lookup_plan = lookup.make_plan(
            rows=rows,
            heads=24,
            table_rows=layout.rows,
            table_offset=layout.table_offset,
            scale_offset=layout.scale_offset,
        )
        super().__init__(
            owner,
            [self.q_weight, self.k_weight, weight.encoded, weight.scales],
            allocate,
            byte_limit,
        )
        with self.preparing():
            self.lookup = self.part(lookup.prepare, lookup_plan, self.device, table)
            self.projection = self.part(
                linear.prepare, owner, prefix + ".wkv.weight", rows=rows
            )
            self.blend = self.part(blend.prepare, self.plan, self.device)
            self._lookups = self.device.view(self.lookup.outputs.value, (rows, 6144))
            self._projected = self.device.view(
                self.projection.outputs.value, (rows, 5, 5120)
            )
            self.outputs = Outputs(
                self.blend.outputs.value,
                self.blend.outputs.gate,
                (self.lookup.outputs.status, self.projection.outputs.status),
            )

    def run(self, residual, indices, token_mask):
        self.check_open()
        rows = self.plan.rows
        for value, shape, dtype in (
            (residual, (rows, 4, 5120), "bfloat16"),
            (indices, (rows, 24), "int64"),
            (token_mask, (rows,), "uint8"),
        ):
            self.check_input(value, shape, dtype)
        self.lookup.run(indices)
        self.projection.run(self._lookups)
        self.blend.run(
            blend.Inputs(
                residual, self._projected, self.q_weight, self.k_weight, token_mask
            )
        )
        return self.outputs


def prepare(
    rank_weights, layer, *, table, layout, rows, allocate=None, byte_limit=128 << 20
):
    return Workspace(rank_weights, layer, table, layout, rows, allocate, byte_limit)
