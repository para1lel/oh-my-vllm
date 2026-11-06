from contextlib import contextmanager

import tvm_ffi

from ..runtime.native import load
from .api import FUNCTIONS, Outputs
from .components import Storage
from .layout import validate_cache


class Workspace:
    def __init__(self, plan, device, weights, allocate):
        self.module, self.source_id, self.library = load("fa_fp8", FUNCTIONS)
        self.plan, self.weights, self.device = plan, tuple(weights), device
        self.storage = Storage(*(allocate(*field) for field in plan.storage))
        self.outputs = Outputs(
            None if plan.variant == 1 else self.storage.value, self.storage.residual
        )
        self.ranges = tvm_ffi.Shape(
            tuple(x for span in plan.write_ranges for x in span)
        )
        self.closed = False
        self.bindings = []

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("FP8 FA workspace is closed")
        owners = (self.weights, self.storage, self.module, self.bindings, self.ranges)
        try:
            yield
        finally:
            del owners

    def close(self):
        self.closed = True

    def warmup(self, weights, inputs, batch, state):
        if self.plan.needs_warmup:
            self.run(weights, inputs, batch, state)

    def bind_layout(self, state):
        validate_cache(state.codes, state.scales)
        if tuple(state.tables.shape) != (self.plan.batch, self.plan.table_width):
            raise ValueError("FP8 FA cache table differs from the prepared plan")
        pair = (state.codes, state.scales, state.tables, state.slots, state.write_mask)
        if not any(
            all(a is b for a, b in zip(pair, old, strict=True)) for old in self.bindings
        ):
            self.bindings.append(pair)

    def run(self, weights, inputs, batch, state):
        if self.closed:
            raise RuntimeError("FP8 FA workspace is closed")
        self.bind_layout(state)
        w, p = weights, self.plan
        self.module.fa_fp8_run(
            inputs.value,
            inputs.residual,
            w.norm,
            w.projection,
            w.projection_scale,
            w.query_norm,
            w.key_norm,
            batch.positions,
            state.codes,
            state.scales,
            state.slots,
            state.tables,
            batch.sequence_rows,
            batch.lengths,
            state.slots if state.write_mask is None else state.write_mask,
            state.write_mask is not None,
            w.output,
            w.output_scale,
            *self.storage,
            self.ranges,
            p.variant,
            p.max_queries,
            p.first_token,
            p.max_context,
            p.min_context,
        )
        return self.outputs


def prepare(plan, device, weights, *, allocate=None):
    return Workspace(
        plan, device, weights, device.empty if allocate is None else allocate
    )
