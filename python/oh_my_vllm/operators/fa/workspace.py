from contextlib import contextmanager

from ..runtime.native import load
from .api import FUNCTIONS, Outputs
from .components import Storage


class Workspace:
    def __init__(self, plan, device, weights, allocate):
        self.module, self.source_id, self.library = load("fa", FUNCTIONS)
        self.plan, self.weights, self.device = plan, tuple(weights), device
        self.storage = Storage(*(allocate(*field) for field in plan.storage))
        self.outputs = Outputs(
            None if plan.variant == 1 else self.storage.value, self.storage.residual
        )
        self.closed = False
        self.tables = None

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("FA workspace is closed")
        owners = (self.weights, self.storage, self.module)
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
        if self.tables is not state.tables:
            if tuple(state.tables.shape) != (self.plan.batch, self.plan.table_width):
                raise ValueError("FA cache table layout differs from the prepared plan")
            self.tables = state.tables

    def run(self, weights, inputs, batch, state):
        if self.closed:
            raise RuntimeError("FA workspace is closed")
        self.bind_layout(state)
        w, p = weights, self.plan
        self.module.fa(
            inputs.value,
            inputs.residual,
            w.norm,
            w.projection,
            w.projection_scale,
            w.query_norm,
            w.key_norm,
            batch.positions,
            *state.cache,
            state.slots,
            state.tables,
            batch.sequence_rows,
            batch.lengths,
            w.output,
            w.output_scale,
            *self.storage,
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
