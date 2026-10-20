from contextlib import contextmanager
from typing import NamedTuple

from ..runtime.native import load
from .api import FUNCTIONS, Outputs


class Storage(NamedTuple):
    value: object
    residual: object
    gathered: object
    normalized: object
    input: object
    input_scales: object
    qkvz: object
    ba: object
    qkv: object
    output: object
    output_scales: object
    recurrent: object
    partials: object
    gates: object
    chunk_metadata: object
    transforms: object
    prefixes: object
    snapshots: object
    coefficients: object
    empty_embedding: object
    empty_tokens: object
    empty_media: object
    empty_media_rows: object


class Workspace:
    def __init__(self, plan, device, weights, allocate):
        self.module, self.source_id, self.library = load("gdn", FUNCTIONS)
        self.plan, self.weights, self.device = plan, tuple(weights), device
        self.storage = Storage(*(allocate(*field) for field in plan.storage))
        self.native_storage = self.storage[:19]
        self.outputs = Outputs(self.storage.value, self.storage.residual)
        self.closed = False

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("GDN workspace is closed")
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

    def run(self, weights, inputs, batch, state):
        if self.closed:
            raise RuntimeError("GDN workspace is closed")
        w, s = weights, self.storage
        embedding = w.embedding is not None
        if embedding and inputs.tokens is None:
            raise ValueError("GDN embedding input requires token IDs")
        if (inputs.media is None) != (inputs.media_rows is None):
            raise ValueError("GDN media input requires media rows")
        self.module.gdn(
            inputs.value,
            inputs.residual,
            w.norm,
            w.qkvz,
            w.qkvz_scale,
            w.ba,
            w.convolution,
            w.a_log,
            w.dt_bias,
            w.head_norm,
            w.output,
            w.output_scale,
            state.recurrent,
            state.convolution,
            batch.sequence_rows,
            batch.read_slots,
            batch.write_slots,
            w.embedding if embedding else s.empty_embedding,
            inputs.tokens if embedding else s.empty_tokens,
            inputs.media if inputs.media is not None else s.empty_media,
            inputs.media_rows if inputs.media_rows is not None else s.empty_media_rows,
            embedding,
            self.plan.verify,
            *self.native_storage,
            self.plan.max_queries,
        )
        return self.outputs


def prepare(plan, device, weights, *, allocate=None):
    return Workspace(
        plan, device, weights, device.empty if allocate is None else allocate
    )
