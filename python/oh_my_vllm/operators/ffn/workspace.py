from contextlib import contextmanager
from typing import NamedTuple

from ..runtime.native import load
from .api import FUNCTIONS, Outputs


class Storage(NamedTuple):
    value: object
    residual: object
    input: object
    input_scales: object
    gate_up: object
    product: object
    product_scales: object
    partials: object


class Workspace:
    def __init__(self, plan, device, weights, allocate):
        self.module, self.source_id, self.library = load("ffn", FUNCTIONS)
        self.plan, self.weights, self.device = plan, tuple(weights), device
        self.storage = Storage(*(allocate(*field) for field in plan.storage))
        self.outputs = Outputs(self.storage.value, self.storage.residual)
        self.closed = False

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("FFN workspace is closed")
        owners = (self.weights, self.storage, self.module)
        try:
            yield
        finally:
            del owners

    def close(self):
        self.closed = True

    def warmup(self, weights, inputs):
        if self.plan.needs_warmup:
            self.run(weights, inputs)

    def run(self, weights, inputs):
        if self.closed:
            raise RuntimeError("FFN workspace is closed")
        w = weights
        self.module.ffn(
            inputs.value,
            inputs.residual,
            w.norm,
            w.gate_up,
            w.gate_up_scale,
            w.down,
            w.down_scale,
            *self.storage,
        )
        return self.outputs


def prepare(plan, device, weights, *, allocate=None):
    return Workspace(
        plan, device, weights, device.empty if allocate is None else allocate
    )
