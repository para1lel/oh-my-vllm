from contextlib import contextmanager
from typing import NamedTuple

from ..runtime.native import load
from .api import FUNCTIONS, Outputs


class Storage(NamedTuple):
    hidden: object
    output: object
    normalized: object
    projected: object
    query: object
    key: object
    value: object
    padded_attention: object
    normalizer: object
    attended: object
    activated: object


class Workspace:
    def __init__(self, plan, device, weights, allocate):
        self.module, self.source_id, self.library = load("vision", FUNCTIONS)
        self.plan, self.device, self.weights = plan, device, tuple(weights)
        values = tuple(allocate(*field) for field in plan.storage)
        self.storage = Storage(*values[:11])
        self.classic_storage = values[11] if plan.rows > 32 else None
        self.classic = (
            self.module.vision_classic_prepare(self.classic_storage)
            if self.classic_storage is not None
            else None
        )
        self.patches = values[-1] if plan.variant == "input" else None
        self.outputs = Outputs(self.storage.output)
        self.closed = False

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("vision workspace is closed")
        owners = (
            self.weights,
            self.storage,
            self.patches,
            self.classic_storage,
            self.module,
        )
        lease = (
            self.module.vision_classic_graph_lease(self.classic)
            if self.classic is not None
            else None
        )
        try:
            yield
        finally:
            del lease, owners

    def close(self):
        if not self.closed:
            if self.classic is not None:
                self.module.vision_classic_close(self.classic)
                self.classic = None
            self.closed = True

    def run(self, weights, inputs):
        if self.closed:
            raise RuntimeError("vision workspace is closed")
        p, s, value = self.plan, self.storage, inputs.value
        if p.variant == "input":
            self.module.vision_patch(
                value,
                *weights.patch,
                self.patches,
                s.projected,
                s.hidden,
                *p.grid,
                *weights.block[:2],
                s.normalized,
            )
            value = s.hidden
        block_output = s.hidden if p.variant == "merge" else s.output
        self.module.vision_block(
            value,
            *weights.block,
            block_output,
            s.normalized,
            s.projected,
            s.query,
            s.key,
            s.value,
            s.padded_attention,
            s.normalizer,
            s.attended,
            s.activated,
            *p.grid,
            self.classic,
            p.variant == "input",
        )
        if p.variant == "merge":
            self.module.vision_merge(
                block_output,
                *weights.merger,
                s.output,
                s.normalized,
                s.projected,
                s.activated,
            )
        return self.outputs


def prepare(plan, device, weights, *, allocate=None):
    return Workspace(
        plan, device, weights, device.empty if allocate is None else allocate
    )
