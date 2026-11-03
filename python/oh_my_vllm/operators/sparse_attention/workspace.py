from contextlib import contextmanager

from ..runtime.native import load
from .api import FUNCTIONS, Outputs
from .plan import scratch_bytes


class Workspace:
    def __init__(self, plan, device, allocate):
        self.module, self.source_id, self.library = load("sparse_attention", FUNCTIONS)
        self.plan, self.device = plan, device
        self.outputs = Outputs(*(allocate(*field) for field in plan.storage))
        self.scratch = allocate(
            (scratch_bytes(plan.rows, plan.heads, plan.slots),), "uint8"
        )
        self.closed = False

    def check_open(self):
        if self.closed:
            raise RuntimeError("sparse-attention workspace is closed")

    @contextmanager
    def graph_lease(self):
        self.check_open()
        owners = self.outputs, self.scratch, self.module
        try:
            yield
        finally:
            del owners

    def close(self):
        self.closed = True

    def run(self, inputs):
        self.check_open()
        if tuple(inputs.query.shape) != (self.plan.rows, self.plan.heads, 512) or tuple(
            inputs.indices.shape
        ) != (self.plan.rows, self.plan.slots):
            raise ValueError("sparse-attention input differs from the workspace plan")
        self.module.attend(*inputs, self.scratch, *self.outputs)
        return self.outputs


def prepare(plan, device, *, allocate=None):
    return Workspace(plan, device, device.empty if allocate is None else allocate)
