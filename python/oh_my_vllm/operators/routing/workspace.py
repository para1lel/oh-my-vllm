from contextlib import contextmanager

from ..runtime.native import load
from .api import FUNCTIONS, Outputs, Weights


class Workspace:
    def __init__(self, plan, device, weights, allocate):
        self.module, self.source_id, self.library = load("routing", FUNCTIONS)
        self.plan, self.device, self.weights = plan, device, Weights(*weights)
        self.storage = Outputs(*(allocate(*field) for field in plan.storage))
        self.outputs = self.storage
        self.closed = False

    @contextmanager
    def graph_lease(self):
        self.check_open()
        owners = self.weights, self.storage, self.module
        try:
            yield
        finally:
            del owners

    def check_open(self):
        if self.closed:
            raise RuntimeError("routing workspace is closed")

    def close(self):
        self.closed = True

    def run(self, inputs):
        self.check_open()
        self.module.route(
            inputs.logits, *self.weights, inputs.image_mask, *self.outputs
        )
        return self.outputs


def prepare(plan, device, weights, *, allocate=None):
    return Workspace(
        plan, device, weights, device.empty if allocate is None else allocate
    )
