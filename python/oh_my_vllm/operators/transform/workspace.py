from contextlib import contextmanager

from ..runtime.native import load
from .api import FUNCTIONS, Outputs


class Workspace:
    def __init__(self, plan, device, allocate):
        self.module, self.source_id, self.library = load("transform", FUNCTIONS)
        self.plan, self.device = plan, device
        self.outputs = Outputs(*(allocate(*field) for field in plan.storage))
        self.closed = False

    def check_open(self):
        if self.closed:
            raise RuntimeError("transform workspace is closed")

    @contextmanager
    def graph_lease(self):
        self.check_open()
        owners = self.outputs, self.module
        try:
            yield
        finally:
            del owners

    def close(self):
        self.closed = True

    def run(self, inputs):
        self.check_open()
        if self.plan.kind == "rms":
            self.module.rms(*inputs, self.outputs.value, self.plan.epsilon)
        else:
            self.module.rope(*inputs, self.outputs.value, self.plan.inverse)
        return self.outputs


def prepare(plan, device, *, allocate=None):
    return Workspace(plan, device, device.empty if allocate is None else allocate)
