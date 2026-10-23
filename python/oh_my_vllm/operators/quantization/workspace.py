from contextlib import contextmanager

from ..runtime.native import load
from .api import FUNCTIONS, Outputs


class Workspace:
    def __init__(self, plan, device, allocate):
        self.module, self.source_id, self.library = load("quantization", FUNCTIONS)
        self.plan, self.device = plan, device
        self.outputs = Outputs(*(allocate(*field) for field in plan.storage))
        self.closed = False

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("quantization workspace is closed")
        owners = (self.outputs, self.module)
        try:
            yield
        finally:
            del owners

    def close(self):
        self.closed = True

    def run(self, inputs):
        if self.closed:
            raise RuntimeError("quantization workspace is closed")
        self.module.quantize(inputs.value, *self.outputs, self.plan.native_mode)
        return self.outputs


def prepare(plan, device, *, allocate=None):
    return Workspace(plan, device, device.empty if allocate is None else allocate)
