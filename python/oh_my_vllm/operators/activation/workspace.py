from contextlib import contextmanager

from ..runtime.native import load
from .api import FUNCTIONS, Outputs


class Workspace:
    def __init__(self, plan, device, allocate):
        self.module, self.source_id, self.library = load("activation", FUNCTIONS)
        self.plan, self.device = plan, device
        self.outputs = Outputs(*(allocate(*field) for field in plan.storage))
        self.empty_route = allocate((0,), "float32")
        self.closed = False

    def check_open(self):
        if self.closed:
            raise RuntimeError("activation workspace is closed")

    @contextmanager
    def graph_lease(self):
        self.check_open()
        owners = self.outputs, self.empty_route, self.module
        try:
            yield
        finally:
            del owners

    def close(self):
        self.closed = True

    def run(self, inputs):
        self.check_open()
        if (inputs.route is not None) != self.plan.routed:
            raise ValueError("SwiGLU routing input differs from the plan")
        self.module.swiglu(
            inputs.gate,
            inputs.up,
            inputs.route if self.plan.routed else self.empty_route,
            self.outputs.value,
            self.plan.limit,
            self.plan.routed,
        )
        return self.outputs


def prepare(plan, device, *, allocate=None):
    return Workspace(plan, device, device.empty if allocate is None else allocate)
