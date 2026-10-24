from contextlib import contextmanager

from ..runtime.native import load
from .api import FUNCTIONS, CombineOutputs, DispatchOutputs, Outputs, Weights


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


class DispatchWorkspace(Workspace):
    def __init__(self, plan, device, allocate):
        self.module, self.source_id, self.library = load("routing", FUNCTIONS)
        self.plan, self.device, self.weights = plan, device, ()
        self.outputs = DispatchOutputs(*(allocate(*field) for field in plan.storage))
        self.combined = CombineOutputs(
            *(allocate(*field) for field in plan.combine_storage)
        )
        self.source_rows = allocate(*plan.scratch)
        self.storage = self.outputs, self.combined, self.source_rows
        self.closed = False

    def run(self, inputs):
        self.check_open()
        self.module.dispatch(
            *inputs,
            *self.outputs,
            self.source_rows,
            self.plan.experts,
            self.plan.expert_start,
            self.plan.expert_count,
        )
        return self.outputs

    def combine(self, down):
        self.check_open()
        self.module.combine(
            down, self.outputs.source_map, self.outputs.status, *self.combined
        )
        return self.combined


def prepare_dispatch(plan, device, *, allocate=None):
    return DispatchWorkspace(
        plan, device, device.empty if allocate is None else allocate
    )
