from contextlib import contextmanager

from ..runtime.native import load
from .api import FUNCTIONS, Outputs
from .plan import WORKSPACE_BYTES


class Workspace:
    def __init__(self, plan, device, weights, allocate):
        if (
            tuple(weights.value.shape) != plan.weight_shape
            or str(weights.value.dtype) != plan.input_dtype
        ):
            raise ValueError("unquantized projection weights differ from the plan")
        self.module, self.source_id, self.library = load("linear", FUNCTIONS)
        self.plan, self.device, self.weights = plan, device, weights
        self.outputs = Outputs(allocate(plan.output_shape, plan.output_dtype))
        self.scratch = allocate((WORKSPACE_BYTES,), "uint8")
        self.owner = self.module.linear_prepare(self.scratch)
        self.closed = False

    def check_open(self):
        if self.closed:
            raise RuntimeError("unquantized projection workspace is closed")

    @contextmanager
    def graph_lease(self):
        self.check_open()
        owners = (self.owner, self.weights, self.outputs, self.scratch, self.module)
        lease = self.module.linear_graph_lease(self.owner)
        try:
            yield
        finally:
            del lease, owners

    def close(self):
        # Existing graph leases keep the handle, workspace, and tensor owners.
        self.closed = True

    def run(self, inputs):
        self.check_open()
        self.module.linear_project(
            self.owner,
            inputs.value,
            self.weights.value,
            self.outputs.value,
            self.plan.groups,
        )
        return self.outputs


def prepare(plan, device, weights, *, allocate=None):
    return Workspace(
        plan, device, weights, device.empty if allocate is None else allocate
    )
