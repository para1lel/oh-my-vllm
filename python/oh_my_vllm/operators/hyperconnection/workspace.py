from contextlib import contextmanager
from typing import NamedTuple

from ..runtime.native import load
from .api import FUNCTIONS, Outputs


class Storage(NamedTuple):
    collapsed: object
    expanded: object
    next_pre: object
    mixed: object
    post: object
    comb: object


class Workspace:
    def __init__(self, plan, device, weights, allocate):
        self.module, self.source_id, self.library = load("hyperconnection", FUNCTIONS)
        self.plan, self.device, self.weights = plan, device, weights
        self.storage = Storage(*(allocate(*field) for field in plan.storage))
        self.outputs = Outputs(*self.storage[:3])
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
            raise RuntimeError("HC workspace is closed")

    def close(self):
        self.closed = True

    def begin(self, residual, incoming_pre):
        self.check_open()
        storage = self.storage
        self.module.hc_prepare(
            residual,
            incoming_pre,
            *self.weights,
            storage.mixed,
            storage.post,
            storage.comb,
            storage.collapsed,
            storage.next_pre,
        )
        return self.outputs.collapsed, self.outputs.next_pre

    def finish(self, residual, branch):
        self.check_open()
        storage = self.storage
        self.module.hc_expand(
            residual, branch, storage.post, storage.comb, storage.expanded
        )
        return self.outputs.expanded

    def run(self, inputs):
        self.begin(inputs.residual, inputs.incoming_pre)
        self.finish(inputs.residual, inputs.branch)
        return self.outputs


def prepare(plan, device, weights, *, allocate=None):
    return Workspace(
        plan, device, weights, device.empty if allocate is None else allocate
    )
