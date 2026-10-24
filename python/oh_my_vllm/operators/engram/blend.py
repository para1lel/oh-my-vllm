"""Blend finite BF16 Engram keys/value into four residual copies.

Projected rows contain four keys followed by one shared value. Normalize each
copy separately with epsilon 1e-20, then apply the signed square root with a
1e-6 floor before sigmoid. A zero token mask preserves the residual.
Retain inputs through Eager completion or graph release.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple

from ..runtime.native import load
from .api import FUNCTIONS


class Inputs(NamedTuple):
    residual: object
    projected: object
    q_weight: object
    k_weight: object
    mask: object


class Outputs(NamedTuple):
    gate: object
    value: object


@dataclass(frozen=True)
class Plan:
    rows: int

    @property
    def storage(self):
        return (
            ((self.rows, 4), "float32"),
            ((self.rows, 4, 5120), "bfloat16"),
        )

    @property
    def reuse_key(self):
        return (self.rows,)


def make_plan(*, rows):
    if type(rows) is not int or not 0 <= rows <= 384:
        raise ValueError("Engram blend requires zero to 384 rows")
    return Plan(rows)


class Workspace:
    def __init__(self, plan, device, allocate):
        self.module, self.source_id, self.library = load("engram", FUNCTIONS)
        self.plan, self.device = plan, device
        self.outputs = Outputs(*(allocate(*field) for field in plan.storage))
        self.closed = False

    def check_open(self):
        if self.closed:
            raise RuntimeError("Engram blend workspace is closed")

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
        self.module.engram_blend(*inputs, *self.outputs)
        return self.outputs


def prepare(plan, device, *, allocate=None):
    return Workspace(plan, device, device.empty if allocate is None else allocate)
