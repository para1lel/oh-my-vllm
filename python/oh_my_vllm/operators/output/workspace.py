from contextlib import contextmanager
from typing import NamedTuple

from ..runtime.native import load
from .api import FUNCTIONS, Outputs


class Storage(NamedTuple):
    hidden: object
    logits: object
    keys: object
    ordered: object
    sorted_probabilities: object
    indices: object
    probabilities: object
    tokens: object
    selected_probability: object
    sort_workspace: object


class Workspace:
    def __init__(self, plan, device, weights, allocate):
        self.module, self.source_id, self.library = load("sample", FUNCTIONS)
        self.plan, self.device, self.weights = plan, device, tuple(weights)
        fields = tuple(allocate(*field) for field in plan.storage)
        self.closed = False
        if plan.hidden_only:
            self.storage = None
            self.outputs = Outputs(fields[0])
        else:
            temporary = allocate(
                (max(1, self.module.sample_workspace(plan.rows)),), "uint8"
            )
            self.storage = Storage(*fields, temporary)
            self.outputs = Outputs(
                fields[0], fields[1], fields[6], fields[7], fields[8]
            )
            self.draw_storage = self.storage[2:]
            self.distribution_storage = self.storage[2:7]

    @contextmanager
    def graph_lease(self):
        self.check_open()
        owners = (self.weights, self.storage, self.outputs, self.module)
        try:
            yield
        finally:
            del owners

    def close(self):
        self.closed = True

    def check_open(self):
        if self.closed:
            raise RuntimeError("output workspace is closed")

    def hidden(self, weights, inputs, state):
        self.check_open()
        self.module.final_hidden(
            inputs.value,
            inputs.residual,
            weights.norm,
            self.outputs.hidden,
            state.cache,
            state.slots,
        )
        return self.outputs

    def project(self, weights, inputs):
        self.check_open()
        self.module.sample_project(
            inputs.value,
            inputs.residual,
            weights.norm,
            weights.projection,
            self.outputs.hidden,
            self.outputs.logits,
        )
        return self.outputs

    def draw(self, settings, rng):
        self.check_open()
        self.module.sample(
            self.outputs.logits,
            settings.history,
            settings.mask,
            settings.parameters,
            rng,
            *self.draw_storage,
            self.plan.allow_invalid,
            settings.greedy,
        )
        return self.outputs

    def distribution(self, settings):
        self.check_open()
        self.module.sample_distribution(
            self.outputs.logits,
            settings.history,
            settings.mask,
            settings.parameters,
            *self.distribution_storage,
            self.storage.sort_workspace,
            self.plan.allow_invalid,
            settings.greedy,
        )
        return self.outputs

    def advance_presence(self, history):
        self.check_open()
        self.module.sample_advance_presence(history, self.outputs.tokens)
        return self.outputs


def prepare(plan, device, weights, *, allocate=None):
    return Workspace(
        plan, device, weights, device.empty if allocate is None else allocate
    )
