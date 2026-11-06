import hashlib
from contextlib import ExitStack, contextmanager

from ..fa_fp8.workspace import prepare as prepare_attention
from ..ffn.workspace import prepare as prepare_ffn
from ..output.workspace import prepare as prepare_output
from ..runtime.native import load
from .api import FUNCTIONS, AcceptOutputs, ContextOutputs
from .input import InputWorkspace


def identity(operations):
    source = hashlib.sha256(
        "".join(op.source_id for op in operations).encode()
    ).hexdigest()
    libraries = tuple(
        dict.fromkeys(
            path
            for op in operations
            for path in getattr(op, "libraries", (op.library,))
        )
    )
    return source, libraries


class ContextWorkspace:
    def __init__(self, plan, device, weights, allocate):
        self.plan, self.device, self.weights = plan, device, tuple(weights)
        self.input = InputWorkspace(
            plan.rows, plan.storage[:5], (w.input for w in self.weights), allocate
        )
        self.outputs = ContextOutputs(*(allocate(*field) for field in plan.storage[5:]))
        self.attention = prepare_attention(
            plan.attention,
            device,
            tuple(w.attention for w in self.weights),
            allocate=allocate,
        )
        self.source_id, self.libraries = identity((self.input, self.attention))
        self.library, self.closed, self.bindings = self.libraries[0], False, None

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("MTP FP8 context workspace is closed")
        with self.attention.graph_lease():
            yield

    def close(self):
        self.closed = True
        self.attention.close()

    def warmup(self, weights, inputs):
        if self.plan.rows > 32:
            self.run(weights, inputs)

    def run(self, weights, inputs):
        if self.closed:
            raise RuntimeError("MTP FP8 context workspace is closed")
        self.bindings = inputs
        value = self.input.run(weights.input, inputs.input)
        self.attention.run(weights.attention, value, inputs.batch, inputs.state)
        self.input.module.mtp_fp8_endpoints(
            value.value,
            value.residual,
            inputs.batch.sequence_rows,
            inputs.endpoint_rows,
            *self.outputs,
        )
        return self.outputs


class DraftWorkspace:
    def __init__(self, plan, device, weights, allocate):
        self.plan, self.device, self.weights = plan, device, tuple(weights)
        self.input = (
            None
            if plan.first
            else InputWorkspace(
                plan.rows, plan.storage, (w.input for w in self.weights), allocate
            )
        )
        self.attention = prepare_attention(
            plan.attention,
            device,
            tuple(w.attention for w in self.weights),
            allocate=allocate,
        )
        self.ffn = prepare_ffn(
            plan.ffn, device, tuple(w.ffn for w in self.weights), allocate=allocate
        )
        self.output = prepare_output(
            plan.output,
            device,
            tuple(w.output for w in self.weights),
            allocate=allocate,
        )
        operations = (self.attention, self.ffn, self.output)
        self.source_id, self.libraries = identity(
            operations if plan.first else (self.input, *operations)
        )
        self.library, self.closed, self.bindings = self.libraries[0], False, None
        self.outputs = self.output.outputs

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("MTP FP8 draft workspace is closed")
        with ExitStack() as resources:
            for operation in (self.attention, self.ffn, self.output):
                resources.enter_context(operation.graph_lease())
            yield

    def close(self):
        self.closed = True
        for operation in (self.attention, self.ffn, self.output):
            operation.close()

    def project(self, weights, inputs):
        if self.closed:
            raise RuntimeError("MTP FP8 draft workspace is closed")
        if self.plan.first and inputs.first is None:
            raise ValueError("first MTP draft requires an explicit context endpoint")
        if not self.plan.first and inputs.input is None:
            raise ValueError(
                "later MTP drafts require shifted target hidden and token inputs"
            )
        self.bindings = inputs
        value = (
            inputs.first
            if self.plan.first
            else self.input.run(weights.input, inputs.input)
        )
        value = self.attention.run(weights.attention, value, inputs.batch, inputs.state)
        value = self.ffn.run(weights.ffn, value)
        self.output.project(weights.output, value)
        return self.outputs


class AcceptWorkspace:
    def __init__(self, plan, device, allocate):
        self.module, self.source_id, self.library = load("mtp_fp8", FUNCTIONS)
        self.plan, self.device = plan, device
        values = tuple(allocate(*field) for field in plan.storage)
        self.outputs, self.scratch = AcceptOutputs(*values[:3]), values[3:]
        self.closed, self.bindings = False, None

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("MTP FP8 acceptance workspace is closed")
        yield

    def close(self):
        self.closed = True

    def run(self, inputs):
        if self.closed:
            raise RuntimeError("MTP FP8 acceptance workspace is closed")
        self.bindings = inputs
        self.module.mtp_accept(
            inputs.target,
            inputs.proposal,
            inputs.drafts,
            inputs.rng,
            inputs.private_slots,
            *self.outputs,
            *self.scratch,
            inputs.capacity,
        )
        return self.outputs


def prepare_accept(plan, device, *, allocate=None):
    return AcceptWorkspace(plan, device, device.empty if allocate is None else allocate)


def prepare_context(plan, device, weights, *, allocate=None):
    return ContextWorkspace(
        plan, device, weights, device.empty if allocate is None else allocate
    )


def prepare_draft(plan, device, weights, *, allocate=None):
    return DraftWorkspace(
        plan, device, weights, device.empty if allocate is None else allocate
    )
