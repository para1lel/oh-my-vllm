import hashlib
from contextlib import ExitStack, contextmanager

from ..fa.components import context_projection
from ..fa.workspace import prepare as prepare_attention
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
            plan.storage[:5], (w.input for w in weights), allocate
        )
        self.outputs = ContextOutputs(*(allocate(*field) for field in plan.storage[5:]))
        self.attention = prepare_attention(
            plan.attention,
            device,
            tuple(w.attention for w in weights),
            allocate=allocate,
        )
        self.projection = (
            context_projection(self.attention, device) if plan.fused else None
        )
        self.source_id, self.libraries = identity((self.input, self.attention))
        self.library = self.libraries[0]
        self.closed = False

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("MTP context workspace is closed")
        owners = (self.weights, self.input, self.outputs, self.projection)
        with self.attention.graph_lease():
            try:
                yield
            finally:
                del owners

    def close(self):
        self.closed = True
        self.attention.close()

    def run(self, weights, inputs):
        if self.closed:
            raise RuntimeError("MTP context workspace is closed")
        i, b, c = inputs.input, inputs.batch, inputs.state
        self.attention.bind_layout(c)
        if self.projection is not None and i.media is None and i.media_rows is None:
            w, fa, s, p = (
                weights.input,
                weights.attention,
                self.input.storage,
                self.projection,
            )
            if self.input.module.mtp_context(
                i.hidden,
                i.tokens,
                w.embedding,
                w.hidden_norm,
                w.embedding_norm,
                w.projection,
                i.hidden_rows,
                i.boundary_hidden,
                i.boundary_rows,
                s.empty_media,
                i.hidden_rows,
                False,
                fa.norm,
                fa.projection,
                fa.projection_scale,
                fa.key_norm,
                b.positions,
                c.cache[0],
                c.slots,
                c.tables,
                b.sequence_rows,
                b.lengths,
                s.packed,
                s.value,
                s.zero,
                s.partials,
                p.residual,
                p.quantized,
                p.scales,
                p.projection,
                p.partials,
                self.outputs.value,
                self.outputs.residual,
            ):
                return self.outputs
        value = self.input.run(weights.input, i)
        self.attention.run(weights.attention, value, b, c)
        self.input.module.mtp_endpoints(
            value.value, value.residual, b.sequence_rows, *self.outputs
        )
        return self.outputs


class DraftWorkspace:
    def __init__(self, plan, device, weights, allocate):
        self.plan, self.device, self.weights = plan, device, tuple(weights)
        self.input = InputWorkspace(plan.storage, (w.input for w in weights), allocate)
        self.attention = prepare_attention(
            plan.attention,
            device,
            tuple(w.attention for w in weights),
            allocate=allocate,
        )
        self.ffn = prepare_ffn(
            plan.ffn, device, tuple(w.ffn for w in weights), allocate=allocate
        )
        self.output = prepare_output(
            plan.output, device, tuple(w.output for w in weights), allocate=allocate
        )
        self.outputs = self.output.outputs
        self.source_id, self.libraries = identity(
            (self.input, self.attention, self.ffn, self.output)
        )
        self.library = self.libraries[0]
        self.closed = False

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("MTP draft workspace is closed")
        owners = (self.weights, self.input)
        with ExitStack() as resources:
            for operation in (self.attention, self.ffn, self.output):
                resources.enter_context(operation.graph_lease())
            try:
                yield
            finally:
                del owners

    def close(self):
        self.closed = True
        for operation in (self.attention, self.ffn, self.output):
            operation.close()

    def project(self, weights, inputs):
        if self.closed:
            raise RuntimeError("MTP draft workspace is closed")
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
        self.module, self.source_id, self.library = load("mtp", FUNCTIONS)
        self.plan, self.device = plan, device
        values = tuple(allocate(*field) for field in plan.storage)
        self.outputs, self.scratch = AcceptOutputs(*values[:3]), values[3:]
        self.closed = False

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("MTP acceptance workspace is closed")
        owners = (self.outputs, self.scratch, self.module)
        try:
            yield
        finally:
            del owners

    def close(self):
        self.closed = True

    def run(self, inputs):
        if self.closed:
            raise RuntimeError("MTP acceptance workspace is closed")
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


def prepare_context(plan, device, weights, *, allocate=None):
    return ContextWorkspace(
        plan, device, weights, device.empty if allocate is None else allocate
    )


def prepare_draft(plan, device, weights, *, allocate=None):
    return DraftWorkspace(
        plan, device, weights, device.empty if allocate is None else allocate
    )


def prepare_accept(plan, device, *, allocate=None):
    return AcceptWorkspace(plan, device, device.empty if allocate is None else allocate)
