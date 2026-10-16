import hashlib
from dataclasses import dataclass

from .fa import FA
from .native import load

FUNCTIONS = ("mtp_input", "mtp_endpoints")


def identity(operations):
    source_id = hashlib.sha256(
        "".join(op.source_id for op in operations).encode()
    ).hexdigest()
    libraries = tuple(
        dict.fromkeys(
            path
            for op in operations
            for path in getattr(op, "libraries", (op.library,))
        )
    )
    return source_id, libraries


@dataclass(frozen=True)
class InputWeights:
    embedding: object
    hidden_norm: object
    embedding_norm: object
    projection: object


class MTPInput:
    def __init__(self, weights, rows, allocate):
        self.module, self.source_id, self.library = load("mtp", FUNCTIONS)
        self.weights = weights
        self.buffers = tuple(
            allocate((rows, width), "bfloat16") for width in (10240, 5120, 5120)
        )
        self.empty_media = allocate((0, 5120), "bfloat16")

    def __call__(
        self,
        hidden,
        tokens,
        hidden_rows,
        boundary_hidden,
        boundary_rows,
        media=None,
        media_indices=None,
    ):
        w = self.weights
        self.module.mtp_input(
            hidden,
            tokens,
            w.embedding,
            w.hidden_norm,
            w.embedding_norm,
            w.projection,
            hidden_rows,
            boundary_hidden,
            boundary_rows,
            self.empty_media if media is None else media,
            hidden_rows if media_indices is None else media_indices,
            media is not None,
            *self.buffers,
        )
        return self.buffers[1:3]


class MTPContext:
    def __init__(
        self,
        input_weights,
        attention_weights,
        rows,
        allocate,
        *,
        max_queries=None,
        batch=1,
    ):
        self.input = MTPInput(input_weights, rows, allocate)
        self.endpoints = tuple(allocate((batch, 5120), "bfloat16") for _ in range(2))
        self.attention = FA(
            attention_weights,
            rows,
            allocate,
            variant="kv_only",
            max_queries=max_queries,
            first_token=1,
        )
        self.source_id, self.libraries = identity((self.input, self.attention))
        self.library = self.libraries[0]

    def __call__(self, input_arguments, attention_arguments):
        value, zero = self.input(*input_arguments)
        self.attention(value, zero, *attention_arguments)
        self.input.module.mtp_endpoints(
            value, zero, attention_arguments[4], *self.endpoints
        )
        return self.endpoints


class MTPDraft:
    def __init__(self, input_operation, attention, ffn, sample, *, variant="full"):
        if variant not in {"full", "query_only"}:
            raise ValueError("invalid MTP draft variant")
        self.input, self.attention, self.ffn, self.sample = (
            input_operation,
            attention,
            ffn,
            sample,
        )
        self.variant = variant
        operations = (input_operation, attention, ffn, sample)
        self.source_id, self.libraries = identity(operations)
        self.library = self.libraries[0]

    def __call__(self, input_arguments, attention_arguments, sampling_arguments):
        value, residual = (
            input_arguments
            if self.variant == "query_only"
            else self.input(*input_arguments)
        )
        value, residual = self.attention(value, residual, *attention_arguments)
        value, residual = self.ffn(value, residual)
        return self.sample(value, residual, *sampling_arguments)
