import hashlib
from contextlib import contextmanager
from dataclasses import dataclass

from .fa import FA
from .native import load

FUNCTIONS = ("mtp_input", "mtp_endpoints", "mtp_context")


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
        self.partials = allocate((4 if rows <= 32 else 0, rows, 5120), "float32")
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
            self.partials,
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
        self._context_module = self.input.module
        self._context_source = self.input.source_id
        self._attention_module = self.attention.module
        self._attention_source = self.attention.source_id
        self._input_buffers, self._fa_buffers = (
            self.input.buffers,
            self.attention.buffers,
        )
        self._input_weights, self._attention_weights = input_weights, attention_weights
        self._context_function = self.input.module.mtp_context
        self.kv_partials = None
        owner = getattr(allocate, "__self__", None)
        view = getattr(owner, "view", None)
        if (
            (rows, batch) in {(4, 1), (16, 4)}
            and self.attention.max_queries == 4
            and type(self.input) is MTPInput
            and type(self.attention) is FA
            and self._context_format(rows)
            and callable(view)
        ):
            self.kv_partials = view(self.attention.buffers[7], (8, rows, 5120))

    def _context_format(self, rows):
        layout = (
            ((rows, 5120), "bfloat16"),
            ((rows, 5120), "bfloat16"),
            ((rows, 5120), "float8_e4m3fn"),
            ((rows, 40), "float32"),
            ((rows, 2048), "bfloat16"),
            ((rows, 24, 256), "bfloat16"),
            ((rows, 24, 256), "bfloat16"),
            ((rows, 24, 64, 256), "float32"),
            ((rows, 24, 64), "float32"),
            ((rows, 6144), "float8_e4m3fn"),
            ((rows, 48), "float32"),
            ((8, rows, 5120) if rows <= 8 else (1,), "float32"),
        )
        try:
            return len(self.attention.buffers) == len(layout) and all(
                tuple(tensor.shape) == shape and str(tensor.dtype) == dtype
                for tensor, (shape, dtype) in zip(
                    self.attention.buffers, layout, strict=True
                )
            )
        except (AttributeError, TypeError):
            return False

    def _context_eligible(self, input_arguments, attention_arguments):
        if (
            self.kv_partials is None
            or type(self.input) is not MTPInput
            or type(self.attention) is not FA
            or self.input.module is not self._context_module
            or self.input.source_id != self._context_source
            or self.attention.module is not self._attention_module
            or self.attention.source_id != self._attention_source
            or self.input.buffers is not self._input_buffers
            or self.attention.buffers is not self._fa_buffers
            or self.input.weights is not self._input_weights
            or self.attention.weights is not self._attention_weights
            or self.attention.variant != 1
            or self.attention.first_token != 1
            or self.attention.max_queries != 4
            or not isinstance(input_arguments, (tuple, list))
            or len(input_arguments) not in (5, 6, 7)
            or not isinstance(attention_arguments, (tuple, list))
            or len(attention_arguments) != 6
        ):
            return False
        if len(input_arguments) > 5 and input_arguments[5] is not None:
            return False
        if len(input_arguments) > 6 and input_arguments[6] is not None:
            return False
        try:
            tokens, sequence_rows = input_arguments[1], attention_arguments[4]
            if len(tokens.shape) != 1 or len(sequence_rows.shape) != 1:
                return False
            rows, batch = tokens.shape[0], sequence_rows.shape[0] - 1
            return (
                (rows, batch) in {(4, 1), (16, 4)}
                and rows == self.input.buffers[1].shape[0]
                and rows == self.attention.buffers[1].shape[0]
                and batch == self.endpoints[0].shape[0]
            )
        except (AttributeError, IndexError, TypeError):
            return False

    @contextmanager
    def graph_lease(self):
        owners = (
            self.input.module,
            self.attention.module,
            self.input.weights,
            self.attention.weights,
            self.input.buffers,
            self.input.partials,
            self.attention.buffers,
            self.kv_partials,
            self.endpoints,
        )
        try:
            yield
        finally:
            del owners

    def __call__(self, input_arguments, attention_arguments):
        if self._context_eligible(input_arguments, attention_arguments):
            hidden, tokens, hidden_rows, boundary, boundary_rows = input_arguments[:5]
            positions, cache, slots, tables, sequences, lengths = attention_arguments
            weight, fa = self.input.weights, self.attention.weights
            packed, value, zero = self.input.buffers
            _, summed, quantized, scales, kv_packed = self.attention.buffers[:5]
            if self._context_function(
                hidden,
                tokens,
                weight.embedding,
                weight.hidden_norm,
                weight.embedding_norm,
                weight.projection,
                hidden_rows,
                boundary,
                boundary_rows,
                self.input.empty_media,
                hidden_rows,
                False,
                fa.norm,
                fa.projection,
                fa.projection_scale,
                fa.key_norm,
                positions,
                cache,
                slots,
                tables,
                sequences,
                lengths,
                packed,
                value,
                zero,
                self.input.partials,
                summed,
                quantized,
                scales,
                kv_packed,
                self.kv_partials,
                self.endpoints[0],
                self.endpoints[1],
            ):
                return self.endpoints
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
