from contextlib import contextmanager
from dataclasses import dataclass, replace

from .native import load

FUNCTIONS = (
    "fa",
    "fa_mx_prepare_weights",
    "fa_mx_prepare",
    "fa_mx_graph_lease",
    "fa_mx_close",
    "fa_mx_images",
)
MX_WORKSPACE_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class Weights:
    norm: object
    projection: object
    projection_scale: object
    query_norm: object
    key_norm: object
    output: object
    output_scale: object
    mx: object = None


def prepare_weights(weights, allocate):
    if weights.mx is not None:
        return weights
    module, _, _ = load("fa", FUNCTIONS)
    prepared = tuple(
        allocate(shape, dtype)
        for shape, dtype in (
            ((14336, 5120), "float8_e4m3fn"),
            ((14336 * 160,), "uint8"),
            ((5120, 6144), "float8_e4m3fn"),
            ((5120 * 192,), "uint8"),
        )
    )
    owner = module.fa_mx_prepare_weights(
        weights.norm,
        weights.projection,
        weights.projection_scale,
        *prepared[:2],
        weights.output,
        weights.output_scale,
        *prepared[2:],
    )
    return replace(weights, mx=owner)


class FA:
    def __init__(
        self,
        weights,
        rows,
        allocate,
        *,
        variant="full",
        max_queries=None,
        first_token=0,
        max_context=None,
        min_context=None,
        prepare=True,
    ):
        self.module, self.source_id, self.library = load("fa", FUNCTIONS)
        self.mx = None
        self.closed = False
        self.prepared = prepare
        self.weights = weights
        self.variant = {"full": 0, "kv_only": 1, "query_only": 2}[variant]
        self.max_queries = max_queries or rows
        self.first_token = first_token
        self.max_context = max_context or self.max_queries
        self.min_context = min_context or 0
        columns = (14336, 2048, 12288)[self.variant]
        if (
            rows <= 32
            and self.variant != 1
            and self.max_queries == 1
            and (
                (rows >= 4 and self.max_context >= 32768)
                or (rows == 2 and self.first_token == 0 and self.min_context >= 32768)
            )
        ):
            splits = 128
        else:
            splits = (
                64 if rows <= 32 else 4 if rows <= 4096 and self.variant != 1 else 1
            )
        attention_rows = 0 if self.variant == 1 and rows > 32 else rows
        self.buffers = tuple(
            allocate(shape, dtype)
            for shape, dtype in (
                ((attention_rows, 5120), "bfloat16"),
                ((rows, 5120), "bfloat16"),
                ((rows, 5120), "float8_e4m3fn"),
                ((rows, 40), "float32"),
                ((rows, columns), "bfloat16"),
                ((attention_rows, 24, 256), "bfloat16"),
                ((attention_rows, 24, 256), "bfloat16"),
                ((attention_rows, 24, splits, 256), "float32"),
                ((attention_rows, 24, splits), "float32"),
                ((attention_rows, 6144), "float8_e4m3fn"),
                ((attention_rows, 48), "float32"),
                ((8, rows, 5120) if rows <= 8 else (1,), "float32"),
            )
        )

        self.use_mx = self.variant == 0 and rows > 128
        if self.use_mx:
            padded = (rows + 127) // 128 * 128
            self.mx_buffers = tuple(
                allocate(shape, dtype)
                for shape, dtype in (
                    ((MX_WORKSPACE_BYTES,), "uint8"),
                    ((padded, 5120), "float8_e4m3fn"),
                    ((padded * 160,), "uint8"),
                    ((padded, 14336), "bfloat16"),
                    ((padded, 6144), "float8_e4m3fn"),
                    ((padded * 192,), "uint8"),
                    ((padded, 5120), "bfloat16"),
                )
            )
            if prepare:
                if weights.mx is None:
                    raise ValueError("MXFP8 FA weights must be prepared before use")
                self.mx = self.module.fa_mx_prepare(weights.mx, *self.mx_buffers, rows)

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("FA is closed")
        if not self.prepared:
            raise RuntimeError("FA sizing instance cannot execute")
        lease = self.module.fa_mx_graph_lease(self.mx) if self.mx is not None else None
        try:
            yield
        finally:
            del lease

    def close(self):
        if not self.closed:
            if self.mx is not None:
                self.module.fa_mx_close(self.mx)
                self.mx = None
            self.closed = True

    def __call__(
        self, value, residual, positions, cache, slots, tables, sequence_rows, lengths
    ):
        if self.closed:
            raise RuntimeError("FA is closed")
        if not self.prepared:
            raise RuntimeError("FA sizing instance cannot execute")
        w = self.weights
        if self.use_mx and (self.mx is None or w.mx is None):
            raise RuntimeError("MXFP8 FA is not prepared")
        self.module.fa(
            value,
            residual,
            w.norm,
            w.projection,
            w.projection_scale,
            w.query_norm,
            w.key_norm,
            positions,
            cache,
            slots,
            tables,
            sequence_rows,
            lengths,
            w.output,
            w.output_scale,
            *self.buffers,
            self.variant,
            self.max_queries,
            self.first_token,
            self.max_context,
            self.min_context,
            self.mx,
            w.mx if self.use_mx else None,
        )
        return (None, self.buffers[1]) if self.variant == 1 else self.buffers[:2]
