from dataclasses import dataclass

from .native import load


@dataclass(frozen=True)
class Weights:
    norm: object
    projection: object
    projection_scale: object
    query_norm: object
    key_norm: object
    output: object
    output_scale: object


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
    ):
        self.module, self.source_id, self.library = load("fa", ("fa",))
        self.weights = weights
        self.variant = {"full": 0, "kv_only": 1, "query_only": 2}[variant]
        self.max_queries = max_queries or rows
        self.first_token = first_token
        columns = (14336, 2048, 12288)[self.variant]
        splits = 64 if rows <= 32 else 1
        self.buffers = tuple(
            allocate(shape, dtype)
            for shape, dtype in (
                ((rows, 5120), "bfloat16"),
                ((rows, 5120), "bfloat16"),
                ((rows, 5120), "float8_e4m3fn"),
                ((rows, 40), "float32"),
                ((rows, columns), "bfloat16"),
                ((rows, 24, 256), "bfloat16"),
                ((rows, 24, 256), "bfloat16"),
                ((rows, 24, splits, 256), "float32"),
                ((rows, 24, splits), "float32"),
                ((rows, 6144), "float8_e4m3fn"),
                ((rows, 48), "float32"),
            )
        )

    def __call__(
        self, value, residual, positions, cache, slots, tables, sequence_rows, lengths
    ):
        w = self.weights
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
        )
        return (None, self.buffers[1]) if self.variant == 1 else self.buffers[:2]
