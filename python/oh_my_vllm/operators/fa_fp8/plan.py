from dataclasses import dataclass

from ..runtime.storage import storage_bytes


def projection_ranges(rows, variant, write_ranges):
    default = () if variant == "query_only" else ((0, rows),)
    ranges = default if write_ranges is None else tuple(tuple(x) for x in write_ranges)
    previous = 0
    for span in ranges:
        if len(span) != 2 or any(type(x) is not int for x in span):
            raise ValueError("write ranges require integer start/end pairs")
        start, end = span
        if not previous <= start < end <= rows:
            raise ValueError("write ranges must be ordered, disjoint, and in bounds")
        previous = end
    if variant == "query_only" and ranges:
        raise ValueError("query-only FA cannot produce K/V")
    return ranges


@dataclass(frozen=True)
class Plan:
    rows: int
    batch: int
    table_width: int
    max_queries: int
    max_context: int
    min_context: int
    variant: int
    first_token: int
    write_ranges: tuple
    storage: tuple

    @property
    def reuse_key(self):
        return (
            self.rows,
            self.batch,
            self.table_width,
            self.max_queries,
            self.max_context,
            self.min_context,
            self.variant,
            self.first_token,
            self.write_ranges,
        )

    @property
    def needs_warmup(self):
        return self.variant != 1 or any(
            end - start > 32 for start, end in self.write_ranges
        )

    @property
    def scratch_bytes(self):
        return storage_bytes(self.storage)


def make_plan(
    *,
    rows,
    batch,
    table_width,
    max_queries=None,
    max_context=None,
    min_context=0,
    variant="full",
    first_token=0,
    write_ranges=None,
):
    queries = rows if max_queries is None else max_queries
    context = queries if max_context is None else max_context
    minimum = 0 if min_context is None else min_context
    dimensions = (rows, batch, table_width, queries, context, minimum, first_token)
    if (
        any(type(x) is not int for x in dimensions)
        or variant not in {"full", "kv_only", "query_only"}
        or first_token not in (0, 1)
        or not 1 <= queries <= rows <= 32768
        or not 1 <= batch <= rows <= batch * queries
        or not 1 <= table_width <= 2048
        or not queries <= context <= min(262144, table_width * 128)
        or not 0 <= minimum <= context
    ):
        raise ValueError("invalid FP8 FA plan dimensions or variant")
    spans = projection_ranges(rows, variant, write_ranges)
    written = sum(end - start for start, end in spans)
    mode = {"full": 0, "kv_only": 1, "query_only": 2}[variant]
    active = 0 if mode == 1 else rows
    splits = (
        (16 if batch >= 3 else 32 if batch == 2 else 64)
        if rows <= 32
        else 4
        if rows <= 4096 and queries < 512 and table_width >= 64
        else 1
    )
    storage = (
        ((active, 5120), "bfloat16"),
        ((rows, 5120), "bfloat16"),
        ((rows, 5120), "float8_e4m3fn"),
        ((rows, 40), "float32"),
        ((active, 12288), "bfloat16"),
        ((written, 2048), "bfloat16"),
        ((active, 24, 256), "bfloat16"),
        ((active, 24, 256), "bfloat16"),
        ((active, 24, splits, 256), "float32"),
        ((active, 24, splits), "float32"),
        ((active, 6144), "float8_e4m3fn"),
        ((active, 48), "float32"),
        (((8 if rows <= 8 else 3), active, 5120) if rows <= 32 else (1,), "float32"),
    )
    return Plan(
        rows,
        batch,
        table_width,
        queries,
        context,
        minimum,
        mode,
        first_token,
        spans,
        storage,
    )
