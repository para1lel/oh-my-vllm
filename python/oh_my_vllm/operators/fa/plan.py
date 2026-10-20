from dataclasses import dataclass


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
        )

    @property
    def needs_warmup(self):
        return self.rows > 32


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
):
    queries = rows if max_queries is None else max_queries
    context = queries if max_context is None else max_context
    minimum = 0 if min_context is None else min_context
    if (
        variant not in {"full", "kv_only", "query_only"}
        or first_token not in (0, 1)
        or not 1 <= queries <= rows <= 32768
        or not 1 <= batch <= rows <= batch * queries
        or not 1 <= table_width <= 335
        or context > table_width * 784
        or not queries <= context <= 262144
        or not 0 <= minimum <= context
    ):
        raise ValueError("invalid FA plan dimensions or variant")
    variant = {"full": 0, "kv_only": 1, "query_only": 2}[variant]
    long_batch = rows <= 32 and variant != 1 and queries == 1
    context = max(queries, 32768 if long_batch and context >= 32768 else queries)
    minimum = (
        32768
        if long_batch and rows == 2 and first_token == 0 and minimum >= 32768
        else 0
    )
    if long_batch and ((rows >= 4 and context >= 32768) or minimum >= 32768):
        splits = 128
    else:
        splits = (
            128
            if rows <= 32 and variant != 1 and 1 < queries <= 5
            else 64
            if rows <= 32
            else 4
            if rows <= 4096 and variant != 1
            else 1
        )
    active = 0 if variant == 1 and rows > 32 else rows
    storage = (
        ((active, 5120), "bfloat16"),
        ((rows, 5120), "bfloat16"),
        ((rows, 5120), "float8_e4m3fn"),
        ((rows, 40), "float32"),
        ((rows, (14336, 2048, 12288)[variant]), "bfloat16"),
        ((active, 24, 256), "bfloat16"),
        ((active, 24, 256), "bfloat16"),
        ((active, 24, splits, 256), "float32"),
        ((active, 24, splits), "float32"),
        ((active, 6144), "float8_e4m3fn"),
        ((active, 48), "float32"),
        ((8 if rows <= 8 else 3, rows, 5120) if rows <= 32 else (1,), "float32"),
    )
    return Plan(
        rows,
        batch,
        table_width,
        queries,
        context,
        minimum,
        variant,
        first_token,
        storage,
    )
