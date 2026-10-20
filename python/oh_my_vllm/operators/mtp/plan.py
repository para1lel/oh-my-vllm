from dataclasses import dataclass

from ..fa.plan import make_plan as attention_plan
from ..ffn.plan import make_plan as ffn_plan
from ..output.plan import make_plan as output_plan


def input_storage(rows):
    return (
        *(((rows, width), "bfloat16") for width in (10240, 5120, 5120)),
        ((4 if rows <= 32 else 0, rows, 5120), "float32"),
        ((0, 5120), "bfloat16"),
    )


@dataclass(frozen=True)
class ContextPlan:
    rows: int
    batch: int
    attention: object
    fused: bool
    storage: tuple

    @property
    def reuse_key(self):
        return (self.rows, self.batch, self.attention.reuse_key, self.fused)


def context_plan(*, rows, batch=1, table_width=335, max_queries=None):
    if not 1 <= batch <= rows:
        raise ValueError("invalid MTP context batch")
    attention = attention_plan(
        rows=rows,
        batch=batch,
        table_width=table_width,
        variant="kv_only",
        max_queries=max_queries,
        first_token=1,
    )
    fused = (rows, batch) in {(4, 1), (16, 4)} and attention.max_queries == 4
    storage = input_storage(rows) + (((batch, 5120), "bfloat16"),) * 2
    return ContextPlan(rows, batch, attention, fused, storage)


@dataclass(frozen=True)
class DraftPlan:
    rows: int
    first: bool
    attention: object
    ffn: object
    output: object
    storage: tuple

    @property
    def reuse_key(self):
        return (
            self.rows,
            self.first,
            self.attention.reuse_key,
            self.ffn.reuse_key,
            self.output.reuse_key,
        )


def draft_plan(*, rows, first=False, table_width=335):
    if not 1 <= rows <= 4:
        raise ValueError("MTP drafts support one to four requests")
    attention = attention_plan(
        rows=rows,
        batch=rows,
        table_width=table_width,
        max_queries=1,
        first_token=1,
        variant="query_only" if first else "full",
    )
    return DraftPlan(
        rows,
        first,
        attention,
        ffn_plan(rows=rows),
        output_plan(rows=rows),
        input_storage(rows),
    )


@dataclass(frozen=True)
class AcceptPlan:
    batch: int
    storage: tuple

    @property
    def reuse_key(self):
        return (self.batch,)


def accept_plan(*, batch):
    if not 1 <= batch <= 4:
        raise ValueError("MTP acceptance supports one to four requests")
    storage = (
        ((batch, 7), "int64"),
        ((batch,), "int32"),
        ((batch,), "int32"),
        ((batch,), "float64"),
        ((batch, (248320 + 255) // 256), "float32"),
    )
    return AcceptPlan(batch, storage)
