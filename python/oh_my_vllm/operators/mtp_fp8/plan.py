from dataclasses import dataclass

from ..fa_fp8.plan import make_plan as attention_plan
from ..ffn.plan import make_plan as ffn_plan
from ..output.plan import make_plan as output_plan
from ..runtime.storage import storage_bytes


def input_storage(rows):
    return (
        *(((rows, width), "bfloat16") for width in (10240, 5120, 5120)),
        ((4 if rows <= 32 else 0, rows, 5120), "float32"),
        ((0, 5120), "bfloat16"),
    )


def table_columns(table_width, max_context):
    if not 1 <= max_context <= 262144:
        raise ValueError("invalid MTP maximum context")
    width = (max_context + 127) // 128 if table_width is None else table_width
    if not 1 <= width <= 2048 or max_context > width * 128:
        raise ValueError("MTP table does not cover the maximum context")
    return width


@dataclass(frozen=True)
class ContextPlan:
    rows: int
    batch: int
    attention: object
    storage: tuple

    @property
    def reuse_key(self):
        return (self.rows, self.batch, self.attention.reuse_key)

    @property
    def mtp_scratch_bytes(self):
        return storage_bytes(self.storage)

    @property
    def metadata_storage(self):
        rows, batch, width = self.rows, self.batch, self.attention.table_width
        return (
            ((rows,), "int64"),
            ((rows,), "int32"),
            ((rows,), "int32"),
            ((3, rows), "int64"),
            ((rows,), "int32"),
            ((batch, width), "int32"),
            ((batch + 1,), "int32"),
            ((rows,), "int32"),
            ((batch,), "int32"),
        )


def context_plan(
    *,
    rows,
    batch=1,
    table_width=None,
    max_context=262144,
    max_queries=None,
    write_ranges=None,
):
    if not 1 <= batch <= rows <= 32768:
        raise ValueError("invalid MTP context batch")
    attention = attention_plan(
        rows=rows,
        batch=batch,
        table_width=table_columns(table_width, max_context),
        max_context=max_context,
        variant="kv_only",
        max_queries=max_queries,
        first_token=1,
        write_ranges=write_ranges,
    )
    storage = input_storage(rows) + (((batch, 5120), "bfloat16"),) * 2
    return ContextPlan(rows, batch, attention, storage)


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

    @property
    def mtp_scratch_bytes(self):
        return storage_bytes(self.storage)


@dataclass(frozen=True)
class AcceptPlan:
    batch: int
    storage: tuple

    @property
    def reuse_key(self):
        return (self.batch,)

    @property
    def mtp_scratch_bytes(self):
        return storage_bytes(self.storage)


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


def draft_plan(*, rows, first=False, table_width=None, max_context=262144):
    if not 1 <= rows <= 4:
        raise ValueError("MTP drafts support one to four requests")
    attention = attention_plan(
        rows=rows,
        batch=rows,
        table_width=table_columns(table_width, max_context),
        max_context=max_context,
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
        () if first else input_storage(rows),
    )
