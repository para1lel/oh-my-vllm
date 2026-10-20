from dataclasses import dataclass
from itertools import pairwise


def chunked_prefill(rows, batch, max_queries, verify):
    return not verify and max_queries > 128


def chunk_workspace(rows, batch, max_queries, verify):
    chunks = 0
    if chunked_prefill(rows, batch, max_queries, verify):
        capacity = (max_queries + 63) // 64 + (max_queries + 783) // 784 + 1
        chunks = batch * capacity
    return (
        ((chunks, 4), "int32"),
        ((chunks, 48, 1, 2, 64, 128), "bfloat16"),
        ((chunks, 48, 128), "float32"),
        ((chunks, 48, 128, 128), "bfloat16"),
        ((chunks, 48, 64, 64), "bfloat16"),
    )


@dataclass(frozen=True)
class Plan:
    rows: int
    batch: int
    max_queries: int
    verify: bool
    storage: tuple

    @property
    def reuse_key(self):
        return (self.rows, self.batch, self.max_queries, self.verify)

    @property
    def needs_warmup(self):
        return self.rows > 32


def make_plan(schedule):
    schedule.validate()
    rows, batch = schedule.sequence_rows[-1], len(schedule.read_slots)
    if rows < 1:
        raise ValueError("GDN requires at least one row")
    maximum = max(b - a for a, b in pairwise(schedule.sequence_rows))
    split = rows > 32 or schedule.verify
    storage = (
        ((rows, 5120), "bfloat16"),
        ((rows, 5120), "bfloat16"),
        ((rows, 5120), "bfloat16"),
        ((rows, 5120), "bfloat16"),
        ((rows, 5120), "float8_e4m3fn"),
        ((rows, 40), "float32"),
        ((rows, 16384), "bfloat16"),
        ((rows, 96), "bfloat16"),
        ((rows, 10240), "bfloat16"),
        ((rows, 6144), "float8_e4m3fn"),
        ((rows, 48), "float32"),
        ((rows if split else 0, 6144), "bfloat16"),
        ((8 if rows <= 8 else 3, rows, 5120) if rows <= 32 else (1,), "float32"),
        ((rows if not schedule.verify and rows > 32 else 0, 96), "float32"),
        *chunk_workspace(rows, batch, maximum, schedule.verify),
        ((1, 5120), "bfloat16"),
        ((rows,), "int64"),
        ((0, 5120), "bfloat16"),
        ((0,), "int32"),
    )
    return Plan(rows, batch, maximum, schedule.verify, storage)
