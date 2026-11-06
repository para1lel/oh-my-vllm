from dataclasses import dataclass
from itertools import pairwise

from ..runtime.storage import storage_bytes
from .metadata import make_layout


def chunked_prefill(rows, batch, max_queries, verify):
    return not verify and max_queries > 128


def chunk_workspace(chunks, batch):
    return (
        ((chunks, 4), "int32"),
        ((batch + 1 if chunks else 0,), "int32"),
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
    chunks: int
    slots: int
    exports: int
    export_slot_upper_bound: int
    storage: tuple

    @property
    def reuse_key(self):
        return (
            self.rows,
            self.batch,
            self.max_queries,
            self.verify,
            self.chunks,
            self.slots,
        )

    @property
    def needs_warmup(self):
        return self.rows > 32

    @property
    def scratch_bytes(self):
        return storage_bytes(self.storage)

    @property
    def chunk_scratch_bytes(self):
        return storage_bytes(self.storage[14:20])


def make_plan(schedule):
    layout = make_layout(schedule)
    rows, batch = schedule.sequence_rows[-1], len(schedule.read_slots)
    maximum = max(b - a for a, b in pairwise(schedule.sequence_rows))
    chunks = (
        layout.chunks if chunked_prefill(rows, batch, maximum, schedule.verify) else 0
    )
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
        *chunk_workspace(chunks, batch),
        ((1, 5120), "bfloat16"),
        ((rows,), "int64"),
        ((0, 5120), "bfloat16"),
        ((0,), "int32"),
        ((schedule.slots,), "int32"),
    )
    export_slots = tuple(slot for slot in schedule.validate() if slot >= 0)
    return Plan(
        rows,
        batch,
        maximum,
        schedule.verify,
        chunks,
        schedule.slots,
        len(export_slots),
        max(export_slots, default=-1) + 1,
        storage,
    )
