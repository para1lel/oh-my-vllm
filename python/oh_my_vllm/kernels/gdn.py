from array import array
from dataclasses import dataclass
from itertools import pairwise

from .native import load


@dataclass(frozen=True)
class Weights:
    norm: object
    qkvz: object
    qkvz_scale: object
    ba: object
    convolution: object
    a_log: object
    dt_bias: object
    head_norm: object
    output: object
    output_scale: object
    embedding: object = None


@dataclass(frozen=True)
class Schedule:
    sequence_rows: tuple
    read_slots: tuple
    write_slots: tuple
    checkpoint_rows: tuple
    checkpoint_slots: tuple
    positions: tuple
    slots: int
    verify: bool

    def validate(self):
        starts = self.sequence_rows
        batch = len(self.read_slots)
        if (
            self.slots < 1
            or self.slots > 65536
            or batch < 1
            or len(starts) != batch + 1
            or not starts
            or starts[0] != 0
            or any(a > b for a, b in pairwise(starts))
            or starts[-1] != len(self.write_slots)
            or starts[-1] > 32768
            or len(self.positions) != batch
            or any(position < 0 for position in self.positions)
            or len(self.checkpoint_rows) != len(self.checkpoint_slots)
            or any(slot < 0 or slot >= self.slots for slot in self.read_slots)
            or any(slot < 0 or slot >= self.slots for slot in self.checkpoint_slots)
        ):
            raise ValueError("invalid GDN sequence or read slot schedule")
        writes = list(self.write_slots)
        expected = set()
        for sequence, (first, last) in enumerate(pairwise(starts)):
            count = last - first
            if self.verify and count > 5:
                raise ValueError("GDN verification has at most five rows per sequence")
            if self.positions[sequence] + count > 262144:
                raise ValueError("GDN sequence exceeds the context limit")
            if not self.verify and count:
                for row in range(first, last):
                    if (self.positions[sequence] + row - first + 1) % 784 == 0:
                        expected.add(row)
                expected.add(last - 1)
        if self.verify:
            if self.checkpoint_rows or any(slot < 0 for slot in writes):
                raise ValueError("GDN verification must save each private row")
        else:
            if set(self.checkpoint_rows) != expected or len(expected) != len(
                self.checkpoint_rows
            ):
                raise ValueError(
                    "GDN prefill must save aligned checkpoints and final rows"
                )
            if any(slot != -1 for slot in writes):
                raise ValueError("GDN prefill uses checkpoint slots only")
            for row, slot in zip(
                self.checkpoint_rows, self.checkpoint_slots, strict=True
            ):
                writes[row] = slot
        private = [slot for slot in writes if slot >= 0]
        if (
            any(slot < -1 or slot >= self.slots for slot in writes)
            or len(private) != len(set(private))
            or set(private).intersection(self.read_slots)
        ):
            raise ValueError("GDN writes require distinct private state slots")
        return tuple(writes)


class GDN:
    def __init__(self, weights, schedule, allocate, upload):
        writes = schedule.validate()
        self.module, self.source_id, self.library = load("gdn", ("gdn",))
        self.weights, self.schedule = weights, schedule
        rows = schedule.sequence_rows[-1]
        split_values = (not schedule.verify and rows > 32) or (
            schedule.verify and len(schedule.read_slots) == 1 and rows == 5
        )
        self.metadata = tuple(
            upload(array("i", values))
            for values in (schedule.sequence_rows, schedule.read_slots, writes)
        )
        self.buffers = tuple(
            allocate(shape, dtype)
            for shape, dtype in (
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
                ((rows if split_values else 0, 6144), "bfloat16"),
                ((8, rows, 5120) if rows <= 8 else (1,), "float32"),
                ((rows if not schedule.verify and rows > 32 else 0, 96), "float32"),
            )
        )
        self.empty_embedding = allocate((1, 5120), "bfloat16")
        self.empty_tokens = allocate((rows,), "int64")
        self.empty_media = allocate((0, 5120), "bfloat16")
        self.empty_media_rows = allocate((0,), "int32")

    def __call__(
        self,
        value,
        residual,
        state,
        convolution_state,
        token_ids=None,
        media=None,
        media_rows=None,
    ):
        weight = self.weights
        embedding = weight.embedding is not None
        if embedding and token_ids is None:
            raise ValueError("GDN embedding input requires token IDs")
        if (media is None) != (media_rows is None):
            raise ValueError("GDN media input requires media rows")
        self.module.gdn(
            value,
            residual,
            weight.norm,
            weight.qkvz,
            weight.qkvz_scale,
            weight.ba,
            weight.convolution,
            weight.a_log,
            weight.dt_bias,
            weight.head_norm,
            weight.output,
            weight.output_scale,
            state,
            convolution_state,
            *self.metadata,
            weight.embedding if embedding else self.empty_embedding,
            token_ids if embedding else self.empty_tokens,
            media if media is not None else self.empty_media,
            media_rows if media_rows is not None else self.empty_media_rows,
            embedding,
            self.schedule.verify,
            *self.buffers,
        )
        return self.buffers[:2]
