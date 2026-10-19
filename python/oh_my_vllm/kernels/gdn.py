from array import array
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import pairwise

from .native import load

FUNCTIONS = (
    "gdn",
    "gdn_mx_prepare_weights",
    "gdn_mx_prepare",
    "gdn_mx_graph_lease",
    "gdn_mx_close",
    "gdn_mx_images",
)
MX_WORKSPACE_BYTES = 128 * 1024 * 1024


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
    mx: object = None


def prepare_weights(weights, allocate):
    if weights.mx is not None:
        return weights
    module, _, _ = load("gdn", FUNCTIONS)
    prepared = tuple(
        allocate(shape, dtype)
        for shape, dtype in (
            ((16384, 5120), "float8_e4m3fn"),
            ((16384 * 160,), "uint8"),
            ((5120, 6144), "float8_e4m3fn"),
            ((5120 * 192,), "uint8"),
        )
    )
    owner = module.gdn_mx_prepare_weights(
        weights.norm,
        weights.qkvz,
        weights.qkvz_scale,
        *prepared[:2],
        weights.output,
        weights.output_scale,
        *prepared[2:],
    )
    return replace(weights, mx=owner)


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
    def __init__(self, weights, schedule, allocate, upload, *, prepare=True):
        writes = schedule.validate()
        self.module, self.source_id, self.library = load("gdn", FUNCTIONS)
        self.weights, self.schedule = weights, schedule
        self.mx = None
        self.closed = False
        self.prepared = prepare
        rows = schedule.sequence_rows[-1]
        self.max_queries = max(b - a for a, b in pairwise(schedule.sequence_rows))
        split_values = (not schedule.verify and rows > 32) or (
            schedule.verify and self.max_queries > 1
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
                *chunk_workspace(
                    rows, len(schedule.read_slots), self.max_queries, schedule.verify
                ),
            )
        )
        self.empty_embedding = allocate((1, 5120), "bfloat16")
        self.empty_tokens = allocate((rows,), "int64")
        self.empty_media = allocate((0, 5120), "bfloat16")
        self.empty_media_rows = allocate((0,), "int32")
        self.use_mx = rows > 128
        if self.use_mx:
            padded = (rows + 127) // 128 * 128
            self.mx_buffers = tuple(
                allocate(shape, dtype)
                for shape, dtype in (
                    ((MX_WORKSPACE_BYTES,), "uint8"),
                    ((padded, 5120), "float8_e4m3fn"),
                    ((padded * 160,), "uint8"),
                    ((padded, 16384), "bfloat16"),
                    ((padded, 6144), "float8_e4m3fn"),
                    ((padded * 192,), "uint8"),
                    ((padded, 5120), "bfloat16"),
                )
            )
            if prepare:
                if weights.mx is None:
                    raise ValueError("MXFP8 GDN weights must be prepared before use")
                self.mx = self.module.gdn_mx_prepare(weights.mx, *self.mx_buffers, rows)

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("GDN is closed")
        if not self.prepared:
            raise RuntimeError("GDN sizing instance cannot execute")
        lease = self.module.gdn_mx_graph_lease(self.mx) if self.mx is not None else None
        try:
            yield
        finally:
            del lease

    def close(self):
        if not self.closed:
            if self.mx is not None:
                self.module.gdn_mx_close(self.mx)
                self.mx = None
            self.closed = True

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
        if self.closed:
            raise RuntimeError("GDN is closed")
        if not self.prepared:
            raise RuntimeError("GDN sizing instance cannot execute")
        weight = self.weights
        if self.use_mx and (self.mx is None or weight.mx is None):
            raise RuntimeError("MXFP8 GDN is not prepared")
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
            self.max_queries,
            self.mx,
            weight.mx if self.use_mx else None,
        )
        return self.buffers[:2]
