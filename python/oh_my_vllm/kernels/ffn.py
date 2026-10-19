from contextlib import contextmanager
from dataclasses import dataclass, replace

from .native import load

FUNCTIONS = (
    "ffn",
    "ffn_mx_prepare_weights",
    "ffn_mx_prepare",
    "ffn_mx",
    "ffn_mx_graph_lease",
    "ffn_mx_close",
    "ffn_mx_images",
)
MX_WORKSPACE_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class Weights:
    norm: object
    gate_up: object
    gate_up_scale: object
    down: object
    down_scale: object
    mx: object = None


def prepare_weights(weights, allocate):
    if weights.mx is not None:
        return weights
    module, _, _ = load("ffn", FUNCTIONS)
    prepared = tuple(
        allocate(shape, dtype)
        for shape, dtype in (
            ((34816, 5120), "float8_e4m3fn"),
            ((34816 * 160,), "uint8"),
            ((5120, 17408), "float8_e4m3fn"),
            ((5120 * 544,), "uint8"),
            ((65536,), "float32"),
        )
    )
    owner = module.ffn_mx_prepare_weights(
        weights.norm,
        weights.gate_up,
        weights.gate_up_scale,
        weights.down,
        weights.down_scale,
        *prepared,
    )
    return replace(weights, mx=owner)


class FFN:
    def __init__(self, weights, rows, allocate, *, prepare=True):
        self.module, self.source_id, self.library = load("ffn", FUNCTIONS)
        self.weights = weights
        self.mx = None
        self.closed = False
        self.prepared = prepare
        self.use_mx = rows > 128
        gate_up_rows = rows if rows <= 64 else 0
        self.buffers = tuple(
            allocate(
                (
                    gate_up_rows
                    if width == 34816
                    else 0
                    if self.use_mx and width in (17408, 136)
                    else rows,
                    width,
                ),
                dtype,
            )
            for width, dtype in (
                (5120, "bfloat16"),
                (5120, "bfloat16"),
                (5120, "float8_e4m3fn"),
                (40, "float32"),
                (34816, "bfloat16"),
                (17408, "float8_e4m3fn"),
                (136, "float32"),
            )
        )
        splits = 2 if rows == 8 else 8
        self.buffers += (
            allocate(
                (splits if rows <= 32 else 1, rows if rows <= 32 else 1, 5120),
                "float32",
            ),
        )
        if self.use_mx:
            self._prepare_mx(rows, allocate, prepare)

    def _prepare_mx(self, rows, allocate, prepare):
        padded = (rows + 127) // 128 * 128
        self.mx_buffers = tuple(
            allocate(shape, dtype)
            for shape, dtype in (
                ((MX_WORKSPACE_BYTES,), "uint8"),
                ((padded, 5120), "float8_e4m3fn"),
                ((padded * 160,), "uint8"),
                ((padded, 34816), "bfloat16"),
                ((padded, 17408), "float8_e4m3fn"),
                ((padded * 544,), "uint8"),
                ((padded, 5120), "bfloat16"),
            )
        )
        if prepare:
            if self.weights.mx is None:
                raise ValueError("MXFP8 FFN weights must be prepared before use")
            self.mx = self.module.ffn_mx_prepare(
                self.weights.mx, *self.mx_buffers, rows
            )

    @contextmanager
    def graph_lease(self):
        if self.closed:
            raise RuntimeError("FFN is closed")
        if not self.prepared:
            raise RuntimeError("FFN sizing instance cannot execute")
        lease = self.module.ffn_mx_graph_lease(self.mx) if self.mx is not None else None
        try:
            yield
        finally:
            del lease

    def close(self):
        if not self.closed:
            if self.mx is not None:
                self.module.ffn_mx_close(self.mx)
                self.mx = None
            self.closed = True

    def __call__(self, value, residual):
        if self.closed:
            raise RuntimeError("FFN is closed")
        if not self.prepared:
            raise RuntimeError("FFN sizing instance cannot execute")
        weight = self.weights
        if self.use_mx:
            if self.mx is None or weight.mx is None:
                raise RuntimeError("MXFP8 FFN is not prepared")
            self.module.ffn_mx(self.mx, weight.mx, value, residual, *self.buffers[:4])
            return self.buffers[:2]
        self.module.ffn(
            value,
            residual,
            weight.norm,
            weight.gate_up,
            weight.gate_up_scale,
            weight.down,
            weight.down_scale,
            *self.buffers,
        )
        return self.buffers[:2]
