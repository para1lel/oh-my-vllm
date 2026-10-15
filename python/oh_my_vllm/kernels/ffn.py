from dataclasses import dataclass

from .native import load


@dataclass(frozen=True)
class Weights:
    norm: object
    gate_up: object
    gate_up_scale: object
    down: object
    down_scale: object


class FFN:
    def __init__(self, weights, rows, allocate):
        self.module, self.source_id, self.library = load("ffn", ("ffn",))
        self.weights = weights
        self.buffers = tuple(
            allocate((rows, width), dtype)
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
        splits = 8 if rows < 8 else 2
        self.buffers += (
            allocate(
                (splits if rows <= 32 else 1, rows if rows <= 32 else 1, 5120),
                "float32",
            ),
        )

    def __call__(self, value, residual):
        weight = self.weights
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
