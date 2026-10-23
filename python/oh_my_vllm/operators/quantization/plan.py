from dataclasses import dataclass

MODES = ("fp8_ue8m0", "fp4_ue8m0", "fp4_e4m3")


@dataclass(frozen=True)
class Plan:
    rows: int
    width: int
    mode: str
    storage: tuple

    @property
    def native_mode(self):
        return MODES.index(self.mode)

    @property
    def reuse_key(self):
        return self.rows, self.width, self.mode


def make_plan(*, rows, width, mode):
    """Contiguous BF16 matrices, including zero rows; at most INT_MAX elements."""
    if (
        type(rows) is not int
        or type(width) is not int
        or not 0 <= rows <= 0x7FFFFFFF
        or not 0 < width <= 0x7FFFFFFF
        or rows * width > 0x7FFFFFFF
        or mode not in MODES
    ):
        raise ValueError("invalid quantization shape or mode")
    group_size = 16 if mode == "fp4_e4m3" else 32
    if width % group_size:
        raise ValueError(f"quantization width must be divisible by {group_size}")
    storage = (
        ((rows, width if mode == "fp8_ue8m0" else width // 2), "uint8"),
        ((rows, width // group_size), "uint8"),
        ((rows, width), "bfloat16"),
        ((rows, width // group_size), "uint8"),
    )
    return Plan(rows, width, mode, storage)
