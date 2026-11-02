from dataclasses import dataclass


def round128(value):
    return (value + 127) // 128 * 128


def scale_bytes(rows, width):
    return round128(rows) * (round128(width) // 32)


@dataclass(frozen=True)
class Plan:
    rows: int
    outputs: int
    width: int
    bits: int

    @property
    def reuse_key(self):
        return self.rows, self.outputs, self.width, self.bits

    @property
    def padded_width(self):
        return round128(self.width) if self.bits == 4 else self.width


def make_plan(*, rows, outputs, width, bits):
    if (
        any(type(value) is not int for value in (rows, outputs, width, bits))
        or not 0 <= rows <= 384
        or not 32 <= outputs <= 32768
        or outputs % 32
        or not 32 <= width <= 16384
        or width % 32
        or bits not in (4, 8)
    ):
        raise ValueError("invalid encoded projection dimensions or weight bits")
    return Plan(rows, outputs, width, bits)
