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


@dataclass(frozen=True)
class GroupedPlan:
    capacity: int
    groups: int
    outputs: int
    width: int

    @property
    def padded_width(self):
        return round128(self.width)

    @property
    def scale_capacity(self):
        return (round128(self.capacity) + self.groups * 128) * (self.padded_width // 32)


def make_grouped_plan(*, capacity, groups, outputs, width):
    make_plan(rows=0, outputs=outputs, width=width, bits=4)
    if (
        type(capacity) is not int
        or not 0 <= capacity <= 2304
        or type(groups) is not int
        or not 1 <= groups <= 384
        or capacity > groups * 384
    ):
        raise ValueError("invalid grouped projection capacity or group count")
    return GroupedPlan(capacity, groups, outputs, width)
