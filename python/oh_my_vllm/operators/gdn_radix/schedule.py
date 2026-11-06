from dataclasses import dataclass
from itertools import pairwise


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
        starts, batch = self.sequence_rows, len(self.read_slots)
        if (
            not 1 <= self.slots <= 65536
            or not 1 <= batch <= 32768
            or len(starts) != batch + 1
            or starts[0] != 0
            or any(a > b for a, b in pairwise(starts))
            or starts[-1] != len(self.write_slots)
            or not 0 <= starts[-1] <= 32768
            or len(self.positions) != batch
            or len(self.checkpoint_rows) != len(self.checkpoint_slots)
            or any(not 0 <= slot < self.slots for slot in self.read_slots)
        ):
            raise ValueError("invalid GDN radix sequence or read slot schedule")
        for sequence, (first, last) in enumerate(pairwise(starts)):
            count, position = last - first, self.positions[sequence]
            if position < 0 or position + count > 262144:
                raise ValueError("GDN radix sequence exceeds the context limit")
            if self.verify and count > 5:
                raise ValueError("GDN verification has at most five rows per sequence")
        writes = list(self.write_slots)
        rows = self.checkpoint_rows
        if len(set(rows)) != len(rows) or any(
            not 0 <= row < len(writes) for row in rows
        ):
            raise ValueError("invalid GDN radix checkpoint rows")
        for row, slot in zip(rows, self.checkpoint_slots, strict=True):
            if not 0 <= slot < self.slots or writes[row] not in (-1, slot):
                raise ValueError("conflicting GDN radix checkpoint export")
            writes[row] = slot
        if self.verify and any(slot < 0 for slot in writes):
            raise ValueError("GDN verification must save each private row")
        exports = [slot for slot in writes if slot >= 0]
        if (
            any(slot < -1 or slot >= self.slots for slot in writes)
            or len(exports) != len(set(exports))
            or set(exports).intersection(self.read_slots)
        ):
            raise ValueError("GDN writes require distinct private state slots")
        return tuple(writes)
