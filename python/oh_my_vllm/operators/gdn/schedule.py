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
