from dataclasses import dataclass


def scratch_bytes(rows, heads, slots):
    tiles = rows * (heads // 16) * ((slots + 63) // 64) if rows <= 4 else 0
    errors = rows * (heads // 16) * 64 if tiles else 0
    return (tiles * (16 * 64 + 16 + 16 + 16 * 512 + 1) + errors) * 4


@dataclass(frozen=True)
class Plan:
    rows: int
    heads: int
    slots: int

    @property
    def reuse_key(self):
        return self.rows, self.heads, self.slots

    @property
    def storage(self):
        return (
            ((self.rows, self.heads, 512), "bfloat16"),
            ((self.rows, self.heads // 16), "uint8"),
        )


def make_plan(*, rows, heads, slots):
    if (
        any(type(x) is not int for x in (rows, heads, slots))
        or not 0 <= rows <= 384
        or heads not in (16, 64)
        or not 0 <= slots <= 640
    ):
        raise ValueError(
            "sparse attention requires rows 0..384, heads 16/64 and slots 0..640"
        )
    return Plan(rows, heads, slots)
