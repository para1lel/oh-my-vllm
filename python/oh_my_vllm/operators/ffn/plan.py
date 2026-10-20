from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    rows: int
    storage: tuple

    @property
    def reuse_key(self):
        return (self.rows,)

    @property
    def needs_warmup(self):
        return self.rows > 32


def make_plan(*, rows):
    if not 1 <= rows <= 32768:
        raise ValueError("invalid FFN row count")
    storage = tuple(
        ((rows if width != 34816 or rows <= 64 else 0, width), dtype)
        for width, dtype in (
            (5120, "bfloat16"),
            (5120, "float32"),
            (5120, "float8_e4m3fn"),
            (40, "float32"),
            (34816, "bfloat16"),
            (17408, "float8_e4m3fn"),
            (136, "float32"),
        )
    )
    storage += (
        (
            (2 if rows == 8 else 8, rows, 5120) if rows <= 32 else (1, 1, 5120),
            "float32",
        ),
    )
    return Plan(rows, storage)
