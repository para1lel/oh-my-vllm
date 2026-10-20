from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    rows: int
    hidden_only: bool
    allow_invalid: bool
    storage: tuple

    @property
    def reuse_key(self):
        return (self.rows, self.hidden_only, self.allow_invalid)


def make_plan(*, rows, hidden_only=False, allow_invalid=False):
    if not 1 <= rows <= (32768 if hidden_only else 20):
        raise ValueError("invalid output row count")
    storage = (((rows, 5120), "bfloat16"),)
    if not hidden_only:
        storage += (
            ((rows, 248320), "bfloat16"),
            ((rows, 248320), "uint64"),
            ((rows, 248320), "uint64"),
            ((rows, 248320), "float32"),
            ((rows, 248320), "int32"),
            ((rows, 248320), "float32"),
            ((rows,), "int64"),
            ((rows,), "float32"),
        )
    return Plan(rows, hidden_only, allow_invalid, storage)
