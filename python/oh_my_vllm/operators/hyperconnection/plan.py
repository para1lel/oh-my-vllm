from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    rows: int

    @property
    def reuse_key(self):
        return (self.rows,)

    @property
    def storage(self):
        rows = self.rows
        return (
            ((rows, 5120), "bfloat16"),
            ((rows, 4, 5120), "bfloat16"),
            ((rows, 4), "float32"),
            ((rows, 24), "float32"),
            ((rows, 4), "float32"),
            ((rows, 4, 4), "float32"),
        )


def make_plan(*, rows):
    if type(rows) is not int or not 0 <= rows <= 384:
        raise ValueError("HC row count must be between zero and 384")
    return Plan(rows)
