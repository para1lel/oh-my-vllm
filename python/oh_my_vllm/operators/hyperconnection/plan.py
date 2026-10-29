from dataclasses import dataclass

from .api import BLAS_WORKSPACE_BYTES


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
            ((rows, 20480), "float32"),
            ((rows, 2), "float32"),
            ((BLAS_WORKSPACE_BYTES if rows else 0,), "uint8"),
        )


def make_plan(*, rows):
    if type(rows) is not int or not 0 <= rows <= 384:
        raise ValueError("HC row count must be between zero and 384")
    return Plan(rows)
