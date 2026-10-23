from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    rows: int

    @property
    def reuse_key(self):
        return (self.rows,)

    @property
    def storage(self):
        return (((self.rows, 512), "bfloat16"),)


def make_plan(*, rows):
    if type(rows) is not int or not 0 <= rows <= 384:
        raise ValueError("pooling row count must be between zero and 384")
    return Plan(rows)
