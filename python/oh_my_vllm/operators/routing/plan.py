from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    rows: int
    experts: int

    @property
    def topk(self):
        return 6 if self.experts == 384 else 3

    @property
    def reuse_key(self):
        return self.rows, self.experts

    @property
    def storage(self):
        shape = self.rows, self.topk
        return ((shape, "int32"), (shape, "float32"))


def make_plan(*, rows, experts):
    if type(rows) is not int or not 0 <= rows <= 384:
        raise ValueError("routing row count must be between zero and 384")
    if type(experts) is not int or experts not in (128, 384):
        raise ValueError("routing requires 128 or 384 experts")
    return Plan(rows, experts)
