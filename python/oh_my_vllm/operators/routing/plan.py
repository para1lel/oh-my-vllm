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


@dataclass(frozen=True)
class DispatchPlan(Plan):
    expert_start: int
    expert_count: int
    encoded: bool = False

    @property
    def reuse_key(self):
        return (
            self.rows,
            self.experts,
            self.expert_start,
            self.expert_count,
            self.encoded,
        )

    @property
    def scale_capacity(self):
        capacity = self.rows * self.topk
        return ((capacity + 127) // 128 + self.expert_count) * 128 * 160

    @property
    def storage(self):
        capacity = self.rows * self.topk
        values = (
            (((capacity, 5120), "uint8"), ((self.scale_capacity,), "uint8"))
            if self.encoded
            else (((capacity, 5120), "bfloat16"),)
        )
        return (
            *values,
            ((capacity,), "float32"),
            ((self.expert_count,), "int32"),
            ((self.expert_count + 1,), "int32"),
            ((self.rows, self.topk), "int32"),
            ((self.rows,), "uint8"),
        )

    @property
    def combine_storage(self):
        return (((self.rows, 5120), "float32"), ((self.rows,), "uint8"))

    @property
    def scratch(self):
        return ((self.rows * self.topk,), "int32")


def make_dispatch_plan(
    *, rows, experts, expert_start=0, expert_count=None, encoded=False
):
    plan = make_plan(rows=rows, experts=experts)
    if type(encoded) is not bool:
        raise ValueError("encoded dispatch mode must be a boolean")
    count = experts if expert_count is None else expert_count
    if type(count) is not int or count not in (experts, experts // 4):
        raise ValueError("dispatch owns all experts or one quarter")
    if (
        type(expert_start) is not int
        or not 0 <= expert_start <= experts - count
        or expert_start % count
    ):
        raise ValueError("dispatch expert interval is not aligned")
    return DispatchPlan(plan.rows, plan.experts, expert_start, count, encoded)
