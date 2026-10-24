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

    @property
    def reuse_key(self):
        return self.rows, self.experts, self.expert_start, self.expert_count

    @property
    def storage(self):
        capacity = self.rows * self.topk
        return (
            ((capacity, 5120), "bfloat16"),
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


def make_dispatch_plan(*, rows, experts, expert_start=0, expert_count=None):
    plan = make_plan(rows=rows, experts=experts)
    count = experts if expert_count is None else expert_count
    if type(count) is not int or count not in (experts, experts // 4):
        raise ValueError("dispatch owns all experts or one quarter")
    if (
        type(expert_start) is not int
        or not 0 <= expert_start <= experts - count
        or expert_start % count
    ):
        raise ValueError("dispatch expert interval is not aligned")
    return DispatchPlan(plan.rows, plan.experts, expert_start, count)
