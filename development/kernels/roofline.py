import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Work:
    hbm_bytes: int
    fp8_flops: int = 0
    bf16_flops: int = 0
    fp32_flops: int = 0
    tf32_flops: int = 0
    fp64_flops: int = 0

    def lower_bound_seconds(self, peaks):
        if any(value < 0 or not math.isfinite(value) for value in vars(self).values()):
            raise ValueError("work must be finite and nonnegative")
        return max(
            self.hbm_bytes / peaks.hbm,
            self.fp8_flops / peaks.fp8
            + self.bf16_flops / peaks.bf16
            + self.tf32_flops / peaks.tf32,
            self.fp32_flops / peaks.fp32,
            self.fp64_flops / peaks.fp64,
        )


@dataclass(frozen=True)
class ExpectedCase:
    operator: str
    calls: float
    work: Work


@dataclass(frozen=True)
class Observation:
    operator: str
    case_id: str
    calls: float
    lower_bound_seconds: float
    measured_seconds: float
    source_id: str

    def efficiency(self):
        if not self.source_id or any(
            not math.isfinite(value) or value <= 0
            for value in (
                self.calls,
                self.lower_bound_seconds,
                self.measured_seconds,
            )
        ):
            raise ValueError("observation has no valid identity or timing")
        value = self.lower_bound_seconds / self.measured_seconds
        return value


def expected_efficiencies(observations, required_cases, source_id, peaks):
    if not required_cases or not source_id:
        raise ValueError("a serving model and source identity are required")
    measured = {}
    for item in observations:
        if item.case_id in measured:
            raise ValueError(f"duplicate case: {item.case_id}")
        measured[item.case_id] = item
    if set(measured) != set(required_cases):
        raise ValueError("missing or unexpected serving shapes")
    sums = {}
    calls = {}
    for case_id, item in measured.items():
        expected = required_cases[case_id]
        lower_bound = expected.work.lower_bound_seconds(peaks)
        if (
            item.source_id != source_id
            or item.operator != expected.operator
            or not math.isclose(item.calls, expected.calls, rel_tol=1e-12)
            or not math.isclose(item.lower_bound_seconds, lower_bound, rel_tol=1e-12)
        ):
            raise ValueError("measurement differs from the serving model or source")
        value = expected.calls * item.efficiency()
        sums[expected.operator] = sums.get(expected.operator, 0) + value
        calls[expected.operator] = calls.get(expected.operator, 0) + expected.calls
    return {name: value / calls[name] for name, value in sums.items()}
