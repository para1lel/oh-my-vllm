import ctypes
import math
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from time import perf_counter

import numpy as np

BOOTSTRAP_SEED = 781049
RESAMPLES = 10000
MINIMUM_SAMPLES = 30
EXTENDED_SAMPLES = 120
TARGET_SECONDS = 0.020
PILOT_SECONDS = 0.025
VERSION = "operator-timing-v1"


@dataclass(frozen=True)
class Sample:
    gpu_seconds: float
    host_seconds: float
    repeats: int


@dataclass(frozen=True)
class PerformanceResult:
    median: float
    lower: float
    upper: float
    decision: str


class Events(AbstractContextManager):
    def __init__(self, device):
        self.device = device
        self.start, self.end = ctypes.c_void_p(), ctypes.c_void_p()
        try:
            device.checked(device.runtime.cudaEventCreate(ctypes.byref(self.start)))
            device.checked(device.runtime.cudaEventCreate(ctypes.byref(self.end)))
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def record(self, event):
        import tvm_ffi

        stream = tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))
        self.device.checked(self.device.runtime.cudaEventRecord(event, stream))

    def begin(self):
        self.record(self.start)

    def record_end(self):
        self.record(self.end)

    def elapsed(self):
        self.device.checked(self.device.runtime.cudaEventSynchronize(self.end))
        elapsed = ctypes.c_float()
        self.device.checked(
            self.device.runtime.cudaEventElapsedTime(
                ctypes.byref(elapsed), self.start, self.end
            )
        )
        return elapsed.value / 1000

    def __exit__(self, *_):
        self.device.destroy_handles(
            (("cudaEventDestroy", self.end), ("cudaEventDestroy", self.start))
        )
        self.start = self.end = None


def sample(prepared, events, repeats, *, graph=None):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("sample repeat count must be positive")
    prepared.reset()
    prepared.device.synchronize()
    if graph is None:
        events.begin()
        started = perf_counter()
        for _ in range(repeats):
            prepared.invoke()
        host = (perf_counter() - started) / repeats
        events.record_end()
        elapsed = events.elapsed()
    else:
        graph()
        elapsed = events.elapsed()
        prepared.reset()
        prepared.device.synchronize()
        started = perf_counter()
        prepared.invoke()
        host = perf_counter() - started
        prepared.device.synchronize()
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("invalid GPU sample duration")
    return Sample(elapsed / repeats, host, repeats)


def repeat_safe(case):
    if case.boundary in {"mtp.accept", "mtp.draft"}:
        return False
    if case.boundary == "output.sample":
        return case.options["greedy"]
    return True


class ShortInterval(ValueError):
    def __init__(self, repeats):
        super().__init__("repeat-safe interval is shorter than20ms")
        self.repeats = repeats


def collect(sampler, count, repeatable, *, repeats=None, retries=None):
    if type(count) is not int or count < 1:
        raise ValueError("sample count must be positive")
    if repeats is not None and (type(repeats) is not int or repeats < 1):
        raise ValueError("sample repeat count must be positive")
    if not repeatable and repeats not in (None, 1):
        raise ValueError("this case must be restored before every invocation")
    fixed = repeats is not None
    if repeats is None:
        for _ in range(3):
            sampler(1)
        pilot = sampler(1)
        repeats = (
            max(1, math.ceil(PILOT_SECONDS / pilot.gpu_seconds)) if repeatable else 1
        )
    for _ in range(8):
        pilot = sampler(repeats)
        duration = pilot.gpu_seconds * repeats
        required = TARGET_SECONDS if fixed else PILOT_SECONDS
        if repeatable and duration < required:
            next_repeats = max(
                repeats + 1, math.ceil(repeats * PILOT_SECONDS / duration)
            )
            if fixed:
                raise ShortInterval(next_repeats)
            repeats = next_repeats
            continue
        samples = [sampler(repeats) for _ in range(count)]
        duration = min(value.gpu_seconds * repeats for value in samples)
        if not repeatable or duration >= TARGET_SECONDS:
            return samples
        next_repeats = max(repeats + 1, math.ceil(repeats * PILOT_SECONDS / duration))
        if retries is not None:
            retries.append(
                dict(repeats=repeats, discarded_samples=count, minimum_seconds=duration)
            )
        if fixed:
            raise ShortInterval(next_repeats)
        repeats = next_repeats
    raise ValueError("cannot obtain a stable measurement interval")


def measure(prepared, *, count=MINIMUM_SAMPLES, events_factory=Events, repeats=None):
    from .timing_graph import TimingGraph

    with events_factory(prepared.device) as events, ExitStack() as graphs:
        active_repeats, replay = None, None

        def sampler(number):
            nonlocal active_repeats, replay
            if prepared.case.mode == "graph" and active_repeats != number:
                graphs.close()
                replay = graphs.enter_context(TimingGraph(prepared, events, number))
                active_repeats = number
            return sample(prepared, events, number, graph=replay)

        return collect(
            sampler,
            count,
            repeat_safe(prepared.case),
            repeats=repeats,
            retries=prepared.setup.setdefault("timing_retries", []),
        )


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) < MINIMUM_SAMPLES
        or not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        raise ValueError("at least30 finite positive samples are required")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    medians = np.median(
        values[rng.integers(0, len(values), size=(RESAMPLES, len(values)))], axis=1
    )
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(np.median(values)), float(low), float(high)


def check(samples, lower_bound, max_ratio):
    if not all(
        math.isfinite(value) and value > 0 for value in (lower_bound, max_ratio)
    ):
        raise ValueError("theoretical latency and ratio limit must be positive")
    median, low, high = interval(
        [sample.gpu_seconds / lower_bound for sample in samples]
    )
    decision = (
        "pass" if high <= max_ratio else "fail" if low > max_ratio else "unresolved"
    )
    return PerformanceResult(median, low, high, decision)


def accept(prepared, lower_bound, max_ratio, *, measure_function=measure):
    samples = measure_function(prepared, count=MINIMUM_SAMPLES)
    result = check(samples, lower_bound, max_ratio)
    if result.decision != "unresolved":
        return samples, result
    repeats = samples[0].repeats
    count = EXTENDED_SAMPLES - len(samples)
    for _ in range(8):
        try:
            samples.extend(measure_function(prepared, count=count, repeats=repeats))
            return samples, check(samples, lower_bound, max_ratio)
        except ShortInterval as error:
            prepared.setup.setdefault("timing_retries", []).append(
                dict(
                    repeats=repeats, discarded_samples=len(samples), reason="extension"
                )
            )
            samples = []
            count, repeats = EXTENDED_SAMPLES, error.repeats
    raise ValueError("cannot obtain a stable extended measurement interval")


def calibrate(sessions, lower_bound):
    if len(sessions) != 5:
        raise ValueError("calibration requires five independent sessions")
    upper = max(
        interval([sample.gpu_seconds / lower_bound for sample in session])[2]
        for session in sessions
    )
    return math.ceil(upper * 10000) / 10000
