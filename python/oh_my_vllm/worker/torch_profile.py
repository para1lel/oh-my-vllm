"""Bounded, explicitly requested benchmark profiling of device timeline events."""

import json
import logging
from contextlib import contextmanager, suppress
from pathlib import Path

logger = logging.getLogger("oh_my_vllm.worker.torch_profile")
ROOT = Path(__file__).resolve().parents[3]


def kernel_timeline(trace: dict) -> list[dict]:
    events = trace["traceEvents"]
    operations = {
        event.get("args", {}).get("External id"): event["name"]
        for event in events
        if event.get("cat") == "cpu_op"
        and event.get("args", {}).get("External id") is not None
    }
    kernels = [
        event
        for event in events
        if event.get("cat") == "kernel" and event.get("ph") == "X"
    ]
    if not kernels:
        raise RuntimeError("torch profiler did not collect CUDA kernel timeline events")
    origin = min(event["ts"] for event in kernels)
    return [
        {
            "kernel": event["name"],
            "operator": operations.get(event.get("args", {}).get("External id")),
            "start_us": event["ts"] - origin,
            "duration_us": event["dur"],
            "device": event.get("args", {}).get("device"),
            "stream": event.get("args", {}).get("stream"),
        }
        for event in sorted(kernels, key=lambda item: item["ts"])
    ]


class BenchmarkProfiler:
    def __init__(self, configuration: dict | None):
        self.profile = None
        self.completed = False
        self.seen = 0
        self.recorded = 0
        self.path = None
        if configuration is None:
            return
        self.skip = configuration["skip_steps"]
        self.steps = configuration["steps"]
        if (
            type(self.skip) is not int
            or not 0 <= self.skip <= 10000
            or type(self.steps) is not int
            or not 1 <= self.steps <= 8
        ):
            raise ValueError(
                "benchmark profiler needs 0..10000 skipped and 1..8 active steps"
            )
        self.path = Path(configuration["path"]).resolve()
        if self.path.is_relative_to(ROOT) or self.path.exists():
            raise ValueError("benchmark profiler needs a new external trace path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x"):
            pass

    @contextmanager
    def step(self, step_id: int, tokens: int):
        if self.path is None or self.completed or tokens == 0:
            yield
            return
        if tokens > 4096:
            raise ValueError(
                "benchmark profiling is limited to 4096 query tokens per step"
            )
        self.seen += 1
        if self.seen <= self.skip:
            yield
            return
        import torch

        if self.profile is None:
            self.profile = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            self.profile.__enter__()
        try:
            with torch.profiler.record_function(f"inference_step_{step_id}"):
                yield
        except BaseException:
            with suppress(Exception):
                self.close(incomplete=True)
            raise
        else:
            self.recorded += 1
            if self.recorded == self.steps:
                self.close()

    def close(self, *, incomplete=False):
        if self.completed or self.path is None:
            return
        self.completed = True
        if self.profile is None:
            logger.warning(
                "torch_profile_window_not_started",
                extra={"fields": {"seen_steps": self.seen, "incomplete": True}},
            )
            return
        profile, self.profile = self.profile, None
        profile.__exit__(None, None, None)
        profile.export_chrome_trace(str(self.path))
        rows = kernel_timeline(json.loads(self.path.read_text()))
        logger.info(
            "torch_profile_window",
            extra={
                "fields": {
                    "trace_path": str(self.path),
                    "steps": self.recorded,
                    "incomplete": incomplete or self.recorded != self.steps,
                    "kernels": len(rows),
                }
            },
        )
        for row in rows:
            logger.info("torch_profile_kernel", extra={"fields": row})
