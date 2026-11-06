import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


class _StartupTrace:
    def __init__(self, rank):
        template = os.environ.get("OH_MY_VLLM_STARTUP_TRACE")
        self.output = None
        if template:
            path = Path(template.format(rank=rank)).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.output = path.open("w")

    def record(self, phase, **values):
        if self.output is None:
            return
        record = {
            "phase": phase,
            "monotonic_s": time.monotonic(),
            **values,
        }
        self.output.write(json.dumps(record) + "\n")
        self.output.flush()

    @contextmanager
    def measure(self, phase):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(phase, duration_s=time.perf_counter() - started)

    def close(self):
        if self.output is not None:
            self.output.close()
            self.output = None
