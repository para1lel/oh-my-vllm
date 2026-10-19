import json
import os
from pathlib import Path

from .execution import execution_policy


class Trace:
    def __init__(self, config, identity, *, tokenizer=None):
        path = os.environ.get("OH_MY_VLLM_EXECUTION_TRACE")
        self.output = None
        if path:
            path = Path(path).expanduser().resolve()
            if path.is_relative_to(Path(__file__).resolve().parents[3]):
                raise ValueError("execution trace must be outside the repository")
            self.output = path.open("x")
            self.record(
                "identity",
                config=config,
                checkpoint=identity,
                tokenizer=tokenizer,
                execution=execution_policy(),
            )

    def record(self, operation, **values):
        if self.output is not None:
            self.output.write(json.dumps({"operation": operation, **values}) + "\n")
            if operation in {"identity", "commit"}:
                self.output.flush()

    def close(self):
        if self.output is not None:
            self.output.close()
            self.output = None
