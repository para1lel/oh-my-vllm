import json
import os
from pathlib import Path

from .execution import execution_policy


class Trace:
    def __init__(
        self,
        config,
        identity,
        *,
        tokenizer=None,
        cache_memory=None,
        execution=None,
        cache_layout=None,
        loaded=None,
    ):
        path = os.environ.get("OH_MY_VLLM_EXECUTION_TRACE")
        self.output = None
        self.failure = None
        if path:
            try:
                path = Path(path).expanduser().resolve()
                if path.is_relative_to(Path(__file__).resolve().parents[3]):
                    raise ValueError("execution trace must be outside the repository")
                self.output = path.open("x")
            except (OSError, ValueError) as error:
                self.failure = str(error)
                return
            values = dict(
                config=config,
                checkpoint=identity,
                tokenizer=tokenizer,
                execution=execution if execution is not None else execution_policy(),
                **({"cache_memory": cache_memory} if cache_memory is not None else {}),
                **({"cache_layout": cache_layout} if cache_layout is not None else {}),
            )
            self.record("identity", **(values | (loaded or {})))

    def record(self, operation, **values):
        if self.output is None:
            return
        try:
            self.output.write(json.dumps({"operation": operation, **values}) + "\n")
            if operation in {
                "identity",
                "commit",
                "accounting_call",
                "accounting_error",
                "worker_scheduled",
                "worker_result",
                "prepared",
            }:
                self.output.flush()
        except (OSError, ValueError, TypeError) as error:
            self.failure = str(error)
            self.close()

    def close(self):
        output, self.output = self.output, None
        close = getattr(output, "close", None)
        if close is not None:
            try:
                close()
            except (OSError, ValueError) as error:
                self.failure = str(error)
