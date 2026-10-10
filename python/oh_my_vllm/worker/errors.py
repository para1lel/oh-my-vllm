"""Classify failures that invalidate a worker's GPU transaction."""

from functools import wraps

import torch


class WorkerUnavailable(RuntimeError):
    """The worker must be reconstructed before it can accept another operation."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(f"worker state is invalid after {kind}; rebuild required")


def failure_kind(error: Exception) -> str:
    if isinstance(error, WorkerUnavailable):
        return error.kind
    if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)):
        return "oom"
    message = str(error).lower()
    if "out of memory" in message and any(
        marker in message for marker in ("cuda", "gpu", "cublas", "cudnn")
    ):
        return "oom"
    if isinstance(error, torch.AcceleratorError) or any(
        marker in message
        for marker in (
            "cuda",
            "cublas",
            "cudnn",
            "device-side assert",
            "illegal memory",
            "launch failure",
        )
    ):
        return "device"
    return "internal"


def transactional_step(function):
    """Fence an incomplete step, including failures after accepted-state writes.

    Request-local failures contained in WorkerOutput do not reach this boundary.
    An escaping exception can follow device writes or host commits, so replaying
    the same step in this worker would be unsafe.
    """

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        kind = getattr(self, "_failure", None)
        if kind is not None:
            raise WorkerUnavailable(kind)
        try:
            return function(self, *args, **kwargs)
        except Exception as error:
            self._failure = failure_kind(error)
            raise

    return wrapped


def resource_operation(function):
    """Fence resource faults in registration; keep rejected inputs request-local."""

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        kind = getattr(self, "_failure", None)
        if kind is not None:
            raise WorkerUnavailable(kind)
        try:
            return function(self, *args, **kwargs)
        except Exception as error:
            kind = failure_kind(error)
            if kind in ("oom", "device"):
                self._failure = kind
            raise

    return wrapped
