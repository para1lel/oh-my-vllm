"""Worker-lifetime CUDA execution policy and one bounded fork/join stream."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache

import torch


@dataclass(frozen=True)
class ExecutionPolicy:
    """Frozen before compilation, so a captured unit keeps its execution policy."""

    multi_stream: bool
    pdl: bool


def _switch(name: str) -> bool:
    value = os.environ.get(name, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


@cache
def execution_policy() -> ExecutionPolicy:
    return ExecutionPolicy(
        multi_stream=_switch("OH_MY_VLLM_MULTI_STREAM"),
        pdl=_switch("OH_MY_VLLM_CUDA_PDL"),
    )


@cache
def branch_stream(device_index: int) -> torch.cuda.Stream:
    """One stream per device, shared by sequentially joined model regions."""
    return torch.cuda.Stream(device=device_index)
