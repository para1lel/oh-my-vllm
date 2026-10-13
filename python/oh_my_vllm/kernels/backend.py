"""Repository CUDA kernels, compiled lazily for the caller stream."""

import os

NAME = os.environ.get("OH_MY_VLLM_KERNEL_BACKEND", "cuda")
if NAME != "cuda":
    raise ValueError(f"unknown OH_MY_VLLM_KERNEL_BACKEND: {NAME!r}")


def kernel(module, name):
    from .cuda_backend import factory_for

    return factory_for(module, name)
