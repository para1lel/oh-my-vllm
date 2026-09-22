"""Explicit custom-kernel selection; unsupported CUDA entries never fall back."""

import os

NAME = os.environ.get("OH_MY_VLLM_KERNEL_BACKEND", "cuda")
if NAME not in ("tilelang", "cuda"):
    raise ValueError(f"unknown OH_MY_VLLM_KERNEL_BACKEND: {NAME!r}")


def kernel(module, name):
    if NAME == "tilelang":
        from importlib import import_module

        frozen = import_module(f"oh_my_vllm.kernels.tilelang_reference.{module}")
        return getattr(frozen, name)
    from .cuda_backend import factory_for

    return factory_for(module, name)
