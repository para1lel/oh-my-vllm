"""Explicit custom-kernel selection; unsupported CUDA entries never fall back."""

import os

NAME = os.environ.get("OH_MY_VLLM_KERNEL_BACKEND", "tilelang")
if NAME not in ("tilelang", "cuda"):
    raise ValueError(f"unknown OH_MY_VLLM_KERNEL_BACKEND: {NAME!r}")


def kernel(factory):
    if NAME == "tilelang":
        from importlib import import_module

        module = factory.__module__.rsplit(".", 1)[-1]
        frozen = import_module(f"oh_my_vllm.kernels.tilelang_reference.{module}")
        return getattr(frozen, factory.__name__)
    from .cuda_backend import factory_for

    return factory_for(factory.__module__.rsplit(".", 1)[-1], factory.__name__)
