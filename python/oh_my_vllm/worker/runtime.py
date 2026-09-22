"""Runtime identity and checks against accidental legacy dependencies."""

import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path


def identity() -> dict:
    if importlib.util.find_spec("vllm") is not None:
        raise RuntimeError("the independent environment must not provide vLLM")
    forbidden = [path for path in sys.path if path and "vllm" in Path(path).parts]
    if forbidden:
        raise RuntimeError(f"legacy source/environment on Python path: {forbidden}")
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "runner": "oh_my_vllm.worker.model_runner.OhMyVllmWorker",
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "triton",
                "tilelang",
                "apache-tvm-ffi",
                "flashinfer-python",
                "transformers",
                "tokenizers",
                "safetensors",
                "xgrammar",
                "pyzmq",
                "msgpack",
            )
        },
        "flashinfer_workspace": os.environ.get("FLASHINFER_WORKSPACE_BASE"),
        "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
        "tilelang_cache": os.environ.get("TILELANG_CACHE_DIR"),
        "native_cuda_cache": os.environ.get("TVM_FFI_CACHE_DIR"),
        "kernel_backend": os.environ.get("OH_MY_VLLM_KERNEL_BACKEND", "tilelang"),
        "vllm_importable": False,
    }


def verify_loaded_modules() -> None:
    """Check after model/graph execution, including libraries loaded lazily."""
    if any(name == "vllm" or name.startswith("vllm.") for name in sys.modules):
        raise RuntimeError("vLLM was imported into the independent worker")
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and "vllm" in Path(parts[5]).parts:
            raise RuntimeError(f"legacy mapped library: {parts[5]}")
