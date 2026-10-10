"""Owned SM100 CUTLASS launch and bounded per-stream scheduler workspaces."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from functools import cache
from pathlib import Path

import torch

from oh_my_vllm.ir.execution import execution_policy

_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}
_PROVENANCE = None
_FLAGS = (
    "-O3",
    "-DNDEBUG",
    "-use_fast_math",
    "--expt-relaxed-constexpr",
    "-static-global-template-stub=false",
    "--generate-code=arch=compute_100a,code=sm_100a",
    "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
)


def _headers(data, roots):
    return {
        str(path.relative_to(data)): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@cache
def compiled():
    """Compile only the checkpoint's two regular FP8 GEMM tactics."""
    from tvm_ffi import load_module
    from tvm_ffi.cpp import build_inline

    from . import _effective_cuda_target, _nvcc_identity

    if (
        os.environ.get("OH_MY_VLLM_CUDA_DEBUG_SYNC") == "1"
        and os.environ.get("OH_MY_VLLM_ENFORCE_EAGER") != "1"
    ):
        raise RuntimeError("CUDA debug sync requires OH_MY_VLLM_ENFORCE_EAGER=1")

    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("owned groupwise FP8 GEMM requires SM100")
    data = importlib.metadata.distribution("flashinfer-python").locate_file(
        "flashinfer/data"
    )
    include = data / "cutlass/include"
    utilities = data / "cutlass/tools/util/include"
    roots = (
        data / "cccl/cub",
        data / "cccl/libcudacxx/include",
        data / "cccl/thrust",
        include,
        utilities,
    )
    source_directory = Path(__file__).parent
    source = (source_directory / "groupwise_fp8.cu").read_text()
    project_headers = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_directory.glob("*.cuh"))
    }
    # All template headers participate in the build identity. A wheel version
    # alone does not identify an edited local dependency installation.
    headers = _headers(data, roots)
    compiler, compiler_version = _nvcc_identity()
    if compiler is None or compiler_version.startswith("nvcc-unavailable:"):
        raise RuntimeError("owned groupwise GEMM compilation needs a CUDA compiler")
    identity = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "headers_sha256": hashlib.sha256(
            json.dumps(headers, sort_keys=True).encode()
        ).hexdigest(),
        "project_headers": project_headers,
        "compiler": str(compiler),
        "compiler_version": compiler_version,
        "torch": str(torch.__version__),
        "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
        "flags": _FLAGS,
        "functions": ["groupwise_fp8"],
        "backend": "cuda",
        "tvm_cuda_target": _effective_cuda_target(),
        "cxx": os.environ.get("CXX", "c++"),
        "python_abi": sys.implementation.cache_tag,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(
        build_inline(
            "oh_my_vllm_groupwise_fp8",
            cuda_sources=source,
            functions=["groupwise_fp8"],
            backend="cuda",
            extra_include_paths=[str(source_directory), *(str(root) for root in roots)],
            extra_cuda_cflags=[
                *_FLAGS,
                f"-DOH_MY_VLLM_GEMM_INPUT_{digest[:20]}",
            ],
        )
    ).resolve()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    module = load_module(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise RuntimeError("owned groupwise module changed while loading")
    if (
        _nvcc_identity() != (compiler, compiler_version)
        or _headers(data, roots) != headers
        or {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_directory.glob("*.cuh"))
        }
        != project_headers
    ):
        raise RuntimeError("owned groupwise compiler or headers changed during build")
    global _PROVENANCE
    _PROVENANCE = dict(
        identity, build_input_sha256=digest, so_sha256=sha, so_path=str(path)
    )
    print(
        "CUDA_GEMM_BUILD_PROVENANCE " + json.dumps(_PROVENANCE),
        file=sys.stderr,
        flush=True,
    )
    return module


def provenance():
    return None if _PROVENANCE is None else dict(_PROVENANCE)


def _workspace(device: torch.device) -> torch.Tensor:
    """Persistent scratch is private to the stream that writes it.

    A worker uses its origin, graph-capture, and branch streams. Four slots
    bound scratch memory to 64 bytes per device and reject unbounded stream churn.
    CUDA Graph objects keep the addresses alive through this worker-lifetime map.
    """
    stream = torch.cuda.current_stream(device)
    key = (device.index, stream.cuda_stream)
    if key not in _WORKSPACES:
        if sum(index == device.index for index, _ in _WORKSPACES) >= 4:
            raise RuntimeError("owned FP8 workspace stream limit exceeded")
        # The pinned CLC/epilogue requires zero scratch. Keep an aligned pointer
        # and let the C++ size guard reject a future nonzero requirement.
        _WORKSPACES[key] = torch.empty(16, device=device, dtype=torch.uint8)
    return _WORKSPACES[key]


def gemm(a, weight, scale_a, scale_weight, *, mma_sm=1, scale_major_k=True):
    """Run regular M > 32 GEMM with explicit PDL and caller-stream initialization."""
    with torch.cuda.device(a.device):
        module = compiled()
    output = torch.empty(
        (a.shape[0], weight.shape[0]), device=a.device, dtype=torch.bfloat16
    )
    module.groupwise_fp8(
        a,
        weight,
        scale_a,
        scale_weight,
        output,
        _workspace(a.device),
        mma_sm,
        execution_policy().pdl,
        scale_major_k,
    )
    return output
