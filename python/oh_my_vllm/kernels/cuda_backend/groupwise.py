"""Owned SM100 CUTLASS launch with zero scheduler workspace."""

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

_WORKSPACES: dict[int, torch.Tensor] = {}
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


def _dependency_roots():
    from . import _nvcc_identity

    compiler, _ = _nvcc_identity()
    if compiler is None:
        raise RuntimeError("native GEMM headers require a CUDA toolkit")
    data = Path(__file__).resolve().parents[4] / "3rdparty/cutlass"
    roots = (
        compiler.parent.parent / "include/cccl",
        data / "include",
        data / "tools/util/include",
    )
    if any(not root.is_dir() for root in roots):
        raise RuntimeError("native CUTLASS or CUDA CCCL headers are missing")
    return data, roots


def _headers(data, roots):
    headers = {}
    for root in roots:
        prefix = (
            root.relative_to(data).as_posix() if root.is_relative_to(data) else "cccl"
        )
        for path in sorted(root.rglob("*")):
            if path.is_file():
                headers[f"{prefix}/{path.relative_to(root)}"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return headers


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
    data, roots = _dependency_roots()
    source_directory = Path(__file__).parent
    source = (source_directory / "groupwise_fp8.cu").read_text()
    project_headers = _project_headers()
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
        "functions": ["groupwise_fp8", "groupwise_fp8_splitk"],
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
            functions=["groupwise_fp8", "groupwise_fp8_splitk"],
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
        or _project_headers() != project_headers
        or (source_directory / "groupwise_fp8.cu").read_text() != source
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


def _project_headers() -> dict[str, str]:
    """Identify owned GEMM headers and their shared CUDA contracts."""
    root = Path(__file__).parent
    paths = [*root.glob("*.cuh"), root / "operators/common.cuh"]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def provenance():
    return None if _PROVENANCE is None else dict(_PROVENANCE)


def _workspace(device: torch.device) -> torch.Tensor:
    """Keep one device sentinel; the native zero-size guard forbids writes."""
    if device.index not in _WORKSPACES:
        _WORKSPACES[device.index] = torch.empty(16, device=device, dtype=torch.uint8)
    return _WORKSPACES[device.index]


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


def gemm_splitk(a, weight, scale_a, scale_weight):
    """Merge FP32 K segments before the projection's single BF16 boundary."""
    rows, width = a.shape
    splits = 8 if width == 17408 and rows <= 8 else 2
    with torch.cuda.device(a.device):
        module = compiled()
    output = torch.empty((rows, weight.shape[0]), device=a.device, dtype=torch.bfloat16)
    partials = torch.empty(
        (splits, rows, weight.shape[0]), device=a.device, dtype=torch.float32
    )
    module.groupwise_fp8_splitk(
        a,
        weight,
        scale_a,
        scale_weight,
        partials,
        output,
        _workspace(a.device),
        splits,
        execution_policy().pdl,
    )
    return output
