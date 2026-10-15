"""Verify the accepted comparison before any formal timing."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def verify_reference(root=ROOT):
    paths = [
        root / "development/kernels/oracles.py",
        root / "development/fp64_math.py",
        root / "development/qwen_fp64_reference.py",
    ]
    return {
        "kind": "independent_torch_mathematics",
        "arithmetic_dtype": "float64",
        "storage_boundaries": "explicit casts in independent operator equations",
        "files": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "runtime_requirements_sha256": hashlib.sha256(
            (root / "requirements/runtime.txt").read_bytes()
        ).hexdigest(),
    }


def source_hashes(root=ROOT):
    paths = [
        *root.glob("python/oh_my_vllm/kernels/**/*.py"),
        *root.glob("python/oh_my_vllm/kernels/**/*.cu"),
        *root.glob("python/oh_my_vllm/kernels/**/*.cuh"),
        root / "python/oh_my_vllm/models/qwen.py",
        root / "python/oh_my_vllm/models/dspark.py",
        *root.glob("python/oh_my_vllm/ir/**/*.py"),
        *root.glob("development/kernels/*.py"),
        root / "development/fp64_math.py",
        root / "development/qwen_fp64_reference.py",
        root / "requirements/runtime.txt",
        root / "benchmarks/kernels.py",
        root / "development/kernels/original-case-ids.json",
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def native_inputs():
    import json

    from oh_my_vllm.kernels.cuda_backend.groupwise import (
        _FLAGS,
        _dependency_roots,
        _headers,
        _project_headers,
    )

    data, roots = _dependency_roots()
    return {
        "headers_sha256": hashlib.sha256(
            json.dumps(_headers(data, roots), sort_keys=True).encode()
        ).hexdigest(),
        "flags": list(_FLAGS),
        "project_headers": _project_headers(),
    }


def validate_native_inputs(loaded, expected):
    if not isinstance(loaded, dict) or any(
        (list(loaded.get(key, ())) if key == "flags" else loaded.get(key)) != value
        for key, value in expected.items()
    ):
        raise RuntimeError("loaded native GEMM inputs differ from this collection")
