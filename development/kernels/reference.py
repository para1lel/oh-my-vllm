"""Verify the accepted comparison before any formal timing."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def verify_reference(root=ROOT):
    manifest = json.loads(
        (root / "development/kernels/tilelang-reference.json").read_text()
    )
    expected = {
        root / "python/oh_my_vllm/kernels/tilelang_reference" / name: digest
        for name, digest in manifest["files"].items()
    }
    expected[root / "requirements/runtime.txt"] = manifest[
        "runtime_requirements_sha256"
    ]
    for path, digest in expected.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"frozen comparison changed: {path.relative_to(root)}")
    return manifest
