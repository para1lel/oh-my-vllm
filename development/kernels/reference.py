"""Verify the accepted comparison before any formal timing."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_PRESERVED_IDS_SHA256 = (
    "2f2d818f81380383832af55ce0f2d6124310727b7fc030afbb8728e9ebf30d9b"
)


def verify_reference(root=ROOT):
    frozen_manifest = root / "development/kernels/tilelang-reference.json"
    manifest = json.loads(frozen_manifest.read_text())
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
    supplemental_path = root / "development/kernels/dspark-reference.json"
    supplemental = json.loads(supplemental_path.read_text())
    if supplemental.get("schema") != 1 or set(supplemental.get("files", {})) != {
        "dspark_tilelang_reference.py"
    }:
        raise ValueError("invalid supplemental DSpark comparison manifest")
    preserved = supplemental.get("preserved_original_case_ids", [])
    encoded = json.dumps(preserved, separators=(",", ":")).encode()
    if (
        len(preserved) != 147
        or hashlib.sha256(encoded).hexdigest() != _PRESERVED_IDS_SHA256
    ):
        raise ValueError("supplemental comparison must preserve the original 147 cases")
    from .cases import cases

    if not set(preserved) <= {case["id"] for case in cases()}:
        raise ValueError("formal case matrix removed an original frozen gate")
    if (
        supplemental.get("frozen_reference_manifest_sha256")
        != hashlib.sha256(frozen_manifest.read_bytes()).hexdigest()
    ):
        raise ValueError("supplemental comparison has a different frozen predecessor")
    for name, digest in supplemental["files"].items():
        path = root / "python/oh_my_vllm/kernels" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(
                f"supplemental comparison changed: {path.relative_to(root)}"
            )
    return manifest | {"dspark_supplemental": supplemental}


def source_hashes(root=ROOT):
    paths = [
        *root.glob("python/oh_my_vllm/kernels/**/*.py"),
        *root.glob("python/oh_my_vllm/kernels/**/*.cu"),
        *root.glob("python/oh_my_vllm/kernels/**/*.cuh"),
        root / "python/oh_my_vllm/models/qwen.py",
        root / "python/oh_my_vllm/models/dspark.py",
        *root.glob("python/oh_my_vllm/ir/**/*.py"),
        *root.glob("development/kernels/*.py"),
        root / "development/kernels/tilelang-reference.json",
        root / "development/kernels/dspark-reference.json",
        root / "requirements/runtime.txt",
        root / "benchmarks/kernels.py",
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
