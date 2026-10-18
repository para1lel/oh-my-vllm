import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from oh_my_vllm.kernels.native import SOURCES, source_files

from .providers import MODULES, RUNTIME_MODULES

ROOT = Path(__file__).resolve().parents[2]
SHARED = (
    "__main__",
    "acceptance",
    "provenance",
    "providers",
    "build",
    "checkpoint",
    "device",
    "ffn",
    "reference",
    "roofline",
    "work",
    "workload",
)


def case_id(shape):
    return hashlib.sha256(
        json.dumps(asdict(shape), sort_keys=True).encode()
    ).hexdigest()[:16]


def python_sources(modules):
    files = {ROOT / "development/kernels" / f"{name}.py" for name in SHARED}
    files.update(
        ROOT / "development" / name
        for name in ("peaks.py", "peak_measurement.py", "peak_kernels.cu")
    )
    for name in modules:
        files.update(
            (
                ROOT / "development/kernels" / f"{name}.py",
                ROOT / "python/oh_my_vllm/kernels" / f"{name}.py",
            )
        )
    files.update(
        ROOT / "python/oh_my_vllm/kernels" / f"{name}.py"
        for name in ("native", "specification")
    )
    files.update(
        ROOT / "python/oh_my_vllm/worker" / f"{name}.py"
        for name in ("checkpoint", "device")
    )
    return files


def digest_sources(digest, files):
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())


def suite_identity():
    files = python_sources(
        {name for names in RUNTIME_MODULES.values() for name in names}
    )
    for name in MODULES:
        source = SOURCES / f"{name}.cu"
        files.update(source_files(source))
    digest = hashlib.sha256()
    digest_sources(digest, files)
    return digest.hexdigest()


def source_identity(native_id, binaries, context, operator):
    digest = hashlib.sha256(native_id.encode())
    digest_sources(digest, python_sources(RUNTIME_MODULES[operator]))
    binary_ids = {}
    for binary in binaries:
        with binary.open("rb") as file:
            binary_ids[str(binary)] = hashlib.file_digest(file, "sha256").hexdigest()
    digest.update(json.dumps(binary_ids, sort_keys=True).encode())
    digest.update(json.dumps(context, sort_keys=True).encode())
    return digest.hexdigest(), binary_ids
