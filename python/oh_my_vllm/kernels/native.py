import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

SOURCES = Path(__file__).with_name("cuda_backend")


def source_fingerprint(name):
    digest = hashlib.sha256()
    for file in [*sorted(SOURCES.glob("*.cuh")), SOURCES / f"{name}.cu"]:
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def cache_directory():
    return Path(
        os.environ.get("OH_MY_VLLM_KERNEL_CACHE", "~/.cache/oh-my-vllm/kernels")
    ).expanduser()


@functools.cache
def load(name, functions):
    from tvm_ffi import load_module

    manifest = cache_directory() / f"{name}.json"
    if not manifest.is_file():
        raise RuntimeError(
            "native module is missing; run python -m development.kernels build"
        )
    metadata = json.loads(manifest.read_text())
    if metadata["source_fingerprint"] != source_fingerprint(name) or metadata[
        "functions"
    ] != list(functions):
        raise RuntimeError(
            "native module is stale; run python -m development.kernels build"
        )
    if metadata["ffi_version"] != importlib.metadata.version("apache-tvm-ffi"):
        raise RuntimeError("native module uses a different FFI version")
    library = Path(metadata["library"])
    with library.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != metadata["binary_id"]:
            raise RuntimeError(
                "native module binary identity differs from its manifest"
            )
    return load_module(str(library)), metadata["source_id"], library
