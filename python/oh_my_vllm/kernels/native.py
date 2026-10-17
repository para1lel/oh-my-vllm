import functools
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path

SOURCES = Path(__file__).with_name("cuda_backend")


def source_files(source):
    pending, visited = [Path(source).resolve()], set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for include in re.findall(r'^\s*#include\s+"([^"]+)"', path.read_text(), re.M):
            pending.append((path.parent / include).resolve())
    return sorted(visited)


def source_digest(source):
    source = Path(source).resolve()
    digest = hashlib.sha256()
    for file in source_files(source):
        digest.update(os.path.relpath(file, source.parent).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def source_fingerprint(name):
    return source_digest(SOURCES / f"{name}.cu")


def cache_directory():
    return Path(
        os.environ.get("OH_MY_VLLM_KERNEL_CACHE", "~/.cache/oh-my-vllm/kernels")
    ).expanduser()


def file_sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def build_digest(
    fingerprint, compiler, flags, ffi_version, cutlass, functions, vendor=None
):
    flags = tuple(flags)
    identity = (compiler, flags[:3], ffi_version)
    digest = hashlib.sha256(fingerprint.encode())
    if cutlass is not None:
        digest.update(cutlass.encode())
    digest.update(repr((identity, flags, tuple(functions))).encode())
    if vendor is not None:
        digest.update(json.dumps(vendor, sort_keys=True).encode())
    return digest.hexdigest()


def vision_vendor_dependencies():
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    directory = (cuda / "lib64").resolve()
    images = []
    for name in ("libcublas.so", "libcublasLt.so", "libcudart.so"):
        path = (directory / name).resolve(strict=True)
        images.append({"path": str(path), "sha256": file_sha256(path)})
    headers = {}
    for name in ("cublas_api.h", "cublas_v2.h", "cublasLt.h", "cuda_runtime_api.h"):
        path = (cuda / "include" / name).resolve(strict=True)
        headers[str(path)] = file_sha256(path)
    return {
        "images": images,
        "headers": headers,
        "include_directory": str((cuda / "include").resolve()),
        "ldflags": [
            *(image["path"] for image in images),
            f"-Wl,-rpath,{directory}",
            "-ldl",
        ],
    }


def verify_vision_vendor_images(module, vendor):
    paths = tuple(str(path) for path in module.vision_classic_images())
    if len(paths) != len(vendor["images"]):
        raise RuntimeError("the loaded Vision vendor image set differs")
    for actual, expected in zip(paths, vendor["images"], strict=True):
        path = Path(actual).resolve(strict=True)
        if str(path) != expected["path"] or file_sha256(path) != expected["sha256"]:
            raise RuntimeError("a loaded Vision vendor image differs from its build")


@functools.cache
def load(name, functions):
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
    vendor = None
    if name == "vision":
        vendor = vision_vendor_dependencies()
        if metadata.get("vision_vendor") != vendor:
            raise RuntimeError("native Vision vendor dependencies are stale")
        identity = build_digest(
            metadata["source_fingerprint"],
            metadata["compiler"],
            metadata["cuda_flags"],
            metadata["ffi_version"],
            metadata["cutlass"],
            metadata["functions"],
            vendor,
        )
        if identity != metadata["source_id"]:
            raise RuntimeError("native Vision build identity differs from its manifest")
    library = Path(metadata["library"])
    with library.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != metadata["binary_id"]:
            raise RuntimeError(
                "native module binary identity differs from its manifest"
            )
    from tvm_ffi import load_module

    module = load_module(str(library))
    if vendor is not None:
        verify_vision_vendor_images(module, vendor)
    return module, metadata["source_id"], library
