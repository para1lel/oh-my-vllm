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


def vendor_dependencies(image_names, header_names):
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    directory = (cuda / "lib64").resolve()
    images = []
    for name in image_names:
        path = (directory / name).resolve(strict=True)
        images.append({"path": str(path), "sha256": file_sha256(path)})
    headers = {}
    for name in header_names:
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


def vision_vendor_dependencies():
    return vendor_dependencies(
        ("libcublas.so", "libcublasLt.so", "libcudart.so"),
        ("cublas_api.h", "cublas_v2.h", "cublasLt.h", "cuda_runtime_api.h"),
    )


def ffn_vendor_dependencies():
    return vendor_dependencies(
        ("libcublasLt.so", "libcudart.so"),
        ("cublas_api.h", "cublasLt.h", "cuda_runtime_api.h"),
    )


def gdn_vendor_dependencies():
    return ffn_vendor_dependencies()


def fa_vendor_dependencies():
    return ffn_vendor_dependencies()


def verify_vendor_images(paths, images, name):
    if len(paths) != len(images):
        raise RuntimeError(f"the loaded {name} vendor image set differs")
    for actual, expected in zip(paths, images, strict=True):
        path = Path(actual).resolve(strict=True)
        if str(path) != expected["path"] or file_sha256(path) != expected["sha256"]:
            raise RuntimeError(f"a loaded {name} vendor image differs from its build")


def verify_vision_vendor_images(module, vendor):
    paths = tuple(str(path) for path in module.vision_classic_images())
    verify_vendor_images(paths, vendor["images"], "Vision")


def verify_ffn_vendor_images(module, vendor):
    paths = tuple(str(path) for path in module.ffn_mx_images())
    images = [vendor["images"][0]] * 3 + [vendor["images"][1]]
    verify_vendor_images(paths, images, "FFN")


def verify_gdn_vendor_images(module, vendor):
    paths = tuple(str(path) for path in module.gdn_mx_images())
    images = [vendor["images"][0]] * 3 + [vendor["images"][1]]
    verify_vendor_images(paths, images, "GDN")


def verify_fa_vendor_images(module, vendor):
    paths = tuple(str(path) for path in module.fa_mx_images())
    images = [vendor["images"][0]] * 3 + [vendor["images"][1]]
    verify_vendor_images(paths, images, "FA")


@functools.cache
def load(name, functions):
    manifest = cache_directory() / f"{name}.json"
    if not manifest.is_file():
        raise RuntimeError(
            "native module is missing; run python -m development.native_build"
        )
    metadata = json.loads(manifest.read_text())
    if metadata["source_fingerprint"] != source_fingerprint(name) or metadata[
        "functions"
    ] != list(functions):
        raise RuntimeError(
            "native module is stale; run python -m development.native_build"
        )
    if metadata["ffi_version"] != importlib.metadata.version("apache-tvm-ffi"):
        raise RuntimeError("native module uses a different FFI version")
    vendor = None
    if name in ("vision", "ffn", "gdn", "fa"):
        vendor = (
            vision_vendor_dependencies()
            if name == "vision"
            else ffn_vendor_dependencies()
            if name == "ffn"
            else gdn_vendor_dependencies()
            if name == "gdn"
            else fa_vendor_dependencies()
        )
        key = {
            "vision": "vision_vendor",
            "ffn": "ffn_mx_vendor",
            "gdn": "gdn_mx_vendor",
            "fa": "fa_mx_vendor",
        }[name]
        if metadata.get(key) != vendor:
            raise RuntimeError(f"native {name} vendor dependencies are stale")
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
            raise RuntimeError(
                f"native {name} build identity differs from its manifest"
            )
    library = Path(metadata["library"])
    with library.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != metadata["binary_id"]:
            raise RuntimeError(
                "native module binary identity differs from its manifest"
            )
    from tvm_ffi import load_module

    module = load_module(str(library))
    if vendor is not None:
        verify = (
            verify_vision_vendor_images
            if name == "vision"
            else verify_ffn_vendor_images
            if name == "ffn"
            else verify_gdn_vendor_images
            if name == "gdn"
            else verify_fa_vendor_images
        )
        verify(module, vendor)
    return module, metadata["source_id"], library
