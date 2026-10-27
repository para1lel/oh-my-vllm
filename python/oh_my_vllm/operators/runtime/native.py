import functools
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path

SOURCES = Path(__file__).resolve().parents[1]
MODULE_SOURCES = {
    name: SOURCES / family / "cuda" / (name + ".cu")
    for name, family in (
        ("device", "runtime"),
        ("activation", "activation"),
        ("sample", "output"),
        ("fa", "fa"),
        ("gdn", "gdn"),
        ("hyperconnection", "hyperconnection"),
        ("ffn", "ffn"),
        ("mtp", "mtp"),
        ("projection", "projection"),
        ("routing", "routing"),
        ("sparse_attention", "sparse_attention"),
        ("transform", "transform"),
        ("vision", "vision"),
    )
}
OPTIONAL_MODULE_SOURCES = {
    "collectives": SOURCES / "runtime/cuda/collectives.cu",
}


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


def source_digest(source, root=None):
    source = Path(source).resolve()
    root = SOURCES if root is None else Path(root).resolve()
    digest = hashlib.sha256()
    for file in source_files(source):
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def source_fingerprint(name):
    sources = MODULE_SOURCES | OPTIONAL_MODULE_SOURCES
    return source_digest(sources[name])


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


def collectives_vendor_dependencies():
    import shutil
    import subprocess

    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    include = Path(os.environ.get("OH_MY_VLLM_NCCL_INCLUDE", "/usr/include")).resolve()
    host_compiler = Path(shutil.which("c++") or "c++").resolve(strict=True)
    library = os.environ.get("OH_MY_VLLM_NCCL_LIBRARY")
    if library is None:
        library = subprocess.check_output(
            [str(host_compiler), "-print-file-name=libnccl.so"], text=True
        ).strip()
    nccl = Path(library).resolve(strict=True)
    runtime = (cuda / "lib64/libcudart.so").resolve(strict=True)
    headers = [
        include / "nccl.h",
        *(
            cuda / "include" / name
            for name in (
                "cuda_runtime.h",
                "cuda_runtime_api.h",
                "cuda_fp16.h",
                "cuda_bf16.h",
                "cuda_fp8.h",
            )
        ),
    ]
    compilers = [
        Path(shutil.which("nvcc") or cuda / "bin/nvcc").resolve(strict=True),
        host_compiler,
    ]
    return {
        "images": [
            {"path": str(path), "sha256": file_sha256(path)} for path in (nccl, runtime)
        ],
        "headers": {str(path.resolve()): file_sha256(path) for path in headers},
        "compilers": {str(path): file_sha256(path) for path in compilers},
        "host_compiler": str(host_compiler),
        "include_directory": str(include),
        "ldflags": [
            str(nccl),
            str(runtime),
            f"-Wl,-rpath,{nccl.parent}",
            f"-Wl,-rpath,{runtime.parent}",
            "-ldl",
        ],
    }


def verify_collectives_vendor_images(module, vendor):
    verify_vendor_images(
        tuple(str(path) for path in module.collective_images()),
        vendor["images"],
        "NCCL",
    )
    header, runtime = module.collective_versions()
    if header != runtime:
        raise RuntimeError("NCCL header and runtime versions differ")


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
    if name == "vision":
        vendor = vision_vendor_dependencies()
    elif name == "collectives":
        vendor = collectives_vendor_dependencies()
    if vendor is not None and metadata.get(f"{name}_vendor") != vendor:
        raise RuntimeError(f"native {name} vendor dependencies are stale")
    if name in (
        "activation",
        "fa",
        "ffn",
        "gdn",
        "hyperconnection",
        "projection",
        "routing",
        "sparse_attention",
        "transform",
        "vision",
        *OPTIONAL_MODULE_SOURCES,
    ):
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
    if name == "vision":
        verify_vision_vendor_images(module, vendor)
    elif name == "collectives":
        verify_collectives_vendor_images(module, vendor)
    return module, metadata["source_id"], library
