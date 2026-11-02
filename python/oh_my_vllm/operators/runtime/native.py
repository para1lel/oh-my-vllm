import functools
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
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
    "paged": SOURCES / "runtime/cuda/paged_cache.cu",
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


ENVIRONMENT_IDENTITY_FIELDS = (
    "python_version",
    "torch_version",
    "torch_cuda_version",
    "cuda_home",
    "cuda_includes",
    "cuda_compiler",
)
STALE_MESSAGE = "native module is stale; run python -m development.native_build"


def cuda_home():
    return Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()


def cuda_include_directories(cuda_directory=None):
    cuda_directory = cuda_home() if cuda_directory is None else Path(cuda_directory)
    target = cuda_directory / "targets/x86_64-linux/include"
    directories = []
    if (target / "cuda_runtime.h").is_file() and (
        target / "crt/host_config.h"
    ).is_file():
        directories.append(target)
    root = cuda_directory / "include"
    if root not in directories:
        directories.append(root)
    return tuple(directories)


def cuda_include_directory(cuda_directory=None):
    return cuda_include_directories(cuda_directory)[0]


def cuda_compiler(cuda_directory=None):
    cuda_directory = cuda_home() if cuda_directory is None else Path(cuda_directory)
    candidate = cuda_directory / "bin/nvcc"
    if not candidate.is_file():
        raise RuntimeError(f"CUDA compiler nvcc is not available in {cuda_directory}")
    return subprocess.check_output([str(candidate), "--version"], text=True)


def build_environment_identity(cuda_directory=None, compiler=None):
    import torch

    cuda_directory = cuda_home() if cuda_directory is None else Path(cuda_directory)
    cuda_directory = cuda_directory.resolve()
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_home": str(cuda_directory),
        "cuda_includes": [
            str(path) for path in cuda_include_directories(cuda_directory)
        ],
        "cuda_compiler": (
            cuda_compiler(cuda_directory) if compiler is None else compiler
        ),
    }


def validate_environment_identity(actual, expected=None):
    expected = build_environment_identity() if expected is None else expected
    for field in ENVIRONMENT_IDENTITY_FIELDS:
        if actual.get(field) != expected.get(field):
            raise ValueError(f"native environment identity differs for {field}")


def require_current_environment(metadata, expected=None):
    try:
        validate_environment_identity(metadata, expected)
    except ValueError as error:
        raise RuntimeError(STALE_MESSAGE) from error


def manifest_environment(metadata):
    return {field: metadata.get(field) for field in ENVIRONMENT_IDENTITY_FIELDS}


def build_digest(
    fingerprint,
    compiler,
    flags,
    ffi_version,
    cutlass,
    functions,
    vendor=None,
    environment=None,
):
    flags = tuple(flags)
    identity = (compiler, flags[:3], ffi_version)
    digest = hashlib.sha256(fingerprint.encode())
    if cutlass is not None:
        digest.update(cutlass.encode())
    digest.update(repr((identity, flags, tuple(functions))).encode())
    if vendor is not None:
        digest.update(json.dumps(vendor, sort_keys=True).encode())
    if environment is not None:
        digest.update(
            json.dumps(
                {
                    field: environment.get(field)
                    for field in ENVIRONMENT_IDENTITY_FIELDS
                },
                sort_keys=True,
            ).encode()
        )
    return digest.hexdigest()


def source_identity(metadata, vendor=None):
    return build_digest(
        metadata["source_fingerprint"],
        metadata["compiler"],
        metadata["cuda_flags"],
        metadata["ffi_version"],
        metadata["cutlass"],
        metadata["functions"],
        vendor,
        manifest_environment(metadata),
    )


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
    packages = {
        name: importlib.metadata.distribution(name)
        for name in ("nvidia-cublas", "nvidia-cuda-runtime")
    }
    names = (
        "libcublas.so.13",
        "libcublasLt.so.13",
        "libcudart.so.13",
        "cublas_api.h",
        "cublas_v2.h",
        "cublasLt.h",
        "cuda_runtime_api.h",
    )
    paths = {}
    for package in packages.values():
        for name in package.files or ():
            if Path(name).name in names:
                paths[Path(name).name] = Path(package.locate_file(name)).resolve(
                    strict=True
                )
    if set(paths) != set(names):
        raise RuntimeError("the Torch CUDA vendor packages are incomplete")
    images = [
        {"path": str(paths[name]), "sha256": file_sha256(paths[name])}
        for name in names[:3]
    ]
    return {
        "images": images,
        "headers": {str(paths[name]): file_sha256(paths[name]) for name in names[3:]},
        "include_directory": str(paths["cublas_api.h"].parent),
        "ldflags": [
            *(image["path"] for image in images),
            f"-Wl,-rpath,{paths['libcublas.so.13'].parent}",
            "-ldl",
        ],
    }


def hyperconnection_vendor_dependencies():
    package = importlib.metadata.distribution("nvidia-cublas")
    names = (
        "libcublas.so.13",
        "libcublasLt.so.13",
        "cublas_api.h",
        "cublas_v2.h",
        "cublasLt.h",
    )
    paths = {
        Path(name).name: Path(package.locate_file(name)).resolve(strict=True)
        for name in package.files or ()
        if Path(name).name in names
    }
    if set(paths) != set(names):
        raise RuntimeError("the tensor runtime cuBLAS package is incomplete")
    images = [
        {"path": str(paths[name]), "sha256": file_sha256(paths[name])}
        for name in names[:2]
    ]
    return {
        "version": package.version,
        "images": images,
        "headers": {str(paths[name]): file_sha256(paths[name]) for name in names[2:]},
        "include_directory": str(paths["cublas_v2.h"].parent),
        "ldflags": [
            *(image["path"] for image in images),
            f"-Wl,-rpath,{paths[names[0]].parent}",
            "-ldl",
        ],
    }


def collectives_vendor_dependencies():
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    nccl_package = importlib.metadata.distribution("nvidia-nccl-cu13")
    runtime_package = importlib.metadata.distribution("nvidia-cuda-runtime")
    package_files = {
        Path(path).name: Path(nccl_package.locate_file(path)).resolve(strict=True)
        for path in nccl_package.files or ()
        if Path(path).name in ("libnccl.so.2", "nccl.h")
    }
    include_value = os.environ.get("OH_MY_VLLM_NCCL_INCLUDE")
    include = (
        Path(include_value).resolve()
        if include_value
        else package_files["nccl.h"].parent
    )
    host_compiler = Path(shutil.which("c++") or "c++").resolve(strict=True)
    library = os.environ.get("OH_MY_VLLM_NCCL_LIBRARY")
    if library is None:
        library = package_files["libnccl.so.2"]
    nccl = Path(library).resolve(strict=True)
    runtime = Path(
        runtime_package.locate_file("nvidia/cu13/lib/libcudart.so.13")
    ).resolve(strict=True)
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
        (cuda / "bin/nvcc").resolve(strict=True),
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


def verify_hyperconnection_vendor_images(module, vendor):
    paths = tuple(str(path) for path in module.hc_images())
    verify_vendor_images(paths, vendor["images"], "HC")


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
        raise RuntimeError(STALE_MESSAGE)
    if metadata["ffi_version"] != importlib.metadata.version("apache-tvm-ffi"):
        raise RuntimeError("native module uses a different FFI version")
    require_current_environment(metadata)
    vendor = None
    if name == "vision":
        vendor = vision_vendor_dependencies()
    elif name == "collectives":
        vendor = collectives_vendor_dependencies()
    elif name == "hyperconnection":
        vendor = hyperconnection_vendor_dependencies()
    if vendor is not None and metadata.get(f"{name}_vendor") != vendor:
        raise RuntimeError(f"native {name} vendor dependencies are stale")
    if source_identity(metadata, vendor) != metadata["source_id"]:
        raise RuntimeError(
            f"native {name} is stale; run python -m development.native_build"
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
    elif name == "hyperconnection":
        verify_hyperconnection_vendor_images(module, vendor)
    return module, metadata["source_id"], library
