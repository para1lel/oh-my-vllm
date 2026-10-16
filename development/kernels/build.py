import functools
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path

from oh_my_vllm.kernels.native import SOURCES, cache_directory, source_digest


@functools.cache
def build(name, functions):
    from tvm_ffi import load_module
    from tvm_ffi.cpp import build_inline

    if os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", "10.0a") != "10.0a":
        raise ValueError("native operators require the B200 architecture 10.0a")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        nvcc = str(Path(cuda) / "bin/nvcc")
    flags = ("-O3", "--generate-code=arch=compute_100a,code=sm_100a", "-lineinfo")
    source = SOURCES / f"{name}.cu"
    source_fingerprint = source_digest(source)
    digest = hashlib.sha256(source_fingerprint.encode())
    compiler = subprocess.check_output([nvcc, "--version"]).decode()
    identity = (compiler, flags, importlib.metadata.version("apache-tvm-ffi"))
    include_paths = [str(source.parent), str(SOURCES)]
    commit = None
    if name != "device":
        cutlass = Path(os.environ["OH_MY_VLLM_CUTLASS"])
        commit = (
            subprocess.check_output(["git", "-C", str(cutlass), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        if commit != "098de2a652cf8f00fd70b2df54051c7eccbb855a":
            raise ValueError("CUTLASS must use the pinned v4.8.0 commit")
        if subprocess.check_output(
            ["git", "-C", str(cutlass), "status", "--porcelain"]
        ):
            raise ValueError("CUTLASS checkout must be clean")
        digest.update(commit.encode())
        flags += ("--expt-relaxed-constexpr", "-static-global-template-stub=false")
        include_paths += [str(cutlass / "include"), str(cutlass / "tools/util/include")]
        if name == "vision":
            include_paths.append(str(cutlass / "examples/77_blackwell_fmha"))
    digest.update(repr((identity, flags, functions)).encode())
    source_id = digest.hexdigest()
    cache = cache_directory()
    directory = cache / source_id
    directory.mkdir(parents=True, exist_ok=True)
    library = build_inline(
        f"oh_my_vllm_{name}",
        cuda_sources=f'#include "{name}.cu"',
        functions=functions,
        extra_cuda_cflags=flags,
        extra_include_paths=include_paths,
        build_directory=str(directory),
        backend="cuda",
    )
    with Path(library).open("rb") as file:
        binary_id = hashlib.file_digest(file, "sha256").hexdigest()
    binary_directory = cache / "binaries" / source_id / binary_id
    binary_directory.mkdir(parents=True, exist_ok=True)
    immutable_library = binary_directory / Path(library).name
    if not immutable_library.exists():
        temporary_library = binary_directory / f"library.{os.getpid()}.tmp"
        shutil.copyfile(library, temporary_library)
        temporary_library.replace(immutable_library)
    metadata = {
        "source_id": source_id,
        "source_fingerprint": source_fingerprint,
        "binary_id": binary_id,
        "library": str(immutable_library.resolve()),
        "functions": list(functions),
        "compiler": compiler,
        "cuda_flags": flags,
        "ffi_version": importlib.metadata.version("apache-tvm-ffi"),
        "cutlass": commit,
    }
    encoded = json.dumps(metadata, indent=2) + "\n"
    immutable_manifest = binary_directory / "manifest.json"
    if immutable_manifest.exists() and immutable_manifest.read_text() != encoded:
        raise ValueError("immutable binary manifest has changed")
    temporary_manifest = binary_directory / f"manifest.{os.getpid()}.tmp"
    temporary_manifest.write_text(encoded)
    temporary_manifest.replace(immutable_manifest)
    temporary = cache / f"{name}.{os.getpid()}.json"
    temporary.write_text(encoded)
    temporary.replace(cache / f"{name}.json")
    return load_module(str(immutable_library)), source_id, immutable_library
