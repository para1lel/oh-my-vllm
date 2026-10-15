import functools
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path

from oh_my_vllm.kernels.native import SOURCES, cache_directory


@functools.cache
def build(name, functions):
    from tvm_ffi import load_module
    from tvm_ffi.cpp import build_inline

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        nvcc = str(Path(cuda) / "bin/nvcc")
    flags = ("-O3", "--generate-code=arch=compute_100a,code=sm_100a", "-lineinfo")
    source = (Path(__file__).parent if name == "device" else SOURCES) / f"{name}.cu"
    files = [*sorted(SOURCES.glob("*.cuh")), source]
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    source_fingerprint = digest.hexdigest()
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
    metadata = {
        "source_id": source_id,
        "source_fingerprint": source_fingerprint,
        "binary_id": binary_id,
        "library": str(Path(library).resolve()),
        "functions": list(functions),
        "compiler": compiler,
        "cuda_flags": flags,
        "ffi_version": importlib.metadata.version("apache-tvm-ffi"),
        "cutlass": commit,
    }
    temporary = cache / f"{name}.{os.getpid()}.json"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(cache / f"{name}.json")
    return load_module(library), source_id, Path(library)
