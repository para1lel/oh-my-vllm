import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path

from oh_my_vllm.operators.fa.api import FUNCTIONS as FA_FUNCTIONS
from oh_my_vllm.operators.ffn.api import FUNCTIONS as FFN_FUNCTIONS
from oh_my_vllm.operators.gdn.api import FUNCTIONS as GDN_FUNCTIONS
from oh_my_vllm.operators.mtp.api import FUNCTIONS as MTP_FUNCTIONS
from oh_my_vllm.operators.output.api import FUNCTIONS as SAMPLE_FUNCTIONS
from oh_my_vllm.operators.runtime.device import FUNCTIONS as DEVICE_FUNCTIONS
from oh_my_vllm.operators.runtime.native import (
    MODULE_SOURCES,
    OPTIONAL_MODULE_SOURCES,
    SOURCES,
    build_digest,
    cache_directory,
    collectives_vendor_dependencies,
    source_digest,
    verify_collectives_vendor_images,
    verify_vision_vendor_images,
    vision_vendor_dependencies,
)
from oh_my_vllm.operators.vision.api import FUNCTIONS as VISION_FUNCTIONS

MODULES = {
    "device": DEVICE_FUNCTIONS,
    "ffn": FFN_FUNCTIONS,
    "fa": FA_FUNCTIONS,
    "gdn": GDN_FUNCTIONS,
    "mtp": MTP_FUNCTIONS,
    "sample": SAMPLE_FUNCTIONS,
    "vision": VISION_FUNCTIONS,
}


def optional_modules():
    from oh_my_vllm.operators.activation.api import FUNCTIONS as ACTIVATION_FUNCTIONS
    from oh_my_vllm.operators.engram.api import FUNCTIONS as ENGRAM_FUNCTIONS
    from oh_my_vllm.operators.hyperconnection.api import FUNCTIONS as HC_FUNCTIONS
    from oh_my_vllm.operators.projection.api import FUNCTIONS as PROJECTION_FUNCTIONS
    from oh_my_vllm.operators.quantization.api import (
        FUNCTIONS as QUANTIZATION_FUNCTIONS,
    )
    from oh_my_vllm.operators.routing.api import FUNCTIONS as ROUTING_FUNCTIONS
    from oh_my_vllm.operators.runtime.collectives import FUNCTIONS
    from oh_my_vllm.operators.transform.api import FUNCTIONS as TRANSFORM_FUNCTIONS

    return {
        "collectives": FUNCTIONS,
        "quantization": QUANTIZATION_FUNCTIONS,
        "engram": ENGRAM_FUNCTIONS,
        "hyperconnection": HC_FUNCTIONS,
        "routing": ROUTING_FUNCTIONS,
        "projection": PROJECTION_FUNCTIONS,
        "transform": TRANSFORM_FUNCTIONS,
        "activation": ACTIVATION_FUNCTIONS,
    }


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
    source = (MODULE_SOURCES | OPTIONAL_MODULE_SOURCES)[name]
    source_fingerprint = source_digest(source)
    compiler = subprocess.check_output([nvcc, "--version"]).decode()
    include_paths = [str(source.parent), str(SOURCES)]
    commit = None
    if (name in MODULES and name != "device") or name == "projection":
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
        flags += ("--expt-relaxed-constexpr", "-static-global-template-stub=false")
        include_paths += [str(cutlass / "include"), str(cutlass / "tools/util/include")]
        if name == "vision":
            include_paths.append(str(cutlass / "examples/77_blackwell_fmha"))
    vendor = vision_vendor_dependencies() if name == "vision" else None
    if name == "collectives":
        vendor = collectives_vendor_dependencies()
        flags += ("--compiler-bindir", vendor["host_compiler"])
    build_options = {}
    if vendor is not None:
        include_paths.append(vendor["include_directory"])
        build_options["extra_ldflags"] = vendor["ldflags"]
    source_id = build_digest(
        source_fingerprint,
        compiler,
        flags,
        importlib.metadata.version("apache-tvm-ffi"),
        commit,
        functions,
        vendor,
    )
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
        **build_options,
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
    if vendor is not None:
        metadata[f"{name}_vendor"] = vendor
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
    module = load_module(str(immutable_library))
    if name == "vision":
        verify_vision_vendor_images(module, vendor)
    elif name == "collectives":
        verify_collectives_vendor_images(module, vendor)
    return module, source_id, immutable_library


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module", action="append", choices=(*MODULES, *OPTIONAL_MODULE_SOURCES)
    )
    options = parser.parse_args()
    selected = (
        MODULES
        if options.module is None
        else {name: (MODULES | optional_modules())[name] for name in options.module}
    )
    for name, functions in selected.items():
        _, identity, _ = build(name, functions)
        print(f"{name}: {identity}")


if __name__ == "__main__":
    main()
