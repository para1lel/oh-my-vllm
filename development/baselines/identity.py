import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PINS = {
    "flashinfer": "deca0e2959d521fe37efffdb6074cd357e836042",
    "flash_qla": "5538c35f0df33ef804eacd90c77d9d9087c1d4e4",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def files_identity(paths):
    return {
        str(Path(path).resolve()): sha256(path) for path in sorted(map(Path, paths))
    }


def installed_content_identity(distribution):
    digest = hashlib.sha256()
    count, size = 0, 0
    for entry in sorted(distribution.files or (), key=str):
        path = Path(distribution.locate_file(entry))
        if not path.is_file() or not (
            ".so" in path.suffixes
            or path.suffix
            in {
                ".py",
                ".so",
                ".h",
                ".hpp",
                ".cuh",
                ".cu",
                ".cpp",
                ".json",
                ".csv",
                ".ptx",
                ".cubin",
                ".o",
                ".fatbin",
            }
        ):
            continue
        digest.update(str(entry).encode())
        digest.update(sha256(path).encode())
        count += 1
        size += path.stat().st_size
    return {"sha256": digest.hexdigest(), "files": count, "bytes": size}


def header_content_identity():
    root = Path(os.environ["FLASHINFER_WORKSPACE_BASE"]).parent / "headers"
    digest = hashlib.sha256()
    files, size = 0, 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(sha256(path).encode())
            files += 1
            size += path.stat().st_size
    if not files:
        raise ValueError("public source headers are missing")
    return {"sha256": digest.hexdigest(), "files": files, "bytes": size}


def environment_identity(gpu):
    import torch

    packages = {}
    for name in (
        "torch",
        "triton",
        "tilelang",
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-base",
        "nvidia-cutlass-dsl-libs-core",
        "nvidia-cutlass-dsl-libs-cu13",
        "nvidia-cutlass-dsl-libs-cu12",
        "nvidia-cudnn-cu13",
        "nvidia-cudnn-frontend",
        "nvidia-cuda-runtime",
        "nvidia-cublas",
        "nvidia-cuda-nvrtc",
        "cuda-bindings",
        "cuda-core",
        "flashinfer-python",
        "flash-qla",
        "numpy",
        "apache-tvm-ffi",
    ):
        distribution = importlib.metadata.distribution(name)
        direct = distribution.read_text("direct_url.json")
        packages[name] = {
            "version": distribution.version,
            "direct_url": json.loads(direct) if direct else None,
            "record_sha256": hashlib.sha256(
                (distribution.read_text("RECORD") or "").encode()
            ).hexdigest(),
            "installed_content": installed_content_identity(distribution),
        }
    source_files = [
        path
        for path in (ROOT / "development/baselines").iterdir()
        if path.suffix in {".py", ".json", ".sh"}
        and path.name not in {"fa_core.py", "gdn_native.py"}
    ]
    source_files.extend(
        ROOT / path
        for path in (
            "requirements/baselines.in",
            "requirements/baselines.txt",
            "development/kernels/workload.py",
            "development/kernels/reference.py",
            "development/serving/trace.py",
            "python/oh_my_vllm/kernels/specification.py",
        )
    )
    device = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=uuid,name,driver_version,pstate,clocks.sm,clocks.mem,power.limit",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return {
        "gpu_uuid": gpu,
        "device_and_clocks": device,
        "compute_capability": torch.cuda.get_device_capability(),
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "packages": packages,
        "source_pins": SOURCE_PINS,
        "header_pins": json.loads(Path(__file__).with_name("headers.json").read_text()),
        "header_contents": header_content_identity(),
        "compiler": subprocess.check_output(
            [str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"), "--version"],
            text=True,
        ),
        "public_compiler_flags": {
            name: os.environ.get(name)
            for name in (
                "FLASHINFER_EXTRA_CFLAGS",
                "FLASHINFER_EXTRA_CUDAFLAGS",
                "TVM_FFI_CUDA_ARCH_LIST",
                "CUDNN_FRONTEND_CUDART_LIB_NAME",
                "CUTE_DSL_KEEP",
                "CUTE_DSL_NO_CACHE",
                "CUDNN_FRONTEND_DISABLE_COMPILED_CACHE",
            )
        },
        "cache_paths": {
            name: os.environ.get(name)
            for name in (
                "OH_MY_VLLM_KERNEL_CACHE",
                "FLASHINFER_WORKSPACE_BASE",
                "TRITON_CACHE_DIR",
                "TILELANG_CACHE_DIR",
                "CUTE_DSL_CACHE_DIR",
                "CUTE_DSL_DUMP_DIR",
                "OH_MY_VLLM_PUBLIC_CUBINS",
                "CUDNN_FRONTEND_COMPILED_CACHE",
                "CUDA_CACHE_PATH",
                "TORCHINDUCTOR_CACHE_DIR",
                "XDG_CACHE_HOME",
            )
        },
        "sources": files_identity(source_files),
    }


def verify_source_pins():
    for distribution_name, source in (
        ("flashinfer-python", "flashinfer"),
        ("flash-qla", "flash_qla"),
    ):
        distribution = importlib.metadata.distribution(distribution_name)
        direct = distribution.read_text("direct_url.json")
        if not direct or SOURCE_PINS[source] not in direct:
            raise ValueError(
                f"{distribution_name} was not installed from its pinned source"
            )


def input_identity(inputs):
    import torch

    result = {}
    for name, tensor in inputs.tensors.items():
        byte_array = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
        result[name] = {
            "shape": tuple(tensor.shape),
            "strides": tuple(tensor.stride()),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(byte_array).hexdigest(),
        }
    return result


def provider_identity(prepared):
    paths = prepared.metadata.get("source_files", ())
    binaries = prepared.metadata.get("binaries", ())
    if not paths or not binaries:
        raise ValueError(
            "own provider must identify its source_files and compiled binaries"
        )
    return {"sources": files_identity(paths), "binaries": files_identity(binaries)}


def save_cute_binaries():
    module = sys.modules.get(
        "cudnn.linear_attention.frost.kernel.gdn_warmup_forward_f16"
    )
    binaries = (
        [compiled.__cubin__ for compiled in module.warmup_forward_cache.values()]
        if module is not None
        else []
    )
    dump_directory = Path(os.environ["CUTE_DSL_DUMP_DIR"])
    binaries.extend(path.read_bytes() for path in dump_directory.glob("*.cubin"))
    directory = Path(os.environ["OH_MY_VLLM_PUBLIC_CUBINS"])
    directory.mkdir(parents=True, exist_ok=True)
    for binary in binaries:
        if not isinstance(binary, bytes) or not binary:
            raise ValueError("live cuDNN GDN kernel has no retained CUBIN")
        digest = hashlib.sha256(binary).hexdigest()
        destination = directory / f"{digest}.cubin"
        if destination.exists():
            if sha256(destination) != digest:
                raise ValueError("retained cuDNN CUBIN contents changed")
            continue
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
            temporary.write(binary)
            pending = Path(temporary.name)
        pending.replace(destination)


def public_binary_identity():
    save_cute_binaries()
    roots = {
        name: os.environ.get(name)
        for name in (
            "FLASHINFER_WORKSPACE_BASE",
            "TRITON_CACHE_DIR",
            "TILELANG_CACHE_DIR",
            "CUTE_DSL_CACHE_DIR",
            "OH_MY_VLLM_PUBLIC_CUBINS",
            "CUDNN_FRONTEND_COMPILED_CACHE",
            "CUDA_CACHE_PATH",
            "TORCHINDUCTOR_CACHE_DIR",
            "XDG_CACHE_HOME",
        )
    }
    files = set()
    for name, root in roots.items():
        if root:
            files.update(
                path
                for path in Path(root).rglob("*")
                if path.is_file()
                and (
                    ".so" in path.suffixes
                    or path.suffix in {".o", ".cubin", ".ptx", ".fatbin"}
                    or (
                        name == "CUDA_CACHE_PATH" and path.name not in {"lock", "index"}
                    )
                )
            )
    return files_identity(files)
