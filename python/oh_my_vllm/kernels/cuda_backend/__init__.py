"""B200 CUDA implementations, compiled lazily; executed on the caller's CUDA stream."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path

_FUNCTIONS = (
    "variant_launch_count",
    "quantize",
    "silu_mul",
    "gates",
    "rms",
    "add_rms",
    "rms_rope",
    "prepare_attention",
    "rope",
    "normalize_qk",
    "recurrent",
    "append",
    "convolution",
    "attention_partial",
    "attention_merge",
    "dspark_norm_rope",
    "dspark_rms_norm",
    "dspark_append",
    "dspark_attention_partial",
    "dspark_attention_merge",
)
_VARIANT_OPERATIONS = (
    "norm",
    "add_norm",
    "gated_norm",
    "qk",
    "recurrent",
    "append",
    "convolution",
)
_BASE_CUDA_FLAGS = ("-O3", "--generate-code=arch=compute_100a,code=sm_100a")
_MANIFEST = "oh_my_vllm_cuda.provenance.json"
_loaded_so_path: Path | None = None
_loaded_so_sha256: str | None = None
_loaded_nvcc_path: Path | None = None
_loaded_nvcc_version: str | None = None
_loaded_compiler_source: str | None = None


def _nvcc_identity() -> tuple[Path | None, str]:
    """Query the compiler selected by TVM FFI, rather than PATH's nvcc."""
    path = None
    try:
        from tvm_ffi.cpp.extension import _find_cuda_home

        path = (Path(_find_cuda_home()) / "bin" / "nvcc").resolve()
        version = subprocess.check_output([str(path), "--version"], text=True)
        return path, version.strip().splitlines()[-1]
    except Exception as exc:
        return path, f"nvcc-unavailable: {exc}"


def _cache_root() -> Path:
    return (
        Path(os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi"))
        .expanduser()
        .resolve()
    )


def _in_cuda_cache(so_path: Path) -> bool:
    root = _cache_root()
    return so_path.parent.parent == root and so_path.parent.name.startswith(
        "oh_my_vllm_cuda_"
    )


def _effective_cuda_target() -> str:
    """Use the same target selector as TVM FFI's CUDA ninja rule."""
    from tvm_ffi.cpp.extension import _get_cuda_target

    configured = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    if configured != "10.0a":
        raise RuntimeError("CUDA backend requires TVM_FFI_CUDA_ARCH_LIST=10.0a")
    return _get_cuda_target()


def _build_input_digest(cuda_source: str, torch_version: str) -> str:
    """Identify every compiler-independent input to the TVM FFI build."""
    inputs = {
        "cuda_source_sha256": hashlib.sha256(cuda_source.encode()).hexdigest(),
        "functions": _FUNCTIONS,
        "cuda_flags": _BASE_CUDA_FLAGS,
        "sm": "sm_100a",
        "tvm_cuda_target": _effective_cuda_target(),
        "backend": "cuda",
        "cxx": os.environ.get("CXX", "c++"),
        "extra_ldflags": (),
        "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
        "torch": str(torch_version),
        "python_abi": sys.implementation.cache_tag,
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def _write_manifest(
    so_path: Path, digest: str, nvcc_path: Path, version: str, sha: str
):
    record = {
        "schema": 1,
        "build_input_sha256": digest,
        "nvcc_path": str(nvcc_path),
        "nvcc_version": version,
        "so_path": str(so_path),
        "so_sha256": sha,
    }
    path = so_path.parent / _MANIFEST
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=so_path.parent, prefix=".cuda-provenance-", delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(record, output, sort_keys=True)
            output.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _cached_manifest(digest: str, expected_nvcc_path: Path | None) -> dict:
    """Locate one trusted offline build for these exact source/toolchain inputs."""
    root = _cache_root()
    matches = []
    for path in root.glob(f"oh_my_vllm_cuda_*/{_MANIFEST}"):
        if path.is_symlink():
            raise RuntimeError(f"symlinked CUDA build manifest: {path}")
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid CUDA build manifest: {path}") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"invalid CUDA build manifest: {path}")
        if record.get("build_input_sha256") == digest and (
            expected_nvcc_path is None
            or record.get("nvcc_path") == str(expected_nvcc_path)
        ):
            matches.append((path, record))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one matching offline CUDA build manifest; found {len(matches)}"
        )
    path, record = matches[0]
    path_value = record.get("so_path")
    if not isinstance(path_value, str):
        raise RuntimeError("offline CUDA build manifest lacks a shared-library path")
    so_path = Path(path_value).resolve()
    if (
        record.get("schema") != 1
        or not so_path.is_file()
        or so_path.parent != path.parent.resolve()
        or not _in_cuda_cache(so_path)
        or not isinstance(record.get("nvcc_path"), str)
        or not isinstance(record.get("nvcc_version"), str)
        or not record["nvcc_path"]
        or not Path(record["nvcc_path"]).is_absolute()
        or (
            expected_nvcc_path is not None
            and Path(record["nvcc_path"]) != expected_nvcc_path
        )
        or not record["nvcc_version"]
        or record["nvcc_version"].startswith("nvcc-unavailable:")
        or record["nvcc_version"] == "nvcc-changed-during-build"
        or not isinstance(record.get("so_sha256"), str)
    ):
        raise RuntimeError(
            "offline CUDA build manifest is incomplete or outside the cache"
        )
    if hashlib.sha256(so_path.read_bytes()).hexdigest() != record["so_sha256"]:
        raise RuntimeError("offline CUDA shared library hash differs from manifest")
    return record


def _record_loaded(
    so_path: Path, sha: str, nvcc_path: Path, version: str, source: str
) -> None:
    global _loaded_so_path, _loaded_so_sha256, _loaded_nvcc_path
    global _loaded_nvcc_version, _loaded_compiler_source
    _loaded_so_path = so_path
    _loaded_so_sha256 = sha
    _loaded_nvcc_path = nvcc_path
    _loaded_nvcc_version = version
    _loaded_compiler_source = source
    print(
        "CUDA_BUILD_PROVENANCE " + json.dumps(provenance(require_loaded=True)),
        file=sys.stderr,
        flush=True,
    )


def provenance(*, require_loaded: bool = False, require_compiler: bool = False) -> dict:
    """Identify the exact shared library this process loaded, if any."""
    path = _loaded_so_path
    if require_loaded and (path is None or _loaded_so_sha256 is None):
        raise RuntimeError("loaded CUDA module is required")
    if require_compiler and (
        _loaded_nvcc_path is None
        or not _loaded_nvcc_version
        or _loaded_nvcc_version.startswith("nvcc-unavailable:")
        or _loaded_nvcc_version == "nvcc-changed-during-build"
    ):
        raise RuntimeError("CUDA compiler identity is required")
    return {
        "nvcc_path": str(_loaded_nvcc_path) if _loaded_nvcc_path else None,
        "nvcc_version": _loaded_nvcc_version,
        "compiler_identity_source": _loaded_compiler_source,
        "so_path": str(path) if path is not None else None,
        "so_sha256": _loaded_so_sha256,
    }


def variant_launch_counts() -> dict[str, dict[str, int]]:
    """Host variant choices; CUDA Graph replay does not re-enter host dispatch."""
    module = compiled()
    return {
        operation: {
            "fast": module.variant_launch_count(index, True),
            "generic": module.variant_launch_count(index, False),
        }
        for index, operation in enumerate(_VARIANT_OPERATIONS)
    }


@cache
def compiled():
    import torch
    from tvm_ffi import load_module
    from tvm_ffi.cpp import build_inline

    if (
        os.environ.get("OH_MY_VLLM_CUDA_DEBUG_SYNC") == "1"
        and os.environ.get("OH_MY_VLLM_ENFORCE_EAGER") != "1"
    ):
        raise RuntimeError(
            "CUDA debug sync requires OH_MY_VLLM_ENFORCE_EAGER=1 before Python starts"
        )
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("the CUDA custom-kernel backend requires B200/SM100")
    cuda_source = Path(__file__).with_name("kernels.cu").read_text()
    input_digest = _build_input_digest(cuda_source, torch.__version__)
    nvcc_path, nvcc_version = _nvcc_identity()
    if nvcc_path is None or nvcc_version.startswith("nvcc-unavailable:"):
        record = _cached_manifest(input_digest, nvcc_path)
        so_path = Path(record["so_path"]).resolve()
        module = load_module(so_path)
        if hashlib.sha256(so_path.read_bytes()).hexdigest() != record["so_sha256"]:
            raise RuntimeError("offline CUDA shared library changed while loading")
        _record_loaded(
            so_path,
            record["so_sha256"],
            Path(record["nvcc_path"]),
            record["nvcc_version"],
            "manifest",
        )
        return module
    compiler_key = hashlib.sha256(f"{nvcc_path}\n{nvcc_version}".encode()).hexdigest()[
        :20
    ]
    so_path = Path(
        build_inline(
            "oh_my_vllm_cuda",
            cuda_sources=cuda_source,
            functions=list(_FUNCTIONS),
            backend="cuda",
            extra_cuda_cflags=[
                *_BASE_CUDA_FLAGS,
                f"-DOH_MY_VLLM_NVCC_ID_{compiler_key}",
            ],
        )
    ).resolve()
    if not _in_cuda_cache(so_path):
        raise RuntimeError("CUDA build returned a library outside the configured cache")
    before = hashlib.sha256(so_path.read_bytes()).hexdigest()
    module = load_module(so_path)
    after = hashlib.sha256(so_path.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("CUDA shared library changed while loading")
    final_nvcc_path, final_nvcc_version = _nvcc_identity()
    final_input_digest = _build_input_digest(cuda_source, torch.__version__)
    if (final_nvcc_path, final_nvcc_version) != (
        nvcc_path,
        nvcc_version,
    ) or final_input_digest != input_digest:
        nvcc_version = "nvcc-changed-during-build"
    else:
        _write_manifest(so_path, input_digest, nvcc_path, nvcc_version, after)
    _record_loaded(so_path, after, nvcc_path, nvcc_version, "live")
    return module


def factory_for(module, name):
    def missing(*args, **kwargs):
        raise NotImplementedError(f"CUDA kernel {module}.{name} is not implemented")

    if (module, name) == ("decode_attention", "_merge"):
        return lambda: compiled().attention_merge
    if (module, name) == ("decode_attention", "_partials"):

        def partials(splits, first, grouped, position_dtype, max_query_len=5):
            if position_dtype not in ("int32", "int64"):
                raise ValueError("CUDA decode position dtype must be int32 or int64")
            if type(max_query_len) is not int or not 1 <= max_query_len <= 8:
                raise ValueError("CUDA decode query bound must be in 1..8")
            fn = compiled().attention_partial

            def launch(q, cache, tables, lengths, starts, partial, lse):
                if (
                    partial.ndim != 4
                    or lse.ndim != 3
                    or (partial.shape[2] != splits or lse.shape[2] != splits)
                ):
                    raise ValueError(
                        "CUDA decode partial buffers must match split count"
                    )
                return fn(
                    q,
                    cache,
                    tables,
                    lengths,
                    starts,
                    partial,
                    lse,
                    first,
                    grouped,
                    position_dtype == "int64",
                    max_query_len,
                )

            return launch

        return partials
    if (module, name) == ("attention", "_append"):
        return lambda *config: compiled().append
    if (module, name) == ("convolution", "_conv"):
        return lambda: compiled().convolution
    if (module, name) == ("gdn", "_normalize_qk"):
        return lambda h, qs, ks: compiled().normalize_qk
    if (module, name) == ("gdn", "_recurrent"):
        return lambda: compiled().recurrent
    if module == "normalization":
        if name == "_rms":

            def rms(h, d, eps, gated, xs, gs):
                fn = compiled().rms
                return lambda x, w, g, out: fn(x, w, g, out, eps, gated)

            return rms
        if name == "_add_rms":
            return lambda d: compiled().add_rms
        if name == "_rms_rotary":
            return lambda h, strides, index: compiled().rms_rope
        if name == "_rope":

            def rope(h, d, rotary, theta, index):
                fn = compiled().rope
                return lambda x, positions, out: fn(x, positions, out, rotary, theta)

            return rope
    if (module, name) == ("elementwise", "_gates"):
        return lambda: compiled().gates
    if (module, name) == ("elementwise", "_silu_mul"):
        return lambda: compiled().silu_mul
    if (module, name) != ("fp8", "_quantize"):
        return missing

    def quantize(column, silu):
        implementation = compiled()

        def launch(x, out, scales):
            implementation.quantize(x, out, scales, column, silu)

        return launch

    return quantize
