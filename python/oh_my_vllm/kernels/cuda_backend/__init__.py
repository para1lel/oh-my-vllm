"""B200 CUDA implementations, compiled lazily; executed on the caller's CUDA stream."""

import hashlib
import json
import subprocess
import sys
from functools import cache
from pathlib import Path

_loaded_so_path: Path | None = None
_loaded_so_sha256: str | None = None
_loaded_nvcc_path: Path | None = None
_loaded_nvcc_version: str | None = None


def _nvcc_identity() -> tuple[Path, str]:
    """Query the compiler selected by TVM FFI, rather than PATH's nvcc."""
    from tvm_ffi.cpp.extension import _find_cuda_home

    path = (Path(_find_cuda_home()) / "bin" / "nvcc").resolve()
    try:
        version = subprocess.check_output([str(path), "--version"], text=True)
        return path, version.strip().splitlines()[-1]
    except Exception as exc:
        return path, f"nvcc-unavailable: {exc}"


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
        "so_path": str(path) if path is not None else None,
        "so_sha256": _loaded_so_sha256,
    }


@cache
def compiled():
    import torch
    from tvm_ffi import load_module
    from tvm_ffi.cpp import build_inline

    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("the CUDA custom-kernel backend requires B200/SM100")
    nvcc_path, nvcc_version = _nvcc_identity()
    compiler_key = hashlib.sha256(f"{nvcc_path}\n{nvcc_version}".encode()).hexdigest()[
        :20
    ]
    so_path = Path(
        build_inline(
            "oh_my_vllm_cuda",
            cuda_sources=Path(__file__).with_name("kernels.cu").read_text(),
            functions=[
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
            ],
            extra_cuda_cflags=[
                "-O3",
                "--generate-code=arch=compute_100a,code=sm_100a",
                f"-DOH_MY_VLLM_NVCC_ID_{compiler_key}",
            ],
        )
    ).resolve()
    before = hashlib.sha256(so_path.read_bytes()).hexdigest()
    module = load_module(so_path)
    after = hashlib.sha256(so_path.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("CUDA shared library changed while loading")
    final_nvcc_path, final_nvcc_version = _nvcc_identity()
    if (final_nvcc_path, final_nvcc_version) != (nvcc_path, nvcc_version):
        nvcc_version = "nvcc-changed-during-build"
    global _loaded_so_path, _loaded_so_sha256, _loaded_nvcc_path, _loaded_nvcc_version
    _loaded_so_path = so_path
    _loaded_so_sha256 = after
    _loaded_nvcc_path = nvcc_path
    _loaded_nvcc_version = nvcc_version
    print(
        "CUDA_BUILD_PROVENANCE " + json.dumps(provenance(require_loaded=True)),
        file=sys.stderr,
        flush=True,
    )
    return module


def factory_for(module, name):
    def missing(*args, **kwargs):
        raise NotImplementedError(f"CUDA kernel {module}.{name} is not implemented")

    if (module, name) == ("decode_attention", "_merge"):
        return lambda *config: compiled().attention_merge
    if (module, name) == ("decode_attention", "_partials"):

        def partials(
            h,
            hk,
            pages,
            table_width,
            splits,
            first,
            bk,
            bq,
            grouped,
            index_types,
            position_dtype,
        ):
            fn = compiled().attention_partial
            return lambda q, cache, tables, lengths, starts, partial, lse: fn(
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
            )

        return partials
    if (module, name) == ("attention", "_append"):
        return lambda *config: compiled().append
    if (module, name) == ("convolution", "_conv"):
        return lambda *config: compiled().convolution
    if (module, name) == ("gdn", "_normalize_qk"):
        return lambda h, qs, ks: compiled().normalize_qk
    if (module, name) == ("gdn", "_recurrent"):
        return lambda *config: compiled().recurrent
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
        return lambda width, block: compiled().silu_mul
    if (module, name) != ("fp8", "_quantize"):
        return missing

    def quantize(width, dtype, column, silu, tile):
        implementation = compiled()

        def launch(x, out, scales):
            implementation.quantize(x, out, scales, column, silu)

        return launch

    return quantize
