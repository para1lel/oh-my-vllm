"""B200 CUDA implementations, compiled lazily; executed on the caller's CUDA stream."""

import hashlib
import subprocess
from functools import cache
from pathlib import Path


def _nvcc_version() -> str:
    """Return the last line of `nvcc --version`, e.g. 'V12.4.131'."""
    try:
        return (
            subprocess.check_output(["nvcc", "--version"], text=True)
            .strip()
            .splitlines()[-1]
        )
    except Exception as exc:
        return f"nvcc-unavailable: {exc}"


def provenance() -> dict:
    """Return CUDA build provenance: nvcc version and compiled .so hash (once built)."""
    import glob
    import os

    info: dict = {"nvcc_version": _nvcc_version()}
    # The TVM FFI cache dir is set by scripts/with-env.sh.
    cache_dir = os.environ.get("TVM_FFI_CACHE_DIR", "")
    if cache_dir:
        so_paths = sorted(
            glob.glob(f"{cache_dir}/**/oh_my_vllm_cuda*.so", recursive=True)
        )
        if so_paths:
            # Hash the most recently modified .so; there is normally only one.
            so_path = max(so_paths, key=lambda p: Path(p).stat().st_mtime)
            info["so_sha256"] = hashlib.sha256(Path(so_path).read_bytes()).hexdigest()
            info["so_path"] = so_path
        else:
            info["so_sha256"] = None
    else:
        info["so_sha256"] = None
    return info


@cache
def compiled():
    import torch
    from tvm_ffi.cpp import load_inline

    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("the CUDA custom-kernel backend requires B200/SM100")
    return load_inline(
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
        extra_cuda_cflags=["-O3", "--generate-code=arch=compute_100a,code=sm_100a"],
    )


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
