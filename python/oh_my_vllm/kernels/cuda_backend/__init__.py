"""B200 CUDA implementations, selected explicitly during staged migration."""

from functools import cache
from pathlib import Path


@cache
def compiled():
    import torch
    from tvm_ffi.cpp import load_inline

    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("the CUDA custom-kernel backend requires B200/SM100")
    return load_inline(
        "oh_my_vllm_cuda",
        cuda_sources=Path(__file__).with_name("kernels.cu").read_text(),
        functions=["quantize", "silu_mul"],
        extra_cuda_cflags=["-O3", "--generate-code=arch=compute_100a,code=sm_100a"],
    )


def factory_for(module, name):
    def missing(*args, **kwargs):
        raise NotImplementedError(f"CUDA kernel {module}.{name} is not implemented")

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
