"""FlashInfer BF16 vocabulary GEMM with an FP32 semantic result."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import Operation, TensorSpec


def _fake_logits(hidden: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    torch._check(hidden.ndim == 2 and head.ndim == 2)
    torch._check(hidden.shape[1] == head.shape[1])
    return torch.empty(
        (hidden.shape[0], head.shape[0]), device=hidden.device, dtype=torch.float32
    )


@torch.library.custom_op("oh_my_vllm_ir::logits_gemm", mutates_args=())
def _ir_logits(hidden: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    return _LOGITS.select(hidden, head).op(hidden, head)


_ir_logits.register_fake(_fake_logits)


@torch.library.custom_op("oh_my_vllm_native::logits_gemm", mutates_args=())
def _native_logits(hidden: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    return F.linear(hidden, head).float()


_native_logits.register_fake(_fake_logits)


@torch.library.custom_op("oh_my_vllm_flashinfer::logits_gemm", mutates_args=())
def _flashinfer_logits(hidden: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    from flashinfer.gemm import mm_bf16

    return mm_bf16(hidden, head.T, backend="cute-dsl").float()


_flashinfer_logits.register_fake(_fake_logits)


_LOGITS = Operation(
    "logits_gemm",
    _ir_logits,
    _native_logits,
    default_priority=("flashinfer",),
)
_LOGITS.register_impl(
    "flashinfer",
    _flashinfer_logits,
    supports=lambda hidden, head: (
        isinstance(hidden, TensorSpec)
        and isinstance(head, TensorSpec)
        and hidden.device.type == "cuda"
        and head.device == hidden.device
        and hidden.shape[0] <= 32
    ),
)


def logits_gemm(hidden: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    return _LOGITS(hidden, head)
