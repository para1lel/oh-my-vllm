"""FlashInfer GDN prefill as a semantic op with a PyTorch recurrence reference."""

from __future__ import annotations

import math

import torch

from oh_my_vllm.kernels import gdn

from .core import Operation, TensorSpec


def _fake_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(q.shape == k.shape and q.ndim == 3 and q.shape[-1] == 128)
    torch._check(v.ndim == 3 and v.shape[0] == q.shape[0] and v.shape[-1] == 128)
    torch._check(v.shape[1] % q.shape[1] == 0)
    torch._check(log_decay.shape == v.shape[:2] and beta.shape == v.shape[:2])
    torch._check(initial_state.shape == (starts.numel() - 1, v.shape[1], 128, 128))
    return (
        torch.empty(v.shape, device=v.device, dtype=v.dtype),
        torch.empty(
            initial_state.shape, device=initial_state.device, dtype=initial_state.dtype
        ),
    )


@torch.library.custom_op("oh_my_vllm_ir::gdn_prefill", mutates_args=())
def _ir_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    args = (q, k, v, log_decay, beta, initial_state, starts)
    return _PREFILL.select(*args).op(*args)


_ir_prefill.register_fake(_fake_prefill)


@torch.library.custom_op("oh_my_vllm_native::gdn_prefill", mutates_args=())
def _native_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _fake_prefill(q, k, v, log_decay, beta, initial_state, starts)
    q64, k64 = q.double(), k.double()
    q64 = q64 * torch.rsqrt(q64.square().sum(-1, keepdim=True) + 1e-6)
    k64 = k64 * torch.rsqrt(k64.square().sum(-1, keepdim=True) + 1e-6)
    repeat = v.shape[1] // q.shape[1]
    q64 = q64.repeat_interleave(repeat, dim=1) / math.sqrt(128)
    k64 = k64.repeat_interleave(repeat, dim=1)
    offsets = starts.cpu().tolist()
    outputs = []
    final_states = []
    for seq in range(len(offsets) - 1):
        current = initial_state[seq].double()
        for token in range(offsets[seq], offsets[seq + 1]):
            current = current * log_decay[token].double().exp()[:, None, None]
            residual = v[token].double() - torch.einsum(
                "hvk,hk->hv", current, k64[token]
            )
            current = current + torch.einsum(
                "hv,hk->hvk", residual * beta[token].double()[:, None], k64[token]
            )
            outputs.append(torch.einsum("hvk,hk->hv", current, q64[token]).to(v.dtype))
        final_states.append(current.to(initial_state.dtype))
    return torch.stack(outputs), torch.stack(final_states)


_native_prefill.register_fake(_fake_prefill)


@torch.library.custom_op("oh_my_vllm_flashinfer::gdn_prefill", mutates_args=())
def _flashinfer_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return gdn.prefill(q, k, v, log_decay, beta, initial_state, starts)


_flashinfer_prefill.register_fake(_fake_prefill)


_PREFILL = Operation(
    "gdn_prefill",
    _ir_prefill,
    _native_prefill,
    default_priority=("flashinfer",),
)
_PREFILL.register_impl(
    "flashinfer",
    _flashinfer_prefill,
    supports=lambda q, k, v, decay, beta, initial, starts: all(
        isinstance(spec, TensorSpec) and spec.device.type == "cuda"
        for spec in (q, k, v, decay, beta, initial, starts)
    ),
)


def gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _PREFILL(q, k, v, log_decay, beta, initial_state, starts)
