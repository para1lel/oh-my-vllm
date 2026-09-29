"""Convolution and GDN recurrence with explicit persistent-state writes."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from oh_my_vllm.kernels import convolution, gdn
from oh_my_vllm.kernels.backend import NAME

from .core import Operation, TensorSpec


def _cuda_specs(*specs: TensorSpec) -> bool:
    return all(spec.device.type == "cuda" for spec in specs)


def _fake_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    pool: torch.Tensor,
    sequence_ids: torch.Tensor,
    starts: torch.Tensor,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
) -> torch.Tensor:
    torch._check(x.ndim == 2 and weight.shape == (x.shape[1], 4))
    torch._check(pool.ndim == 3 and pool.shape[1:] == (x.shape[1], 3))
    torch._check(sequence_ids.shape == (x.shape[0],))
    torch._check(starts.numel() == read_slots.numel() + 1)
    torch._check(write_slots.shape == (x.shape[0],))
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


@torch.library.custom_op("oh_my_vllm_ir::causal_conv", mutates_args=("pool",))
def _ir_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    pool: torch.Tensor,
    sequence_ids: torch.Tensor,
    starts: torch.Tensor,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
) -> torch.Tensor:
    args = (x, weight, pool, sequence_ids, starts, read_slots, write_slots)
    return _CONV.select(*args).op(*args)


_ir_conv.register_fake(_fake_conv)


@torch.library.custom_op("oh_my_vllm_native::causal_conv", mutates_args=("pool",))
def _native_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    pool: torch.Tensor,
    sequence_ids: torch.Tensor,
    starts: torch.Tensor,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
) -> torch.Tensor:
    _fake_conv(x, weight, pool, sequence_ids, starts, read_slots, write_slots)
    offsets = starts.cpu().tolist()
    ids = sequence_ids.cpu().tolist()
    writes = write_slots.cpu().tolist()
    sources = pool.index_select(0, read_slots.long()).clone()
    outputs = []
    for sequence, source in enumerate(sources):
        begin, end = offsets[sequence : sequence + 2]
        if any(value != sequence for value in ids[begin:end]):
            raise ValueError("convolution sequence IDs disagree with offsets")
        history = torch.cat((source, x[begin:end].T), dim=1)
        values = F.conv1d(
            history.float()[None], weight.float()[:, None], groups=x.shape[1]
        )[0].T
        outputs.append(F.silu(values).to(x.dtype))
        for token in range(begin, end):
            if writes[token] >= 0:
                offset = token - begin + 1
                pool[writes[token]] = history[:, offset : offset + 3]
    return torch.cat(outputs, dim=0)


_native_conv.register_fake(_fake_conv)


@torch.library.custom_op("oh_my_vllm_kernel::causal_conv", mutates_args=("pool",))
def _kernel_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    pool: torch.Tensor,
    sequence_ids: torch.Tensor,
    starts: torch.Tensor,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
) -> torch.Tensor:
    return convolution.causal_conv(
        x, weight, pool, sequence_ids, starts, read_slots, write_slots
    )


_kernel_conv.register_fake(_fake_conv)


_CONV = Operation("causal_conv", _ir_conv, _native_conv, default_priority=(NAME,))
_CONV.register_impl(
    NAME,
    _kernel_conv,
    supports=lambda x, weight, pool, sequence_ids, starts, reads, writes: _cuda_specs(
        x, weight, pool, sequence_ids, starts, reads, writes
    ),
)


def _fake_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    pool: torch.Tensor,
    starts: torch.Tensor,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
) -> torch.Tensor:
    torch._check(q.shape == k.shape and q.ndim == 3 and q.shape[-1] == 128)
    torch._check(v.ndim == 3 and v.shape[0] == q.shape[0] and v.shape[-1] == 128)
    torch._check(v.shape[1] % q.shape[1] == 0)
    torch._check(pool.shape[1:] == (v.shape[1], 128, 128))
    torch._check(log_decay.shape == v.shape[:2] and beta.shape == v.shape[:2])
    torch._check(starts.numel() == read_slots.numel() + 1)
    torch._check(write_slots.shape == (q.shape[0],))
    return torch.empty(v.shape, device=v.device, dtype=v.dtype)


@torch.library.custom_op("oh_my_vllm_ir::gdn_recurrent", mutates_args=("pool",))
def _ir_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    pool: torch.Tensor,
    starts: torch.Tensor,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
) -> torch.Tensor:
    args = (q, k, v, log_decay, beta, pool, starts, read_slots, write_slots)
    return _RECURRENT.select(*args).op(*args)


_ir_recurrent.register_fake(_fake_recurrent)


@torch.library.custom_op("oh_my_vllm_native::gdn_recurrent", mutates_args=("pool",))
def _native_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    pool: torch.Tensor,
    starts: torch.Tensor,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
) -> torch.Tensor:
    _fake_recurrent(q, k, v, log_decay, beta, pool, starts, read_slots, write_slots)
    offsets = starts.cpu().tolist()
    writes = write_slots.cpu().tolist()
    sources = pool.index_select(0, read_slots.long()).double().clone()
    repeat = v.shape[1] // q.shape[1]
    normalized_q = q.double() * torch.rsqrt(
        q.double().square().sum(-1, keepdim=True) + 1e-6
    )
    normalized_k = k.double() * torch.rsqrt(
        k.double().square().sum(-1, keepdim=True) + 1e-6
    )
    normalized_q = normalized_q.repeat_interleave(repeat, 1) / math.sqrt(128)
    normalized_k = normalized_k.repeat_interleave(repeat, 1)
    outputs = []
    for sequence, initial in enumerate(sources):
        begin, end = offsets[sequence : sequence + 2]
        current = initial
        for token in range(begin, end):
            current = current * log_decay[token].double().exp()[:, None, None]
            residual = v[token].double() - torch.einsum(
                "hvk,hk->hv", current, normalized_k[token]
            )
            current = current + torch.einsum(
                "hv,hk->hvk",
                residual * beta[token].double()[:, None],
                normalized_k[token],
            )
            outputs.append(
                torch.einsum("hvk,hk->hv", current, normalized_q[token]).to(v.dtype)
            )
            if writes[token] >= 0:
                pool[writes[token]] = current.to(pool.dtype)
    return torch.stack(outputs)


_native_recurrent.register_fake(_fake_recurrent)


@torch.library.custom_op("oh_my_vllm_kernel::gdn_recurrent", mutates_args=("pool",))
def _kernel_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    pool: torch.Tensor,
    starts: torch.Tensor,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
) -> torch.Tensor:
    return gdn.recurrent(
        q, k, v, log_decay, beta, pool, starts, read_slots, write_slots
    )


_kernel_recurrent.register_fake(_fake_recurrent)


_RECURRENT = Operation(
    "gdn_recurrent",
    _ir_recurrent,
    _native_recurrent,
    default_priority=(NAME,),
)
_RECURRENT.register_impl(
    NAME,
    _kernel_recurrent,
    supports=lambda q, k, v, decay, beta, pool, starts, reads, writes: _cuda_specs(
        q, k, v, decay, beta, pool, starts, reads, writes
    ),
)


def causal_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    pool: torch.Tensor,
    sequence_ids: torch.Tensor,
    starts: torch.Tensor,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
) -> torch.Tensor:
    return _CONV(x, weight, pool, sequence_ids, starts, read_slots, write_slots)


def gdn_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    pool: torch.Tensor,
    starts: torch.Tensor,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
) -> torch.Tensor:
    return _RECURRENT(q, k, v, log_decay, beta, pool, starts, read_slots, write_slots)
