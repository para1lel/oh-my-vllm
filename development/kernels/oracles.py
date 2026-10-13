"""Torch-only mathematical references for the complete operator harness."""

import math
from itertools import pairwise

import torch


def normalize_qk(q, k):
    def normalize(x):
        values = x.float()
        return (values * torch.rsqrt(values.square().sum(-1, keepdim=True) + 1e-6)).to(
            x.dtype
        )

    return normalize(q), normalize(k)


def append(cache, k, v, slots):
    valid = slots >= 0
    rows = slots[valid].long()
    cache[rows // 784, 0, rows % 784] = k[valid]
    cache[rows // 784, 1, rows % 784] = v[valid]


def rms_rotary(x, weight, positions):
    from oh_my_vllm.ir.state import _reference_norm_rope

    return _reference_norm_rope(x, weight, positions)


def decode(
    query,
    cache,
    tables,
    lengths,
    *,
    first=0,
    max_tokens=65536,
    starts=None,
    max_query_len=5,
):
    outputs = []
    for row in range(len(query)):
        length = int(lengths[row])
        indices = torch.arange(first, length, device=query.device)
        pages = tables[row, indices // 784].long()
        k, v = cache[pages, 0, indices % 784], cache[pages, 1, indices % 784]
        repeat = query.shape[1] // k.shape[1]
        k, v = (
            k.double().repeat_interleave(repeat, 1),
            v.double().repeat_interleave(repeat, 1),
        )
        scores = torch.einsum("hd,thd->ht", query[row].double(), k) / math.sqrt(
            query.shape[-1]
        )
        outputs.append(
            torch.einsum("ht,thd->hd", scores.softmax(-1), v).to(query.dtype)
        )
    return torch.stack(outputs)


def prefill(q, k, v, log_decay, beta, initial, starts):
    q, k = normalize_qk(q, k)
    repeat = v.shape[1] // q.shape[1]
    q, k = (
        q.double().repeat_interleave(repeat, 1),
        k.double().repeat_interleave(repeat, 1),
    )
    values, decay, update = v.double(), log_decay.double(), beta.double()
    outputs, states = [], []
    for sequence, (left, right) in enumerate(pairwise(starts.cpu().tolist())):
        state = initial[sequence].double()
        for begin in range(left, right, 64):
            end = min(begin + 64, right)
            query, key, value = (x[begin:end].transpose(0, 1) for x in (q, k, values))
            cumulative = decay[begin:end].T.cumsum(-1)
            multiplier = (cumulative[:, :, None] - cumulative[:, None, :]).exp()
            lower = (
                (key @ key.transpose(-1, -2))
                * multiplier
                * update[begin:end].T[:, :, None]
            )
            triangular = lower.tril(-1) + torch.eye(
                end - begin, device=q.device, dtype=torch.float64
            )
            residual = (
                value - (key @ state.transpose(-1, -2)) * cumulative.exp()[:, :, None]
            )
            delta = torch.linalg.solve_triangular(
                triangular, residual * update[begin:end].T[:, :, None], upper=False
            )
            attention = ((query @ key.transpose(-1, -2)) * multiplier).tril()
            result = (query @ state.transpose(-1, -2)) * cumulative.exp()[
                :, :, None
            ] + attention @ delta
            outputs.append((result / math.sqrt(128)).transpose(0, 1).to(v.dtype))
            carried_keys = key * (cumulative[:, -1:] - cumulative).exp()[:, :, None]
            state = (
                state * cumulative[:, -1].exp()[:, None, None]
                + delta.transpose(-1, -2) @ carried_keys
            )
        states.append(state.to(initial.dtype))
    return torch.cat(outputs), torch.stack(states)


def dspark_rms(x, weight, eps=1e-6):
    from oh_my_vllm.ir.dspark import _native_dspark_rms_norm

    return _native_dspark_rms_norm(x, weight, eps)


def dspark_norm_rope(x, weight, positions, inv_freq, attention_factor, eps=1e-6):
    from oh_my_vllm.ir.dspark import _native_dspark_norm_rope

    return _native_dspark_norm_rope(
        x, weight, positions, inv_freq, attention_factor, eps
    )


def quantize(x, *, column_major=False, silu_gate=False):
    if silu_gate:
        gate, up = x.chunk(2, dim=-1)
        x = torch.nn.functional.silu(gate.float()).to(x.dtype) * up
    values = x.cpu().float().reshape(x.shape[0], -1, 128)
    magnitudes = torch.nan_to_num(
        values.abs(), nan=0, posinf=float("inf"), neginf=float("inf")
    )
    scale = magnitudes.amax(-1).clamp_min(1e-10) / 448
    # CPU IEEE division keeps the existing exact FP8 contract. CUDA Torch
    # tensor division uses an approximate reciprocal at some rounding ties.
    divided = values / scale[..., None]
    encoded = torch.fmin(
        torch.full_like(divided, 448),
        torch.fmax(torch.full_like(divided, -448), divided),
    )
    encoded = encoded.reshape(x.shape).to(torch.float8_e4m3fn).to(x.device)
    scale = scale.to(x.device)
    if column_major:
        column_scale = torch.empty(
            (scale.shape[1], scale.shape[0]), dtype=scale.dtype, device=x.device
        ).T
        column_scale.copy_(scale)
        scale = column_scale
    return encoded, scale


def operator(module, name):
    from oh_my_vllm.ir import (
        attention_gate,
        dspark,
        fp8,
        normalized_linear,
        partial_attention,
        pointwise,
        recurrent,
        state,
    )

    direct = {
        ("elementwise", "attention_gate"): attention_gate._native,
        ("attention", "append"): append,
        ("attention_prepare", "prepare_attention"): state._native_prepare,
        ("partial_attention", "prepare_context"): partial_attention._native_context,
        ("partial_attention", "prepare_query"): partial_attention._native_query,
        ("normalization", "add_rms_norm"): pointwise._native_add_rms_norm,
        ("normalization", "rms_rotary"): rms_rotary,
        ("elementwise", "silu_mul"): pointwise._native_silu_mul,
        ("elementwise", "delta_gates"): pointwise._native_delta_gates,
        ("gdn", "normalize_qk"): normalize_qk,
        ("gdn", "recurrent"): recurrent._native_recurrent,
        ("gdn", "prefill"): prefill,
        ("convolution", "causal_conv"): recurrent._native_conv,
        ("decode_attention", "decode"): decode,
        ("fp8", "add_norm_linear"): normalized_linear._native,
        ("fp8", "gated_norm_linear"): normalized_linear._gated_native,
        ("dspark_attention", "rms_norm"): dspark_rms,
        ("dspark_attention", "normalize_rope"): dspark_norm_rope,
        ("dspark_attention", "append"): dspark._native_dspark_append,
        ("dspark_attention", "attention"): dspark._native_dspark_attention,
    }
    if (module, name) == ("normalization", "rms_norm"):
        return lambda x, w, *, gate=None: pointwise._native_rms_norm(x, w, 1e-6, gate)
    if (module, name) == ("fp8", "quantize"):
        return quantize
    if (module, name) == ("fp8", "linear"):
        return lambda x, w, s, *, silu_gate=False: fp8._native_linear(
            x, w, s, silu_gate
        )
    return direct[module, name]
