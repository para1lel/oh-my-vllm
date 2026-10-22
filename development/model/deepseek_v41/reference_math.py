"""Independent PyTorch FP64 mathematics, without native inference operators.

Source contracts: LOCAL-V41 model.py and kernel.py. These functions omit
activation quantization and intermediate BF16/FP32 rounding deliberately.
"""

import math

import torch
import torch.nn.functional as functional


def require_fp64(*values):
    if any(
        not isinstance(value, torch.Tensor) or value.dtype != torch.float64
        for value in values
    ):
        raise ValueError("reference arithmetic requires FP64 tensors")


def rms(value, weight, *, eps):
    require_fp64(value, weight)
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight


def hc_collapse(value, pre):
    require_fp64(value, pre)
    return (pre.unsqueeze(-1) * value).sum(-2)


def hc_expand(value, residual, post, comb):
    require_fp64(value, residual, post, comb)
    return post.unsqueeze(-1) * value.unsqueeze(-2) + torch.einsum(
        "...ij,...id->...jd", comb, residual
    )


def hc_mixes(value, weight, scale, base, *, iterations, eps, norm_eps):
    require_fp64(value, weight, scale, base)
    if type(iterations) is not int or iterations < 1 or eps < 0 or norm_eps <= 0:
        raise ValueError("invalid hyper-connection normalization parameters")
    copies = value.shape[-2]
    flat = value.flatten(-2)
    mixed = (flat @ weight.T) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + norm_eps
    )
    pre = torch.sigmoid(mixed[..., :copies] * scale[0] + base[:copies]) + eps
    post = 2 * torch.sigmoid(
        mixed[..., copies : 2 * copies] * scale[1] + base[copies : 2 * copies]
    )
    residual = (mixed[..., 2 * copies :] * scale[2] + base[2 * copies :]).reshape(
        *value.shape[:-2], copies, copies
    )
    residual = torch.softmax(residual, dim=-1) + eps
    residual = residual / (residual.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        residual = residual / (residual.sum(-1, keepdim=True) + eps)
        residual = residual / (residual.sum(-2, keepdim=True) + eps)
    return pre, post, residual


def rotary(
    value,
    positions,
    *,
    base,
    inverse=False,
    original_context=0,
    factor=1.0,
    beta_fast=32.0,
    beta_slow=1.0,
):
    require_fp64(value)
    width = value.shape[-1]
    if value.ndim < 2 or width == 0 or width % 2:
        raise ValueError("rotary input must have an even channel count")
    if (
        positions.dtype not in (torch.int32, torch.int64)
        or positions.shape != (value.shape[0],)
        or bool((positions < 0).any())
        or base <= 1
        or factor <= 0
        or original_context < 0
        or beta_slow <= 0
        or beta_fast < beta_slow
    ):
        raise ValueError("invalid rotary positions or frequencies")
    index = torch.arange(width // 2, dtype=torch.float64, device=value.device)
    frequency = base ** (-2 * index / width)
    if original_context:
        low = max(
            math.floor(
                width
                * math.log(original_context / (beta_fast * 2 * math.pi))
                / (2 * math.log(base))
            ),
            0,
        )
        high = min(
            math.ceil(
                width
                * math.log(original_context / (beta_slow * 2 * math.pi))
                / (2 * math.log(base))
            ),
            width - 1,
        )
        ramp = ((index - low) / max(high - low, 1e-3)).clamp(0, 1)
        frequency = frequency * (1 - ramp + ramp / factor)
    angle = positions.to(device=value.device, dtype=torch.float64)[:, None] * frequency
    if inverse:
        angle = -angle
    angle = angle.reshape(value.shape[0], *([1] * (value.ndim - 2)), width // 2)
    real, imaginary = value[..., 0::2], value[..., 1::2]
    return torch.stack(
        (
            real * angle.cos() - imaginary * angle.sin(),
            real * angle.sin() + imaginary * angle.cos(),
        ),
        dim=-1,
    ).flatten(-2)


def sparse_attention(query, key_value, sink, indices, *, scale):
    require_fp64(query, key_value, sink)
    if len(indices) != query.shape[0]:
        raise ValueError("one sparse index row is required per query")
    outputs = []
    for row, selection in enumerate(indices):
        selected = torch.as_tensor(selection, device=query.device)
        if selected.ndim != 1 or (
            selected.numel() and selected.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("sparse indices must be one-dimensional integer rows")
        selected = selected.to(torch.int64)
        if bool(((selected < -1) | (selected >= key_value.shape[0])).any()):
            raise ValueError("sparse index leaves the available cache")
        keys = key_value[selected[selected >= 0]]
        scores = (query[row] @ keys.T) * scale
        probabilities = torch.softmax(
            torch.cat((scores, sink[:, None]), dim=-1), dim=-1
        )
        outputs.append(probabilities[:, :-1] @ keys)
    return torch.stack(outputs)


def route(logits, bias, *, topk, scale):
    require_fp64(logits, bias)
    if type(topk) is not int or not 1 <= topk <= logits.shape[-1] or scale <= 0:
        raise ValueError("invalid expert routing parameters")
    scores = functional.softplus(logits).sqrt()
    # Global expert ID resolves exact score ties independently of rank layout.
    indices = torch.argsort(scores + bias, dim=-1, descending=True, stable=True)[
        ..., :topk
    ]
    weights = scores.gather(-1, indices)
    if topk > 1:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return indices, weights * scale
