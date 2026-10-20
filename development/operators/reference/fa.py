import itertools

import torch

from .common import rms


def rotary(value, positions):
    frequency = torch.arange(32, device=value.device, dtype=torch.float64)
    axes = torch.arange(32, device=value.device) % 3
    angle = positions[axes].T * (10000000.0 ** (-frequency / 32))
    angle = angle[:, None, :]
    first, second = value[..., :32], value[..., 32:64]
    return torch.cat(
        (
            first * angle.cos() - second * angle.sin(),
            second * angle.cos() + first * angle.sin(),
            value[..., 64:],
        ),
        dim=-1,
    )


def attend(query, key, value, positions, *, first_token=0):
    repeats = query.shape[1] // key.shape[1]
    key, value = (
        item.repeat_interleave(repeats, dim=1).transpose(0, 1) for item in (key, value)
    )
    scores = query.transpose(0, 1) @ key.transpose(1, 2) / query.shape[-1] ** 0.5
    columns = torch.arange(key.shape[1], device=query.device)
    visible = (columns[None, :] <= positions[:, None]) & (
        columns[None, :] >= first_token
    )
    scores = scores.masked_fill(~visible[None, :, :], -torch.inf)
    return (torch.softmax(scores, dim=-1) @ value).transpose(0, 1)


def encode(values, key_norm):
    scales = values.abs().amax(dim=-1).clamp_min(1e-10) / 448
    scales[:, 0] = (32 * (key_norm + 1).abs().max()).clamp_min(1e-10) / 448
    encoded = (values / scales[..., None]).to(torch.float8_e4m3fn).to(torch.float64)
    return encoded, scales


def compute(
    value, residual, inputs, cache, weights, prefix, *, variant="full", first_token=0
):
    summed = value + residual
    normalized = rms(summed, weights.tensor(prefix + "input_layernorm.weight"))
    attention = prefix + "self_attn."
    slots = inputs["slots"].long()
    if variant != "query_only":
        key = weights.linear(normalized, attention + "k_proj").reshape(-1, 4, 256)
        key = rotary(
            rms(key, weights.tensor(attention + "k_norm.weight")), inputs["positions"]
        )
        values = weights.linear(normalized, attention + "v_proj").reshape(-1, 4, 256)
        cache[slots // 784, :, slots % 784] = torch.stack((key, values), dim=1)
    if variant == "kv_only":
        return None, summed
    query_gate = weights.linear(normalized, attention + "q_proj").reshape(-1, 24, 512)
    query, gate = query_gate.split(256, dim=-1)
    query = rotary(
        rms(query, weights.tensor(attention + "q_norm.weight")), inputs["positions"]
    )
    starts = inputs["sequence_rows"].tolist()
    outputs = []
    for sequence, (first, last) in enumerate(itertools.pairwise(starts)):
        length = int(inputs["lengths"][last - 1])
        positions = torch.arange(length, device=value.device)
        pages = inputs["tables"][sequence, positions // 784].long()
        history = cache[pages, :, positions % 784]
        outputs.append(
            attend(
                query[first:last],
                history[:, 0],
                history[:, 1],
                inputs["lengths"][first:last] - 1,
                first_token=first_token,
            )
        )
    output = torch.cat(outputs) * torch.sigmoid(gate)
    return weights.linear(output.reshape(-1, 6144), attention + "o_proj"), summed


def written_cache(state, inputs, weights, prefix, *, indices=None):
    slots = inputs["slots"].long() if indices is None else indices
    values = state["kv"][slots // 784, :, slots % 784]
    encoded, scales = encode(values, weights.tensor(prefix + "self_attn.k_norm.weight"))
    return {"kv": values, "fp8_kv": encoded, "kv_scales": scales}


def reference(case, inputs, state, weights):
    prefix = f"model.language_model.layers.{case.weights[0]}."
    value, residual = compute(
        inputs["value"], inputs["residual"], inputs, state["kv"], weights, prefix
    )
    return {
        "value": value,
        "residual": residual,
        **written_cache(state, inputs, weights, prefix),
    }
