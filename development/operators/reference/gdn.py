import itertools

import torch
import torch.nn.functional as functional

from ..bindings import EMBEDDING
from .common import rms


def step(state, query, key, value, beta, decay):
    decayed = state * decay.exp()[:, None, None]
    recalled = (decayed @ key[..., None]).squeeze(-1)
    delta = beta[:, None] * (value - recalled)
    updated = decayed + delta[:, :, None] * key[:, None, :]
    output = (updated @ query[..., None]).squeeze(-1) / query.shape[-1] ** 0.5
    return updated, output


def projection_scales(value, weight, starts):
    scales = (value.square() @ weight.square().T).sqrt()
    return [
        scales[first:last].amax(dim=0).cpu().tolist()
        for first, last in itertools.pairwise(starts)
    ]


def reference(case, inputs, state, weights, *, conditioning=None):
    prefix = f"model.language_model.layers.{case.weights[0]}."
    attention = prefix + "linear_attn."
    value = (
        weights.tensor(EMBEDDING, rows=inputs["tokens"].cpu().numpy())
        if case.weights[0] == 0
        else inputs["value"]
    )
    summed = value + inputs["residual"]
    normalized = rms(summed, weights.tensor(prefix + "input_layernorm.weight"))
    projected = weights.linear(normalized, attention + "in_proj_qkv")
    gate = weights.linear(normalized, attention + "in_proj_z").reshape(-1, 48, 128)
    beta = weights.linear(normalized, attention + "in_proj_b").sigmoid()
    rate = weights.linear(normalized, attention + "in_proj_a") + weights.tensor(
        attention + "dt_bias"
    )
    decay = -weights.tensor(attention + "A_log").exp() * torch.logaddexp(
        rate, torch.zeros_like(rate)
    )
    convolution = weights.tensor(attention + "conv1d.weight").reshape(10240, 4)
    starts = inputs["sequence_rows"].tolist()
    writes = inputs["write_slots"].clone()
    if not case.options["verify"]:
        writes[inputs["checkpoint_rows"].long()] = inputs["checkpoint_slots"]
    outputs, recurrent, histories = [], [], []
    for sequence, (first, last) in enumerate(itertools.pairwise(starts)):
        slot = int(inputs["read_slots"][sequence])
        carry = state["recurrent"][slot].clone()
        history = state["convolution"][slot].T
        window = torch.cat((history, projected[first:last]))
        convolved = sum(
            window[tap : tap + last - first] * convolution[:, tap] for tap in range(4)
        )
        query, key, values = functional.silu(convolved).split(
            (2048, 2048, 6144), dim=-1
        )
        query, key = (item.reshape(-1, 16, 128) for item in (query, key))
        query, key = (
            (
                item * torch.rsqrt(item.square().sum(dim=-1, keepdim=True) + 1e-6)
            ).repeat_interleave(3, dim=1)
            for item in (query, key)
        )
        values = values.reshape(-1, 48, 128)
        for local, row in enumerate(range(first, last)):
            carry, output = step(
                carry, query[local], key[local], values[local], beta[row], decay[row]
            )
            outputs.append(output)
            if int(writes[row]) >= 0:
                recurrent.append(carry.clone())
                histories.append(window[local + 1 : local + 4].T.clone())
    output = rms(
        torch.stack(outputs), weights.tensor(attention + "norm.weight"), offset=False
    )
    output = (output * functional.silu(gate)).reshape(-1, 6144)
    if conditioning is not None:
        conditioning["value"] = projection_scales(
            output, weights.tensor(attention + "out_proj.weight"), starts
        )
    return {
        "value": weights.linear(output, attention + "out_proj"),
        "residual": summed,
        "recurrent": torch.stack(recurrent),
        "convolution": torch.stack(histories),
    }
