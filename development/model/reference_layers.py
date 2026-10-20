import torch
import torch.nn.functional as functional

from development.operators.reference.common import linear
from development.operators.reference.fa import attend, rotary
from development.operators.reference.vision import block, norm, patch

from .reference import recurrent, rms


def key_value(weights, prefix, normalized, positions):
    key = weights.linear(normalized, prefix + "k_proj").reshape(-1, 4, 256)
    key = rotary(rms(key, weights.tensor(prefix + "k_norm.weight")), positions)
    value = weights.linear(normalized, prefix + "v_proj").reshape(-1, 4, 256)
    return key, value


def attention(
    weights,
    prefix,
    value,
    previous,
    positions,
    start,
    *,
    query_only=False,
    first_token=0,
):
    packed = weights.linear(value, prefix + "q_proj").reshape(-1, 24, 512)
    query, gate = packed.split(256, dim=-1)
    query = rotary(rms(query, weights.tensor(prefix + "q_norm.weight")), positions)
    if query_only:
        state = previous
    else:
        state = key_value(weights, prefix, value, positions)
        if previous is not None:
            state = tuple(
                torch.cat((old, new)) for old, new in zip(previous, state, strict=True)
            )
    pieces = []
    for first in range(0, len(query), 256):
        last = min(first + 256, len(query))
        indices = torch.arange(start + first, start + last, device=query.device)
        pieces.append(
            attend(query[first:last], *state, indices, first_token=first_token)
        )
    output = (torch.cat(pieces) * torch.sigmoid(gate)).reshape(-1, 6144)
    return weights.linear(output, prefix + "o_proj"), state


def feed_forward(weights, prefix, value):
    normalized = rms(value, weights.tensor(prefix + "post_attention_layernorm.weight"))
    gate = functional.silu(weights.linear(normalized, prefix + "mlp.gate_proj"))
    upper = weights.linear(normalized, prefix + "mlp.up_proj")
    return value + weights.linear(gate * upper, prefix + "mlp.down_proj")


def embeddings(weights, tokens):
    return weights.tensor("model.language_model.embed_tokens.weight", rows=tokens)


def forward(
    weights,
    tokens,
    positions,
    *,
    states=None,
    start=0,
    embedded=None,
    project_all=False,
):
    value = embeddings(weights, tokens) if embedded is None else embedded
    next_states = []
    for index, kind in enumerate(weights.config["layer_types"]):
        prefix = f"model.language_model.layers.{index}."
        normalized = rms(value, weights.tensor(prefix + "input_layernorm.weight"))
        previous = None if states is None else states[index]
        if kind == "linear_attention":
            output, state = recurrent(
                weights, prefix + "linear_attn.", normalized, previous
            )
        elif kind == "full_attention":
            output, state = attention(
                weights, prefix + "self_attn.", normalized, previous, positions, start
            )
        else:
            raise ValueError("unsupported language layer")
        next_states.append(state)
        value = feed_forward(weights, prefix, value + output)
        if (index + 1) % 16 == 0:
            print(f"FP64 full case position{start}: {index + 1}/64 layers", flush=True)
    hidden = rms(value, weights.tensor("model.language_model.norm.weight"))
    logits = weights.linear(hidden if project_all else hidden[-1:], "lm_head")
    return logits if project_all else logits.squeeze(0), hidden, next_states


def vision(weights, pixels, grid):
    shape = dict(height=grid[0], width=grid[1])
    value = patch(pixels[None], shape, weights)
    for layer in range(27):
        value = block(value, shape, weights, layer)
    value = norm(value, weights, "model.visual.merger.norm").reshape(-1, 4608)
    value = functional.gelu(
        linear(value, weights, "model.visual.merger.linear_fc1", bias=True)
    )
    return linear(value, weights, "model.visual.merger.linear_fc2", bias=True)


def mtp_input(weights, hidden, embedded):
    value = torch.cat(
        (
            rms(embedded, weights.tensor("mtp.pre_fc_norm_embedding.weight")),
            rms(hidden, weights.tensor("mtp.pre_fc_norm_hidden.weight")),
        ),
        dim=-1,
    )
    return weights.linear(value, "mtp.fc")


def mtp_context(weights, hidden, embedded, positions):
    projected = mtp_input(weights, hidden, embedded)
    normalized = rms(projected, weights.tensor("mtp.layers.0.input_layernorm.weight"))
    state = key_value(weights, "mtp.layers.0.self_attn.", normalized, positions)
    state = tuple(torch.cat((torch.zeros_like(value[:1]), value)) for value in state)
    return projected[-1:], state


def mtp_draft(weights, projected, state, positions, *, first):
    prefix = "mtp.layers.0."
    normalized = rms(projected, weights.tensor(prefix + "input_layernorm.weight"))
    start = state[0].shape[0] - int(first)
    output, state = attention(
        weights,
        prefix + "self_attn.",
        normalized,
        state,
        positions,
        start,
        query_only=first,
        first_token=1,
    )
    value = feed_forward(weights, prefix, projected + output)
    hidden = rms(value, weights.tensor("mtp.norm.weight"))
    return weights.linear(hidden, "lm_head").squeeze(0), hidden, state
