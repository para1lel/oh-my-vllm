"""Independent CPU FP64 target attention, without runtime quantization.

Each request owns its per-layer AttentionState objects. Each forward pass owns
one shared dict across layers. Global IDs in topk_idxs are compressed-row IDs,
not offsets into a window. candidates is a tuple of sorted block-ID tensors:
a sparse block mask, not a query-by-context matrix. No function mutates an
input state or a previously published shared tensor.
"""

import math
from dataclasses import dataclass

import torch

from development.model.deepseek_v41.reference_math import (
    require_fp64,
    rms,
    rotary,
    sparse_attention,
)


@dataclass(frozen=True)
class AttentionState:
    layer: int
    length: int
    window_kv: torch.Tensor
    pending_values: torch.Tensor | None = None
    pending_gates: torch.Tensor | None = None
    compress_kv: torch.Tensor | None = None
    index_k: torch.Tensor | None = None


def _rotate(value, positions, config, ratio, *, inverse=False):
    width = config["qk_rope_head_dim"]
    if not width:
        return value.clone()
    scaling = config.get("rope_scaling") or {}
    tail = rotary(
        value[..., -width:],
        positions,
        base=config["compress_rope_theta"] if ratio else config["rope_theta"],
        inverse=inverse,
        original_context=scaling.get("original_max_position_embeddings", 0)
        if ratio
        else 0,
        factor=scaling.get("factor", 1),
        beta_fast=scaling.get("beta_fast", 32),
        beta_slow=scaling.get("beta_slow", 1),
    )
    return torch.cat((value[..., :-width], tail), dim=-1)


def _select_global(scores, topk):
    ids = torch.nonzero(torch.isfinite(scores), as_tuple=True)[0]
    order = torch.argsort(scores[ids], descending=True, stable=True)
    return ids[order[:topk]].sort().values


def _candidate_blocks(scores, *, topk, block_size):
    if scores.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=scores.device)
    padded = torch.nn.functional.pad(
        scores, (0, -scores.numel() % block_size), value=-torch.inf
    )
    finite_scores = torch.where(torch.isfinite(padded), padded, -torch.inf)
    block_scores = finite_scores.reshape(-1, block_size).amax(-1)
    reachable = torch.nonzero(torch.isfinite(block_scores), as_tuple=True)[0]
    if reachable.numel() == 0 or topk == 0:
        return reachable[:0]
    # Pin the newest reachable block, then break all other ties by global ID.
    block_scores[reachable[-1]] = torch.inf
    order = torch.argsort(block_scores[reachable], descending=True, stable=True)
    return reachable[order[:topk]].sort().values


def _index_scores(query, keys, head_weights, *, visible, candidates=None, block_size=1):
    scores = keys.new_full((visible,), -torch.inf)
    for first in range(0, visible, 4096):
        last = min(first + 4096, visible)
        ids = torch.arange(first, last, device=keys.device)
        if candidates is not None:
            valid_blocks = candidates[candidates >= 0]
            ids = ids[torch.isin(ids // block_size, valid_blocks)]
        if ids.numel():
            per_head = (query @ keys[ids].T).relu()
            scores[ids] = (per_head * head_weights[:, None]).sum(0)
    return scores


def _compress(weights, prefix, value, previous, config, ratio):
    width, eps = config["head_dim"], config["rms_norm_eps"]
    empty = value.new_empty((0, width))
    projected = weights.linear(value, prefix + ".compressor.wkv")
    old_kv = previous.compress_kv if previous is not None else empty
    old_index = (
        previous.index_k
        if previous is not None
        else value.new_empty((0, config["index_head_dim"]))
    )
    if ratio == 1:
        pooled, pending_values, pending_gates = projected, empty.clone(), empty.clone()
    else:
        gates = weights.linear(value, prefix + ".compressor.wgate")
        if previous is not None:
            projected = torch.cat((previous.pending_values, projected))
            gates = torch.cat((previous.pending_gates, gates))
        complete = projected.shape[0] // ratio * ratio
        pending_values = projected[complete:].clone()
        pending_gates = gates[complete:].clone()
        pooled = (
            projected[:complete].reshape(-1, ratio, width)
            * torch.softmax(gates[:complete].reshape(-1, ratio, width), dim=1)
        ).sum(1)
    latent = rms(pooled, weights.tensor(prefix + ".compressor.norm.weight"), eps=eps)
    group_positions = (
        torch.arange(
            old_kv.shape[0], old_kv.shape[0] + latent.shape[0], device=value.device
        )
        * ratio
    )
    index = rms(
        weights.linear(latent, prefix + ".indexer.wk"),
        weights.tensor(prefix + ".indexer.k_norm.weight"),
        eps=eps,
    )
    index = _rotate(index, group_positions, config, ratio)
    compressed = _rotate(latent, group_positions, config, ratio)
    return (
        torch.cat((old_kv, compressed)),
        torch.cat((old_index, index)),
        pending_values,
        pending_gates,
    )


def _check_shared(shared, positions, config, layer, ratio):
    owner = max(source for source in config["kv_source_layer_ids"] if source <= layer)
    if (
        shared.get("kv_source") != owner
        or shared.get("ratio") != ratio
        or not torch.equal(shared.get("query_positions", positions[:0]), positions)
    ):
        raise ValueError(
            "run the KV owner for the same query rows before its consumers"
        )
    return owner


def _select_rows(
    weights, prefix, value, query_latent, keys, positions, config, ratio, shared
):
    rows = value.shape[0]
    heads, width = config["index_n_heads"], config["index_head_dim"]
    query = weights.linear(query_latent, prefix + ".indexer.wq_b").reshape(
        rows, heads, width
    )
    query = _rotate(query, positions, config, ratio)
    head_weights = weights.linear(value, prefix + ".indexer.weights_proj") / math.sqrt(
        heads * width
    )
    layer = shared["index_source"]
    candidate_source = config.get("candidate_source_layer_id", -1)
    uses_candidates = 0 <= candidate_source < layer
    if uses_candidates and (
        shared.get("candidate_source") != candidate_source
        or shared.get("candidates") is None
    ):
        raise ValueError("run the candidate source before a reindex layer")
    selections, candidates = [], []
    for row, position in enumerate(positions.tolist()):
        scores = _index_scores(
            query[row],
            keys,
            head_weights[row],
            visible=(position + 1) // ratio,
            candidates=shared["candidates"][row] if uses_candidates else None,
            block_size=config.get("candidate_block_size", 1),
        )
        if layer == candidate_source:
            candidates.append(
                _candidate_blocks(
                    scores,
                    topk=config["candidate_topk_blocks"],
                    block_size=config["candidate_block_size"],
                )
            )
        selections.append(_select_global(scores, config["index_topk"]))
    shared["topk_idxs"] = tuple(selections)
    if layer == candidate_source:
        shared["candidate_source"] = layer
        shared["candidates"] = tuple(candidates)


def _global_attention(
    weights,
    prefix,
    value,
    query_latent,
    positions,
    previous,
    layer,
    config,
    ratio,
    shared,
):
    published = dict(shared)
    pending_values = pending_gates = compressed = index = None
    if layer in config["kv_source_layer_ids"]:
        compressed, index, pending_values, pending_gates = _compress(
            weights, prefix, value, previous, config, ratio
        )
        published.update(
            kv_source=layer,
            compress_kv=compressed,
            index_k=index,
            ratio=ratio,
            query_positions=positions.clone(),
            candidates=None,
            candidate_source=None,
            candidate_block_size=config.get("candidate_block_size", 1),
        )
    else:
        _check_shared(published, positions, config, layer, ratio)
    if published["compress_kv"].shape[0] != (int(positions[-1]) + 1) // ratio:
        raise ValueError("shared global KV length does not match the query rows")
    if layer in config["index_source_layer_ids"]:
        published["index_source"] = layer
        _select_rows(
            weights,
            prefix,
            value,
            query_latent,
            published["index_k"],
            positions,
            config,
            ratio,
            published,
        )
    else:
        source = max(
            source for source in config["index_source_layer_ids"] if source <= layer
        )
        if published.get("index_source") != source:
            raise ValueError("run the index source before its consumers")
    return published, compressed, index, pending_values, pending_gates


def forward(weights, prefix, value, *, layer, config, previous=None, start=0, shared):
    """Return target attention output and new accepted-state candidate.

    value is one contiguous CPU FP64 request chunk [rows, hidden]. config is
    the checkpoint text_config. previous must end at start. The caller decides
    whether to accept next_state; previous remains unchanged after this call.
    shared belongs to this request and this chunk only, with layers in order.
    """
    require_fp64(value)
    if value.device.type != "cpu" or value.ndim != 2 or value.shape[0] == 0:
        raise ValueError("target reference needs a nonempty two-dimensional CPU chunk")
    if (
        value.shape[1] != config["hidden_size"]
        or not 0 <= layer < config["num_hidden_layers"]
    ):
        raise ValueError("invalid target layer or hidden width")
    if (
        type(start) is not int
        or start < 0
        or (previous is None and start != 0)
        or (
            previous is not None
            and (previous.length != start or previous.layer != layer)
        )
    ):
        raise ValueError("previous layer state must end at the contiguous chunk start")
    rows, width = value.shape[0], config["head_dim"]
    heads, groups = config["num_attention_heads"], config["o_groups"]
    window, ratio = config["sliding_window"], config["compress_ratios"][layer]
    eps = config["rms_norm_eps"]
    positions = torch.arange(start, start + rows, device=value.device)
    query_latent = rms(
        weights.linear(value, prefix + ".wq_a"),
        weights.tensor(prefix + ".q_norm.weight"),
        eps=eps,
    )
    query = weights.linear(query_latent, prefix + ".wq_b").reshape(rows, heads, width)
    query = _rotate(query, positions, config, ratio)
    local = rms(
        weights.linear(value, prefix + ".wkv"),
        weights.tensor(prefix + ".kv_norm.weight"),
        eps=eps,
    )
    local = _rotate(local, positions, config, ratio)
    old_window = previous.window_kv if previous is not None else local[:0]
    window_start = start - old_window.shape[0]
    local = torch.cat((old_window, local))
    next_window = local[-window:].clone()
    published, compressed, index, pending_values, pending_gates = (
        shared,
        None,
        None,
        None,
        None,
    )
    if ratio:
        published, compressed, index, pending_values, pending_gates = _global_attention(
            weights,
            prefix,
            value,
            query_latent,
            positions,
            previous,
            layer,
            config,
            ratio,
            shared,
        )
        cache = torch.cat((local, published["compress_kv"]))
    else:
        cache = local
    indices = []
    for row, position in enumerate(positions.tolist()):
        selection = torch.arange(
            max(0, position + 1 - window) - window_start,
            position + 1 - window_start,
            device=value.device,
        )
        if ratio:
            global_ids = published["topk_idxs"][row]
            global_ids = global_ids[
                (global_ids >= 0) & (global_ids < (position + 1) // ratio)
            ]
            selection = torch.cat((selection, global_ids + local.shape[0]))
        indices.append(selection)
    output = sparse_attention(
        query, cache, weights.tensor(prefix + ".attn_sink"), indices, scale=width**-0.5
    )
    output = _rotate(output, positions, config, ratio, inverse=True)
    grouped = output.reshape(rows, groups, -1)
    projection = weights.tensor(prefix + ".wo_a.weight").reshape(
        groups, config["o_lora_rank"], -1
    )
    output = torch.einsum("sgd,grd->sgr", grouped, projection).flatten(1)
    del projection
    output = weights.linear(output, prefix + ".wo_b")
    next_state = AttentionState(
        layer,
        start + rows,
        next_window,
        pending_values,
        pending_gates,
        compressed,
        index,
    )
    if ratio:
        shared.update(published)
    return output, next_state
