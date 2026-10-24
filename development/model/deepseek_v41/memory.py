"""CPU storage planner. Budget arithmetic does not establish runtime feasibility."""

import argparse
import json
from pathlib import Path

from oh_my_vllm.models.deepseek_v41.placement import (
    _size,
    _uint,
    weight_placements,
)

from .layout import tensor_layout

RESERVATIONS = ("graphs", "staging", "communication", "chunks", "media")
HOST_RESERVATIONS = ("kv_offload", "uploads", "transfer_leases")
DEFAULT_LIMITS = {
    "requests": 64,
    "context_tokens": 1048576,
    "draft_tokens": 5,
    "prefill_chunk_rows": 128,
    "index_tile_positions": 4096,
}
UNKNOWN_WORKSPACES = [
    "CUDA Graph pools, capture variants, and fixed-address metadata",
    "CUDA context, libraries, NCCL internals, and allocator alignment/fragmentation",
    "Native FP4/FP8 layouts, cache row padding, conversion scratch, and equivalence",
    "Exact chunk continuation, sparse top-k merge scratch, and kernel workspaces",
    "Media patch/frame concurrency, vision activations, and feature transfers",
    "Sampling/RNG/stop state, grammar, prefix copy-on-write, and overlapping leases",
    "Shared pinned registration, NUMA placement, and host startup/load peaks",
]


def _sum(values):
    result = 0
    for value in values:
        result = _uint(result + _uint(value))
    return result


def _keys(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError("Expected a configuration object")
    if set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError(f"Expected fields {required}; optional fields {optional}")


def _rank_bytes(values):
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("The topology has exactly four ranks")
    return [_uint(value) for value in values]


def _configuration(config, strategy):
    if strategy not in ("TP4", "TEP4", "DEP4"):
        raise ValueError("Strategy must be TP4, TEP4, or DEP4")
    _keys(
        config,
        (
            "model",
            "device_capacity_bytes",
            "device_unavailable_bytes",
            "host_pinned_capacity_bytes",
            "host_reservations",
            "reservations",
        ),
        ("limits",),
    )
    capacity = _rank_bytes(config["device_capacity_bytes"])
    unavailable = _rank_bytes(config["device_unavailable_bytes"])
    if any(c == 0 or u > c for c, u in zip(capacity, unavailable, strict=True)):
        raise ValueError("Device capacity must be positive and cover unavailable bytes")
    _uint(config["host_pinned_capacity_bytes"])
    _keys(config["host_reservations"], HOST_RESERVATIONS)
    for value in config["host_reservations"].values():
        _uint(value)
    _keys(config["reservations"], RESERVATIONS)
    for values in config["reservations"].values():
        _rank_bytes(values)
    supplied_limits = config.get("limits", {})
    _keys(supplied_limits, (), DEFAULT_LIMITS)
    limits = DEFAULT_LIMITS | supplied_limits
    for name, maximum in (
        ("requests", 64),
        ("context_tokens", 1048576),
        ("draft_tokens", 5),
        ("prefill_chunk_rows", 64 * 1048576),
        ("index_tile_positions", 1048576),
    ):
        value = _uint(limits[name])
        if value > maximum or (value == 0 and name != "draft_tokens"):
            raise ValueError(f"Unsupported limit: {name}")
    if limits["draft_tokens"] not in (0, 5):
        raise ValueError(
            "The source DSpark block has five rows; use zero to disable it"
        )
    limits["verification_rows"] = limits["draft_tokens"] + 1
    return limits, [c - u for c, u in zip(capacity, unavailable, strict=True)]


def _cache(text, limits, requests):
    context = limits["context_tokens"]
    drafts = limits["draft_tokens"]
    verify = limits["verification_rows"] if drafts else 0
    dim = text["head_dim"]
    window = min(text["sliding_window"], context)
    window_bytes = dim + dim // 32
    global_bytes = (
        dim // 2
        + dim // 16
        + text["index_head_dim"] // 2
        + text["index_head_dim"] // 32
    )
    sources = text["kv_source_layer_ids"]
    ratios = [text["compress_ratios"][layer] for layer in sources]
    global_rows = _size(requests, _sum(context // ratio for ratio in ratios))
    partial_slots = _sum(ratio for ratio in ratios if ratio > 1)
    cache = {}

    def add(name, rows, row_bytes):
        cache[name] = {
            "rows": rows,
            "row_bytes": row_bytes,
            "bytes": _size(rows, row_bytes),
        }

    add("global_sources", global_rows, global_bytes)
    cache["global_sources"]["main_kv_row_bytes"] = dim // 2 + dim // 16
    cache["global_sources"]["index_key_row_bytes"] = (
        text["index_head_dim"] // 2 + text["index_head_dim"] // 32
    )
    cache["global_sources"]["sources"] = [
        {"layer": layer, "ratio": ratio, "rows_per_request": context // ratio}
        for layer, ratio in zip(sources, ratios, strict=True)
    ]
    add(
        "target_windows",
        _size(requests, text["num_hidden_layers"], window),
        window_bytes,
    )
    stages = text["num_nextn_predict_layers"] if drafts else 0
    add("draft_windows", _size(requests, stages, window), window_bytes)
    add("partial_compressors", _size(requests, partial_slots), _size(dim, 2, 4))
    add(
        "partial_shadow",
        _size(requests, partial_slots if drafts else 0),
        _size(dim, 2, 4),
    )
    add(
        "target_window_shadow",
        _size(requests, text["num_hidden_layers"], verify),
        window_bytes,
    )
    add("draft_window_shadow", _size(requests, stages, verify), window_bytes)
    shadow_rows = _sum((verify + ratio - 1) // ratio for ratio in ratios)
    add("global_shadow", _size(requests, shadow_rows), global_bytes)
    add("draft_kv", _size(requests, stages, drafts), window_bytes)
    features = _size(len(text["dspark_target_layer_ids"]), text["hidden_size"], 2)
    add("target_features", requests if drafts else 0, features)
    add("feature_shadow", _size(requests, verify), features)
    add("proposal_probabilities", _size(requests, drafts), _size(text["vocab_size"], 4))
    add(
        "markov_features", _size(requests, drafts), _size(text["dspark_markov_rank"], 2)
    )
    add("sampling_token_history", _size(requests, context), 4)
    add("sampling_penalties", _size(requests, text["vocab_size"]), 4)
    add("engram_history", requests, 3 * (8 + 1) + 8)
    add("engram_history_shadow", requests if drafts else 0, 3 * (8 + 1) + 8)
    hash_columns = (text["engram_max_ngram_size"] - 1) * text["engram_n_heads"]
    add("engram_token_map", text["vocab_size"], 8)
    add(
        "engram_hash_constants",
        len(text["engram_layer_ids"]),
        (2 * hash_columns + 4) * 8,
    )
    add("rope_tables", _size(2, context, text["qk_rope_head_dim"] // 2), 8)
    return cache


def _reservations(config, text, limits, requests, strategy, rank):
    rows = max(limits["prefill_chunk_rows"], requests * limits["verification_rows"])
    group_rows = max(
        limits["prefill_chunk_rows"], limits["requests"] * limits["verification_rows"]
    )
    hidden = text["hidden_size"]
    hash_rows = (text["engram_max_ngram_size"] - 1) * text["engram_n_heads"]
    staging = _size(
        rows, hash_rows, text["engram_head_dim"] * 3 + text["engram_head_dim"] // 32 + 8
    )
    tile = min(limits["index_tile_positions"], limits["context_tokens"])
    heads = text["index_n_heads"] // (1 if strategy == "DEP4" else 4)
    query_heads = text["num_attention_heads"] // (1 if strategy == "DEP4" else 4)
    scores = _size(rows, tile, heads + 1, 4)
    selections = _size(
        rows,
        len(text["index_source_layer_ids"]) * text["index_topk"]
        + text["candidate_topk_blocks"],
        4,
    )
    residuals = _size(rows, text["hc_mult"], hidden, 2, 2)
    queries_outputs = _size(rows, query_heads, text["head_dim"], 2, 2)
    probabilities = _size(requests, limits["verification_rows"], text["vocab_size"], 4)
    expert_rows = rows if strategy == "TP4" else group_rows
    expert_buffers = _size(
        expert_rows, text["num_experts_per_tok"], text["moe_intermediate_size"], 8
    )
    chunks = _sum(
        (scores, selections, residuals, queries_outputs, probabilities, expert_buffers)
    )
    communication = (
        _size(rows, hidden, 4, 2)
        if strategy == "TP4"
        else _size(group_rows, text["num_experts_per_tok"], hidden * 2 + 16, 2)
    )
    if strategy != "DEP4":
        gathered_logits = _size(rows, text["vocab_size"], 4)
        local_logits = _size(rows, text["vocab_size"] // 4, 4)
        communication = max(communication, _sum((gathered_logits, local_logits)))
    minimum = {
        "graphs": 0,
        "staging": staging,
        "communication": communication,
        "chunks": chunks,
        "media": 0,
    }
    return {
        name: {
            "supplied_bytes": config["reservations"][name][rank],
            "minimum_bytes": minimum[name],
            "charged_bytes": max(config["reservations"][name][rank], minimum[name]),
        }
        for name in RESERVATIONS
    }


def plan_memory(config: dict, strategy: str) -> dict:
    """Return a JSON-safe four-rank plan without importing a GPU runtime.

    config contains model, capacities, explicit device/host reservations, and
    optional limits. See docs/models/deepseek_v41/memory.md for the input contract.
    """
    limits, available = _configuration(config, strategy)
    model = config["model"]
    tensors = tensor_layout(model)
    placements = weight_placements(model, tensors, strategy)
    host_tensors = {
        name: tensor
        for name, tensor in placements.items()
        if tensor["placement"] == "shared_pinned_host"
    }
    host_tables = _sum(
        t["stored_bytes"] for n, t in host_tensors.items() if n.endswith(".weight")
    )
    host_scales = _sum(
        t["stored_bytes"] for n, t in host_tensors.items() if n.endswith(".scale")
    )
    host_bytes = _sum(
        (host_tables, host_scales, _sum(config["host_reservations"].values()))
    )
    host_capacity = config["host_pinned_capacity_bytes"]
    ranks = []
    for rank in range(4):
        request_slots = (
            list(range(rank, limits["requests"], 4))
            if strategy == "DEP4"
            else list(range(limits["requests"]))
        )
        requests = len(request_slots)
        weights = _sum(t["rank_resident_bytes"][rank] for t in placements.values())
        cache = _cache(model["text_config"], limits, requests)
        cache_bytes = _sum(c["bytes"] for c in cache.values())
        reservations = _reservations(
            config, model["text_config"], limits, requests, strategy, rank
        )
        reservation_bytes = _sum(r["charged_bytes"] for r in reservations.values())
        accounted = _sum((weights, cache_bytes, reservation_bytes))
        ranks.append(
            {
                "rank": rank,
                "requests": requests,
                "request_slots": request_slots,
                "weight_bytes": weights,
                "cache": cache,
                "cache_bytes": cache_bytes,
                "reservations": reservations,
                "reservation_bytes": reservation_bytes,
                "accounted_bytes": accounted,
                "available_bytes": available[rank],
                "headroom_bytes": available[rank] - accounted,
                "fits_budget": accounted <= available[rank],
            }
        )
    return {
        "schema_version": 1,
        "strategy": strategy,
        "world_size": 4,
        "limits": limits,
        "tensor_count": len(tensors),
        "cache_format": "planned_packed_not_reference_bf16",
        "checkpoint_tensor_bytes": _sum(t["stored_bytes"] for t in placements.values()),
        "tensors": placements,
        "host": {
            "engram_table_bytes": host_tables,
            "engram_scale_bytes": host_scales,
            "engram_authoritative_copies": 1,
            "reader_ranks": [0, 1, 2, 3],
            "reservations": dict(config["host_reservations"]),
            "accounted_bytes": host_bytes,
            "available_bytes": host_capacity,
            "headroom_bytes": host_capacity - host_bytes,
            "fits_budget": host_bytes <= host_capacity,
        },
        "ranks": ranks,
        "fits_supplied_budgets": host_bytes <= host_capacity
        and all(r["fits_budget"] for r in ranks),
        "runtime_feasibility_verified": False,
        "unknown_workspaces": list(UNKNOWN_WORKSPACES),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--strategy", choices=("TP4", "TEP4", "DEP4"), required=True)
    args = parser.parse_args()
    try:
        plan = plan_memory(json.loads(args.config.read_text()), args.strategy)
    except (ValueError, OverflowError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(plan, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
