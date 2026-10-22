"""CPU storage planner. Budget arithmetic does not establish runtime feasibility."""

import argparse
import json
from pathlib import Path

from .layout import tensor_layout

UINT64_MAX = 2**64 - 1
DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}
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


def _uint(value):
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative integer byte size or count")
    if value > UINT64_MAX:
        raise OverflowError("Unsigned 64-bit size overflow")
    return value


def _size(*factors):
    value = 1
    for factor in factors:
        value = _uint(value * _uint(factor))
    return value


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


def _rule(rules, name, placement="replicated", axis=None, owner=None, dtype=None):
    if name in rules:
        raise ValueError(f"Duplicate placement rule: {name}")
    rules[name] = {
        "placement": placement,
        "shard_axis": axis,
        "owner_rank": owner,
        "promoted_dtype": dtype,
    }


def _linear_rules(
    rules, name, axis=None, owner=None, quant=None, dtype=None, bias=False
):
    placement = (
        "owned"
        if owner is not None
        else "sharded"
        if axis is not None
        else "replicated"
    )
    _rule(rules, f"{name}.weight", placement, axis, owner, dtype)
    if quant:
        scale_placement = "consumed_on_load" if dtype else placement
        _rule(rules, f"{name}.scale", scale_placement, axis, owner)
    if bias:
        _rule(rules, f"{name}.bias", placement, axis, owner)


def _attention_rules(rules, name, text, layer, dense_tp):
    output_axis = 0 if dense_tp else None
    input_axis = 1 if dense_tp else None
    _rule(
        rules, f"{name}.attn_sink", "sharded" if dense_tp else "replicated", output_axis
    )
    for norm in ("q_norm", "kv_norm"):
        _rule(rules, f"{name}.{norm}.weight")
    _linear_rules(rules, f"{name}.wq_a", quant="fp8")
    _linear_rules(rules, f"{name}.wq_b", output_axis, quant="fp8")
    _linear_rules(rules, f"{name}.wkv", quant="fp8")
    _linear_rules(rules, f"{name}.wo_a", output_axis, quant="fp8", dtype="BF16")
    _linear_rules(rules, f"{name}.wo_b", input_axis, quant="fp8")
    if layer in text["kv_source_layer_ids"]:
        compressor = f"{name}.compressor"
        _rule(rules, f"{compressor}.norm.weight")
        dtype = "F32" if text["compress_ratios"][layer] > 1 else None
        _linear_rules(rules, f"{compressor}.wkv", dtype=dtype)
        if dtype:
            _linear_rules(rules, f"{compressor}.wgate", dtype=dtype)
    if layer in text["index_source_layer_ids"]:
        indexer = f"{name}.indexer"
        _linear_rules(rules, f"{indexer}.wq_b", output_axis, quant="fp8")
        _linear_rules(rules, f"{indexer}.weights_proj", output_axis)
        if layer in text["kv_source_layer_ids"]:
            _linear_rules(rules, f"{indexer}.wk")
            _rule(rules, f"{indexer}.k_norm.weight")


def _block_rules(rules, name, text, layer, experts, strategy):
    dense_tp = strategy != "DEP4"
    for branch in ("attn", "ffn"):
        for suffix in (
            f"{branch}_norm.weight",
            f"hc_{branch}_fn",
            f"hc_{branch}_base",
            f"hc_{branch}_scale",
        ):
            _rule(rules, f"{name}.{suffix}")
    _attention_rules(rules, f"{name}.attn", text, layer, dense_tp)
    _linear_rules(rules, f"{name}.ffn.gate")
    for bias in ("bias", "bias_vl"):
        _rule(rules, f"{name}.ffn.gate.{bias}")
    for expert in range(experts):
        owner = None if strategy == "TP4" else expert // (experts // 4)
        for projection in ("w1", "w2", "w3"):
            axis = (1 if projection == "w2" else 0) if strategy == "TP4" else None
            _linear_rules(
                rules,
                f"{name}.ffn.experts.{expert}.{projection}",
                axis,
                owner,
                quant="fp4",
            )
    # Shared experts follow the dense layout, not routed-expert ownership.
    for projection in ("w1", "w2", "w3"):
        axis = (1 if projection == "w2" else 0) if dense_tp else None
        _linear_rules(
            rules, f"{name}.ffn.shared_experts.{projection}", axis, quant="fp8"
        )


def _placement_rules(model, strategy):
    text = model["text_config"]
    dense_axis = 0 if strategy != "DEP4" else None
    rules = {}
    _linear_rules(rules, "embed", dense_axis)
    _linear_rules(rules, "head", dense_axis, dtype="F32")
    _rule(rules, "norm.weight")
    for layer in range(text["num_hidden_layers"]):
        _block_rules(
            rules, f"layers.{layer}", text, layer, text["n_routed_experts"], strategy
        )
    for layer in text["engram_layer_ids"]:
        name = f"layers.{layer}.engram"
        for suffix in ("weight", "scale"):
            _rule(rules, f"{name}.embed.{suffix}", "shared_pinned_host")
        for weight in ("q_weight", "k_weight"):
            _rule(rules, f"{name}.{weight}")
        _linear_rules(rules, f"{name}.wkv", quant="fp8")
    _linear_rules(rules, "vision.patch_embed.proj", owner=0, bias=True)
    for layer in range(model["vision_config"]["num_hidden_layers"]):
        name = f"vision.blocks.{layer}"
        for norm in ("norm1", "norm2"):
            _rule(rules, f"{name}.{norm}.weight", "owned", owner=0)
        for projection in ("wqkv", "wo"):
            _linear_rules(rules, f"{name}.attn.{projection}", owner=0, bias=True)
        for projection in ("w1", "w2"):
            _linear_rules(rules, f"{name}.mlp.{projection}", owner=0)
    _rule(rules, "vision.norm.weight", "owned", owner=0)
    for projection in ("w1", "w2"):
        _linear_rules(rules, f"aligner.{projection}", owner=0, bias=True)
    for name in ("image_start", "image_end", "image_newline"):
        _rule(rules, name, "owned", owner=0)
    stages = text["num_nextn_predict_layers"]
    for stage in range(stages):
        _block_rules(
            rules,
            f"mtp.{stage}",
            text,
            text["num_hidden_layers"] + stage,
            text["dspark_n_routed_experts"],
            strategy,
        )
    _linear_rules(rules, "mtp.0.main_proj", quant="fp8")
    _rule(rules, "mtp.0.main_norm.weight")
    last = f"mtp.{stages - 1}"
    _rule(rules, f"{last}.norm.weight")
    _linear_rules(rules, f"{last}.markov_head.embed", dense_axis)
    _linear_rules(rules, f"{last}.markov_head.head", dense_axis, dtype="F32")
    _linear_rules(rules, f"{last}.confidence_head.proj", dtype="F32")
    return rules


def _validate_scales(tensors, rules):
    for name, tensor in tensors.items():
        if not name.endswith(".scale") or tensor["dtype"] != "F8_E8M0":
            continue
        weight = tensors[name.removesuffix(".scale") + ".weight"]
        rows, columns = weight["shape"]
        fp4 = weight["dtype"] == "I8"
        columns *= 2 if fp4 else 1
        host = rules[name]["placement"] == "shared_pinned_host"
        if fp4 or host:
            if columns % 32:
                raise ValueError(f"Unaligned per-row scale blocks: {name}")
            expected = [rows, columns // 32]
        else:
            expected = [(rows + 31) // 32, (columns + 31) // 32]
        if tensor["shape"] != expected:
            raise ValueError(f"Scale shape mismatch: {name}")
        axis = rules[name]["shard_axis"]
        logical = (rows, columns)
        if axis is not None and logical[axis] % (4 * (1 if fp4 and axis == 0 else 32)):
            raise ValueError(f"Shard splits a quantization block: {name}")


def _tensor_placement(tensor, rule):
    dtype = tensor["dtype"]
    shape = tensor["shape"]
    if (
        dtype not in DTYPE_BYTES
        or not shape
        or any(type(d) is not int or d <= 0 for d in shape)
    ):
        raise ValueError("Unsupported tensor dtype or shape")
    stored_bytes = _size(*shape, DTYPE_BYTES[dtype])
    placement = rule["placement"]
    axis = rule["shard_axis"]
    owner = rule["owner_rank"]
    local_shape = list(shape)
    if axis is not None:
        if axis >= len(shape) or shape[axis] % 4:
            raise ValueError("Tensor cannot be divided into four exact shards")
        local_shape[axis] //= 4
    if owner is not None and owner not in range(4):
        raise ValueError("Invalid tensor owner")
    local_source = _size(*local_shape, DTYPE_BYTES[dtype])
    resident_dtype = rule["promoted_dtype"] or dtype
    local_resident = _size(*local_shape, DTYPE_BYTES[resident_dtype])
    rank_source = [
        local_source if owner is None or owner == rank else 0 for rank in range(4)
    ]
    rank_resident = [
        local_resident if owner is None or owner == rank else 0 for rank in range(4)
    ]
    if placement == "shared_pinned_host":
        rank_source = [0] * 4
        rank_resident = [0] * 4
    elif placement == "consumed_on_load":
        rank_resident = [0] * 4
    return {
        "stored_dtype": dtype,
        "stored_shape": list(shape),
        "stored_bytes": stored_bytes,
        "placement": placement,
        "shard_axis": axis,
        "owner_rank": owner,
        "rank_shard_ranges": (
            [
                [rank * shape[axis] // 4, (rank + 1) * shape[axis] // 4]
                for rank in range(4)
            ]
            if axis is not None
            else None
        ),
        "resident_dtype": resident_dtype,
        "rank_source_bytes": rank_source,
        "rank_resident_bytes": rank_resident,
    }


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
    optional limits. See docs/deepseek-v41/memory.md for the input contract.
    """
    limits, available = _configuration(config, strategy)
    model = config["model"]
    tensors = tensor_layout(model)
    rules = _placement_rules(model, strategy)
    if tensors.keys() != rules.keys():
        raise ValueError(
            f"Unmapped tensors {tensors.keys() - rules.keys()}; "
            f"missing tensors {rules.keys() - tensors.keys()}"
        )
    _validate_scales(tensors, rules)
    placements = {
        name: _tensor_placement(tensor, rules[name]) for name, tensor in tensors.items()
    }
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
