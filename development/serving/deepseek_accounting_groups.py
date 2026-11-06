from . import deepseek_accounting_math as arithmetic
from .deepseek_accounting_contract import HISTORY_CACHE_BYTES, received_assignments
from .deepseek_accounting_history import (
    _choose,
    _collect,
    _gather,
    _route_work,
    engram_alternatives,
    historical_reads,
    index_alternatives,
)
from .deepseek_accounting_index import choose_index_groups
from .deepseek_accounting_math import DEEPSEEK_V41_MODEL, WORLD_SIZE, Group, Work


def _layer_work(
    records,
    rank,
    count,
    layer,
    one,
    model,
    strategy,
    weights,
    touched_weights,
    expert_weights,
    kv_seen,
):
    record = records[rank]
    assignments = received_assignments(records, rank, layer.layer_id, strategy)
    touched = (
        record["touched_experts"][layer.layer_id]
        if touched_weights is None
        else touched_weights[rank][layer.layer_id]
    )
    work = arithmetic.routed_expert_work(
        assignments,
        touched,
        one,
        strategy,
        include_weights=(
            weights[rank]
            if expert_weights is None
            else expert_weights[rank][layer.layer_id]
        ),
    )
    local = {}
    if count:
        local = dict(arithmetic.local_work(count, one, strategy, weights[rank]))
    kv, _ = historical_reads(record["request_rows"], layer, model)
    window_bytes = (
        sum(min(row["start"], model.window_size - 1) for row in record["request_rows"])
        * model.head_dim
        * 2
    )
    if layer.compress_ratio:
        source = max(
            (source for source in model.kv_source_layers if source <= layer.layer_id),
            default=layer.layer_id,
        )
        shared = kv - window_bytes
        previous = kv_seen[rank].get(source, 0)
        kv -= min(shared, previous, HISTORY_CACHE_BYTES)
        kv_seen[rank][source] = max(previous, shared)
    pairs = sum(
        arithmetic.window_pairs(row["start"], row["count"], model.window_size)
        + (
            arithmetic.compressed_pairs(
                row["start"],
                row["count"],
                layer.compress_ratio,
                model.index_topk,
            )
            if layer.compress_ratio
            else 0
        )
        for row in record["request_rows"]
    )
    share = 1 if strategy == "dep4" else WORLD_SIZE
    local["sparse_attention"] = Work(
        hbm_bytes=kv,
        bf16_flops=4 * pairs * model.num_heads * model.head_dim / share,
    )
    local["routed_experts"] = work
    return local


def _append_communications(ranks, records, rank, count, layer, model, strategy):
    if strategy == "dep4":
        dense = Work()
        dispatch = _route_work(records, rank, layer.layer_id, model, 2)
        routed = _route_work(records, rank, layer.layer_id, model, 4)
    else:
        dense = _collect("all_reduce", "fp32", count * model.hidden_size * 4)
        dispatch = Work()
        if strategy == "tp4":
            routed = arithmetic.add_work(
                _gather(
                    "bf16",
                    count * model.experts_per_token * model.moe_intermediate * 2,
                ),
                _gather("fp32", count * model.hidden_size * 4),
            )
        else:
            routed = _collect("all_reduce", "fp32", count * model.hidden_size * 4)
    for name, value in (
        ("dense", dense),
        ("dispatch", dispatch),
        ("routed", routed),
    ):
        ranks[rank].append(Group(f"{layer.layer_id}.communication.{name}", value))


def _append_heads(ranks, records, model, strategy, head_weights):
    for rank, record in enumerate(records):
        logit_rows = sum(row["logit_rows"] for row in record["request_rows"])
        ranks[rank].append(
            Group(
                "output_head",
                arithmetic.output_head_work(
                    logit_rows, model, strategy, include_weights=head_weights[rank]
                )
                if logit_rows
                else Work(),
            )
        )
        ranks[rank].append(
            Group(
                "head.communication",
                _gather("fp32", logit_rows * model.vocab_size * 4)
                if strategy != "dep4" and logit_rows
                else Work(),
            )
        )


def _embedding_groups(totals, model, strategy):
    ranks = [
        [
            Group(
                "embedding",
                arithmetic.embedding_work(count, model, strategy) if count else Work(),
            )
        ]
        for count in totals
    ]
    for rank, count in enumerate(totals):
        ranks[rank].append(
            Group(
                "embedding.communication",
                _collect("all_reduce", "bf16", count * model.hidden_size * 2)
                if strategy != "dep4" and count
                else Work(),
            )
        )
    table = arithmetic.layer_table(model)
    return ranks, table


def call_groups(
    records,
    model=DEEPSEEK_V41_MODEL,
    strategy="tp4",
    *,
    weights=None,
    head_weights=None,
    touched_weights=None,
    expert_weights=None,
    key_weights=None,
):
    """Select consistent index and Engram placements across all ranks."""
    weights = [True] * WORLD_SIZE if weights is None else weights
    head_weights = weights if head_weights is None else head_weights
    totals = [sum(row["count"] for row in r["request_rows"]) for r in records]
    ranks, table = _embedding_groups(totals, model, strategy)
    kv_seen = [{} for _ in range(WORLD_SIZE)]
    for layer in table:
        one = arithmetic.layer_model(model, layer)
        for rank, count in enumerate(totals):
            local = _layer_work(
                records,
                rank,
                count,
                layer,
                one,
                model,
                strategy,
                weights,
                touched_weights,
                expert_weights,
                kv_seen,
            )
            for name in (
                "attention_input",
                "hyper_connection",
                "compressor",
                "sparse_attention",
                "attention_output",
                "router",
                "routed_experts",
                "shared_expert",
                "engram_projection",
                "engram_gate",
            ):
                ranks[rank].append(
                    Group(f"{layer.layer_id}.{name}", local.get(name, Work()))
                )
            _append_communications(ranks, records, rank, count, layer, model, strategy)
        engram_options = engram_alternatives(records, layer, model)
        engram = _choose(engram_options) if strategy != "dep4" else engram_options[0]
        for chosen in (engram,):
            for rank in range(WORLD_SIZE):
                ranks[rank].extend(
                    Group(f"{layer.layer_id}.{group.name}", group.work)
                    for group in chosen[rank]
                )
    _append_heads(ranks, records, model, strategy, head_weights)
    options = [
        index_alternatives(
            records,
            layer,
            model,
            strategy,
            weights,
            None
            if key_weights is None
            else [rank[layer.layer_id] for rank in key_weights],
        )
        for layer in table
    ]
    return choose_index_groups(ranks, options, records, model, strategy)
