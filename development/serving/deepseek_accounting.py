"""Accepted-output target reference and authenticated four-rank trace contract.

The collective profile is an estimate, not a physical latency proof.
"""

import hashlib
import json
import math
from dataclasses import dataclass, replace
from functools import lru_cache

from . import deepseek_accounting_math as arithmetic
from .deepseek_accounting_math import (
    DEEPSEEK_V41_MODEL,
    RATES,
    WORLD_SIZE,
    Group,
    Work,
    collective_profile_digest,
    critical_seconds,
    scale_accumulation_work,
)

VERSION = "deepseek-http-reference-v3.1"
RULE_VERSION = "k32-fp32-scale-accumulation-v3.1"
HISTORY_CACHE_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class HistoryGroup(Group):
    reusable_hbm_bytes: float = 0


@dataclass(frozen=True, slots=True)
class PooledHistoryGroup(HistoryGroup):
    pass


def digest(value):
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def rate_profile_digest():
    return digest(
        [VERSION, RULE_VERSION, HISTORY_CACHE_BYTES, arithmetic.rate_profile_digest()]
    )


def build_binding(**values):
    binding = {
        "version": VERSION,
        "rate_profile": rate_profile_digest(),
        **values,
    }
    validate_binding(binding)
    return binding


class BindingError(ValueError):
    """The accounting identity is absent or different."""


class TraceError(ValueError):
    """A call record does not satisfy the trace contract."""


@dataclass(frozen=True, slots=True)
class TargetTrace:
    """HTTP identity and ordered calls after trace validation."""

    binding: dict
    manifest: dict
    calls: tuple[tuple[dict, ...], ...]


BINDING_FIELDS = (
    "source",
    "native_modules",
    "checkpoint_config",
    "checkpoint_index",
    "tokenizer",
    "gpu_uuids",
    "strategy",
    "effective_strategy",
    "owner_plan",
    "sampling",
    "case_id",
    "rate_profile",
    "accepted",
    "proposed",
)


def validate_binding(binding, *, expected=None):
    if not isinstance(binding, dict) or binding.get("version") != VERSION:
        raise BindingError("accounting identity has a different version")
    if binding.get("rate_profile") != rate_profile_digest():
        raise BindingError("accounting rate profile differs")
    try:
        arithmetic.validate_execution_strategy(
            binding.get("strategy"), binding.get("effective_strategy")
        )
    except ValueError as error:
        raise BindingError(str(error)) from error
    for name in BINDING_FIELDS:
        if name not in binding:
            raise BindingError(f"accounting identity lacks {name}")
        value = binding[name]
        if value is None or value == "" or value == [] or value == {}:
            raise BindingError(f"accounting identity has an empty {name}")
    uuids = binding["gpu_uuids"]
    if (
        not isinstance(uuids, list)
        or len(uuids) != WORLD_SIZE
        or len(set(uuids)) != WORLD_SIZE
    ):
        raise BindingError("accounting identity requires four different GPU UUIDs")
    for name in ("accepted", "proposed"):
        if type(binding[name]) is not int or binding[name] < 0:
            raise BindingError(f"{name} must be a nonnegative integer")
    if binding["accepted"] > binding["proposed"]:
        raise BindingError("accepted tokens exceed proposed tokens")
    if (
        not isinstance(binding.get("manifest_sha256"), str)
        or len(binding["manifest_sha256"]) != 64
    ):
        raise BindingError("accounting identity lacks manifest_sha256")
    if expected is not None and binding != expected:
        raise BindingError("accounting identity changed")
    return binding


def _uint(value, name):
    if type(value) is not int or value < 0:
        raise TraceError(f"{name} must be a nonnegative integer")
    return value


def _manifest(manifest, model):
    if not isinstance(manifest, dict) or manifest.get("mode") not in (
        "ordinary",
        "dspark5",
    ):
        raise TraceError("manifest requires ordinary or dspark5 mode")
    requests = manifest.get("requests")
    if not isinstance(requests, list) or not requests:
        raise TraceError("manifest requires requests")
    ids = set()
    for ordinal, row in enumerate(requests):
        for name in ("ordinal", "id", "input_tokens", "output_tokens", "owner_rank"):
            _uint(row.get(name), name)
        if row["ordinal"] != ordinal or row["id"] in ids or row["id"] < 1:
            raise TraceError("manifest requires ordered, distinct request IDs")
        if row["owner_rank"] >= WORLD_SIZE:
            raise TraceError("manifest owner is outside the world")
        if min(row["input_tokens"], row["output_tokens"]) < 1:
            raise TraceError("manifest lengths must be positive")
        if row["input_tokens"] + row["output_tokens"] - 1 > model.max_context_tokens:
            raise TraceError("manifest exceeds model context")
        response_id = row.get("response_id", "")
        if (
            not isinstance(response_id, str)
            or not response_id.startswith("chatcmpl_")
            or response_id.rsplit("_", 1)[-1] != str(row["id"])
        ):
            raise TraceError("manifest response and trace IDs differ")
        ids.add(row["id"])
    return {row["id"]: row for row in requests}


def _matrix(value, layers, name):
    if not isinstance(value, list) or len(value) != layers:
        raise TraceError(f"{name} must have one row per layer")
    for row in value:
        if not isinstance(row, list) or len(row) != WORLD_SIZE:
            raise TraceError(f"{name} must have four columns")
        for count in row:
            _uint(count, name)


def _request_rows(record, requests, state, mode, strategy, model):
    rows = record.get("request_rows")
    if not isinstance(rows, list):
        raise TraceError("call requires request_rows")
    seen = set()
    accepted = proposed = 0
    for row in rows:
        for field in ("id", "start", "count", "accepted", "proposed", "logit_rows"):
            _uint(row.get(field), field)
        key = row["id"]
        if key not in requests or key in seen or row["count"] < 1:
            raise TraceError("call has an unknown, duplicate, or empty request")
        seen.add(key)
        request = requests[key]
        if strategy == "dep4" and request["owner_rank"] != record["rank"]:
            raise TraceError("DEP4 request differs from its manifest owner")
        computed, output = state[key]
        if row["start"] != computed:
            raise TraceError("recomputed or missing target rows")
        end = computed + row["count"]
        if record["phase"] == "prefill":
            if computed >= request["input_tokens"] or end > request["input_tokens"]:
                raise TraceError("prefill repeats or exceeds the prompt")
            logits = int(end == request["input_tokens"])
            if row["accepted"] or row["proposed"] or row["logit_rows"] != logits:
                raise TraceError("invalid prefill output")
            output += logits
        else:
            if computed < request["input_tokens"] or output < 1:
                raise TraceError("decode precedes complete prefill")
            if row["count"] != row["accepted"] + 1:
                raise TraceError("credited rows differ from accepted tokens plus one")
            drafts = model.draft_tokens if mode == "dspark5" else 0
            if not row["accepted"] <= row["proposed"] <= drafts:
                raise TraceError("invalid accepted or proposed draft count")
            if row["logit_rows"] != row["count"]:
                raise TraceError("decode logit rows differ from credited rows")
            output += row["count"]
        if output > request["output_tokens"]:
            raise TraceError("trace output exceeds HTTP usage")
        state[key] = (end, output)
        accepted += row["accepted"]
        proposed += row["proposed"]
    return accepted, proposed


def validate_trace(trace, *, strategy, model=DEEPSEEK_V41_MODEL, expected_binding=None):
    if not isinstance(trace, dict) or set(trace) != {"binding", "manifest", "calls"}:
        raise TraceError("trace requires binding, manifest, and calls")
    binding = validate_binding(trace["binding"], expected=expected_binding)
    if binding["strategy"] != strategy:
        raise TraceError("trace strategy differs from the requested strategy")
    if digest(trace["manifest"]) != binding["manifest_sha256"]:
        raise BindingError("HTTP manifest digest differs")
    requests = _manifest(trace["manifest"], model)
    if trace["manifest"]["mode"] == "ordinary" and strategy != "tp4":
        raise TraceError("ordinary cases use TP4")
    records = trace["calls"]
    if not isinstance(records, list) or not records:
        raise TraceError("trace requires call records")
    calls = {}
    for record in records:
        rank, call = (
            _uint(record.get("rank"), "rank"),
            _uint(record.get("call"), "call"),
        )
        if rank >= WORLD_SIZE or record.get("phase") not in ("prefill", "decode"):
            raise TraceError("invalid rank or phase")
        ranks = calls.setdefault(call, {})
        if rank in ranks:
            raise TraceError("trace repeats a rank call")
        ranks[rank] = record
        for name in ("expert_assignments", "dispatch_rows"):
            _matrix(record.get(name), model.num_layers, name)
        touched = record.get("touched_experts")
        if not isinstance(touched, list) or len(touched) != model.num_layers:
            raise TraceError("touched_experts must have one count per layer")
        for count in touched:
            _uint(count, "touched_experts")
    states = [{key: (0, 0) for key in requests} for _ in range(WORLD_SIZE)]
    accepted = proposed = 0
    for call in sorted(calls):
        ranks = calls[call]
        if set(ranks) != set(range(WORLD_SIZE)):
            raise TraceError("trace lacks a rank record for a call")
        if len({record["phase"] for record in ranks.values()}) != 1:
            raise TraceError("rank call phases differ")
        if strategy != "dep4":
            for rank in range(1, WORLD_SIZE):
                if any(
                    ranks[rank][key] != ranks[0][key]
                    for key in ("request_rows", "expert_assignments", "dispatch_rows")
                ):
                    raise TraceError("replicated ranks have different call counters")
        for rank, record in ranks.items():
            a, p = _request_rows(
                record,
                requests,
                states[rank],
                trace["manifest"]["mode"],
                strategy,
                model,
            )
            if strategy == "dep4" or rank == 0:
                accepted += a
                proposed += p
            token_rows = sum(row["count"] for row in record["request_rows"])
            for layer in range(model.num_layers):
                assignments = record["expert_assignments"][layer]
                dispatch = record["dispatch_rows"][layer]
                if sum(assignments) != token_rows * model.experts_per_token:
                    raise TraceError("expert assignments differ from credited rows")
                local_experts = model.routed_experts // WORLD_SIZE
                for destination in range(WORLD_SIZE):
                    count, distinct = assignments[destination], dispatch[destination]
                    if count > token_rows * min(local_experts, model.experts_per_token):
                        raise TraceError("expert destination exceeds its causal limit")
                    if strategy == "dep4":
                        if not (
                            math.ceil(
                                count
                                / max(1, min(local_experts, model.experts_per_token))
                            )
                            <= distinct
                            <= min(count, token_rows)
                        ):
                            raise TraceError(
                                "dispatch rows differ from distinct token destinations"
                            )
                    elif distinct:
                        raise TraceError("non-DEP4 trace has dispatch rows")
                received = received_assignments(
                    list(ranks.values()), rank, layer, strategy
                )
                bins = model.routed_experts if strategy == "tp4" else local_experts
                touched = record["touched_experts"][layer]
                global_tokens = (
                    sum(
                        sum(row["count"] for row in r["request_rows"])
                        for r in ranks.values()
                    )
                    if strategy == "dep4"
                    else token_rows
                )
                minimum_touched = math.ceil(received / max(1, global_tokens))
                if strategy == "dep4":
                    minimum_touched = max(
                        minimum_touched,
                        max(
                            math.ceil(
                                source["expert_assignments"][layer][rank]
                                / max(1, source["dispatch_rows"][layer][rank])
                            )
                            for source in ranks.values()
                        ),
                    )
                if not minimum_touched <= touched <= min(bins, received):
                    raise TraceError("touched expert count exceeds assignment bounds")
    for rank, state in enumerate(states):
        for key, request in requests.items():
            if strategy == "dep4" and request["owner_rank"] != rank:
                continue
            if state[key] != (
                request["input_tokens"] + request["output_tokens"] - 1,
                request["output_tokens"],
            ):
                raise TraceError("incomplete prefill or output differs from HTTP usage")
    if (accepted, proposed) != (binding["accepted"], binding["proposed"]):
        raise TraceError("trace counters differ from the accounting identity")
    return TargetTrace(
        binding,
        trace["manifest"],
        tuple(
            tuple(calls[call][rank] for rank in range(WORLD_SIZE))
            for call in sorted(calls)
        ),
    )


def received_assignments(records, rank, layer, strategy):
    if strategy == "tp4":
        return sum(records[rank]["expert_assignments"][layer])
    if strategy == "tep4":
        return records[rank]["expert_assignments"][layer][rank]
    return sum(record["expert_assignments"][layer][rank] for record in records)


def valid_index_pairs(rows, layer, model=DEEPSEEK_V41_MODEL):
    if not layer.owns_index_keys:
        return 0
    limit = model.candidate_positions if layer.uses_candidate_mask else None
    return sum(
        arithmetic.index_score_pairs(
            row["start"], row["count"], layer.compress_ratio, candidate_limit=limit
        )
        for row in rows
    )


def historical_reads(rows, layer, model=DEEPSEEK_V41_MODEL):
    """Minimum distinct historical rows, shared by all heads and queries of a call."""
    kv = keys = 0
    for row in rows:
        start, end = row["start"], row["start"] + row["count"]
        kv += min(start, model.window_size - 1)
        if layer.compress_ratio:
            ratio = layer.compress_ratio
            new = end // ratio - start // ratio
            kv += max(0, min(model.index_topk, end // ratio) - new)
            if layer.owns_index_keys:
                cap = model.candidate_positions if layer.uses_candidate_mask else end
                keys += max(0, min(cap, end // ratio) - new)
    return kv * model.head_dim * 2, keys * model.index_head_dim * 2


@lru_cache(maxsize=8192)
def _collect(kind, dtype, size, *, output=None):
    return arithmetic.collective_work(
        kind, dtype, size, size if output is None else output
    )


def _gather(dtype, output):
    return _collect("all_gather", dtype, math.ceil(output / WORLD_SIZE), output=output)


def _choose(alternatives):
    return min(alternatives, key=lambda ranks: critical_seconds(ranks).seconds)


def index_alternatives(records, layer, model, strategy, weights, key_weights=None):
    """Compare head and query placement with fixed dense and expert placement."""
    heads, queries = [], []
    for record, include_weights in zip(records, weights, strict=True):
        rows = record["request_rows"]
        count = sum(row["count"] for row in rows)
        if not count or not layer.owns_index_keys:
            heads.append(
                [
                    Group(name, Work())
                    for name in (
                        "index.projection",
                        "index.score",
                        "index.communication",
                    )
                ]
            )
            queries.append(
                [
                    Group(name, Work())
                    for name in (
                        "index.projection",
                        "index.query_gather",
                        "index.score",
                        "index.top_gather",
                    )
                ]
            )
            continue
        shards = 1 if strategy == "dep4" else WORLD_SIZE
        one = arithmetic.layer_model(model, layer)
        pairs = valid_index_pairs(rows, layer, model)
        new_keys = (
            sum(
                (row["start"] + row["count"]) // layer.compress_ratio
                - row["start"] // layer.compress_ratio
                for row in rows
            )
            if layer.owns_compressed_kv
            else 0
        )
        base = arithmetic.index_projection_work(
            count,
            new_keys,
            one,
            strategy,
            include_weights,
            None if key_weights is None else key_weights[record["rank"]],
        )
        _, keys = historical_reads(rows, layer, model)
        score = Work(
            bf16_flops=2 * pairs * model.index_heads * model.index_head_dim / shards,
            hbm_bytes=keys,
        )
        comm = _collect("all_reduce", "fp32", pairs * 4) if shards > 1 else Work()
        heads.append(
            [
                Group("index.projection", base),
                HistoryGroup("index.score", score, reusable_hbm_bytes=keys),
                Group("index.communication", comm),
            ]
        )
        if shards == 1:
            queries.append(heads[-1])
            continue
        query_pairs = pairs / WORLD_SIZE
        query_keys = keys / WORLD_SIZE
        query_bytes = count * model.index_heads * (model.index_head_dim + 1) * 2
        top_bytes = (
            sum(
                arithmetic.compressed_pairs(
                    row["start"], row["count"], layer.compress_ratio, model.index_topk
                )
                for row in rows
            )
            * 4
        )
        queries.append(
            [
                Group("index.projection", base),
                Group("index.query_gather", _gather("bf16", query_bytes)),
                PooledHistoryGroup(
                    "index.score",
                    Work(
                        bf16_flops=2
                        * query_pairs
                        * model.index_heads
                        * model.index_head_dim,
                        hbm_bytes=query_keys + query_bytes + top_bytes,
                    ),
                    reusable_hbm_bytes=query_keys,
                ),
                Group("index.top_gather", _gather("fp32", top_bytes)),
            ]
        )
    return heads, queries


def engram_alternatives(records, layer, model):
    full, sharded = [], []
    for record in records:
        count = sum(row["count"] for row in record["request_rows"])
        full.append(
            [Group("engram.lookup", arithmetic.engram_lookup_work(count, layer, model))]
        )
        sharded.append(
            [
                Group(
                    "engram.lookup",
                    arithmetic.engram_lookup_work(
                        count, layer, model, shards=WORLD_SIZE
                    ),
                ),
                Group(
                    "engram.reduce",
                    _collect(
                        "all_reduce",
                        "bf16",
                        count * model.engram_columns * model.engram_head_dim * 2,
                    )
                    if count and layer.has_engram
                    else Work(),
                ),
            ]
        )
    return full, sharded


def _route_work(records, rank, layer, model, element_bytes):
    matrix = [record["dispatch_rows"][layer] for record in records]
    wire = arithmetic.route_wire_pressure_bytes(
        matrix, rank, model.hidden_size * element_bytes
    )
    return Work(
        nvlink_bytes=wire,
        collective_seconds=(
            arithmetic.dep4_all_to_all_seconds(
                wire, "bf16" if element_bytes == 2 else "fp32"
            )
            if wire
            else 0
        ),
    )


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
    kv_seen = [{} for _ in range(WORLD_SIZE)]
    for layer in table:
        one = arithmetic.layer_model(model, layer)
        for rank, count in enumerate(totals):
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
                sum(
                    min(row["start"], model.window_size - 1)
                    for row in record["request_rows"]
                )
                * model.head_dim
                * 2
            )
            if layer.compress_ratio:
                source = max(
                    (
                        source
                        for source in model.kv_source_layers
                        if source <= layer.layer_id
                    ),
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
                            count
                            * model.experts_per_token
                            * model.moe_intermediate
                            * 2,
                        ),
                        _gather("fp32", count * model.hidden_size * 4),
                    )
                else:
                    routed = _collect(
                        "all_reduce", "fp32", count * model.hidden_size * 4
                    )
            for name, value in (
                ("dense", dense),
                ("dispatch", dispatch),
                ("routed", routed),
            ):
                ranks[rank].append(
                    Group(f"{layer.layer_id}.communication.{name}", value)
                )
        engram_options = engram_alternatives(records, layer, model)
        engram = _choose(engram_options) if strategy != "dep4" else engram_options[0]
        for chosen in (engram,):
            for rank in range(WORLD_SIZE):
                ranks[rank].extend(
                    Group(f"{layer.layer_id}.{group.name}", group.work)
                    for group in chosen[rank]
                )
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


def _resource_times(groups, *, pooled=False):
    groups = [g for g in groups if isinstance(g, PooledHistoryGroup) == pooled]
    work = arithmetic.add_work(
        *(arithmetic.scale_work(group.work, group.repetitions) for group in groups)
    )
    work = scale_accumulation_work(work)
    return (
        work.hbm_bytes / RATES["hbm_bytes"],
        work.fp8_flops / RATES["fp8_flops"] + work.bf16_flops / RATES["bf16_flops"],
        work.fp32_flops / RATES["fp32_flops"],
        work.nvlink_bytes / RATES["nvlink_bytes"],
        work.host_bytes / RATES["host_bytes"],
        work.collective_seconds,
    )


def _resource_vector(ranks):
    fixed = tuple(value for groups in ranks for value in _resource_times(groups))
    pooled = [0.0] * 6
    for groups in ranks:
        for index, value in enumerate(_resource_times(groups, pooled=True)):
            pooled[index] += value / WORLD_SIZE
    return (*fixed, *pooled)


def _resource_price(vector):
    fixed, pooled = vector[: WORLD_SIZE * 6], vector[WORLD_SIZE * 6 :]
    global_floor = max(
        sum(fixed[resource::6]) / WORLD_SIZE + pooled[resource] for resource in range(6)
    )
    return max(max(fixed), global_floor)


def candidate_handoff(records, model):
    # A query owner keeps the selected block IDs until a head-partitioned consumer.
    block_size = 8
    blocks = model.candidate_positions // block_size
    rows = records[0]["request_rows"]
    size = (
        sum(
            arithmetic.compressed_pairs(
                row["start"] + block_size - 1, row["count"], block_size, blocks
            )
            for row in rows
        )
        * 4
    )
    return [
        Group(
            "index.candidate_handoff",
            arithmetic.add_work(_gather("fp32", size), Work(hbm_bytes=size)),
        )
    ]


@lru_cache(maxsize=2048)
def _credit_index(groups, previous):
    credited = []
    seen = previous
    for group in groups:
        if isinstance(group, HistoryGroup):
            size = group.reusable_hbm_bytes
            credit = min(size, previous, HISTORY_CACHE_BYTES)
            work = replace(group.work, hbm_bytes=group.work.hbm_bytes - credit)
            credited.append(
                PooledHistoryGroup(group.name, work)
                if isinstance(group, PooledHistoryGroup)
                else Group(group.name, work)
            )
            seen = max(previous, size)
        else:
            credited.append(group)
    return tuple(credited), seen


def _index_step(options, previous):
    return _cached_index_step(tuple(tuple(groups) for groups in options), previous)


@lru_cache(maxsize=2048)
def _cached_index_step(options, previous):
    credited = [
        _credit_index(tuple(groups), old)
        for groups, old in zip(options, previous, strict=True)
    ]
    groups = [row[0] for row in credited]
    seconds = critical_seconds(groups).seconds
    resources = _resource_vector(groups)
    return groups, tuple(row[1] for row in credited), seconds, resources


def choose_index_groups(common, options, records, model, strategy):
    layers = [
        layer for layer in range(model.num_layers) if layer in model.index_source_layers
    ]
    if strategy == "dep4":
        seen = (0,) * WORLD_SIZE
        for layer in layers:
            if layer in model.kv_source_layers:
                seen = (0,) * WORLD_SIZE
            groups, seen, _, _ = _index_step(options[layer][0], seen)
            for rank in range(WORLD_SIZE):
                common[rank].extend(
                    PooledHistoryGroup(f"{layer}.{g.name}", g.work)
                    if isinstance(g, PooledHistoryGroup)
                    else Group(f"{layer}.{g.name}", g.work)
                    for g in groups[rank]
                )
        return common
    base_serial = critical_seconds(common).seconds
    base_resources = _resource_vector(common)
    states = [((), base_serial, base_resources, (0,) * WORLD_SIZE)]
    for layer in layers:
        next_states = []
        for choices, serial, total, seen in states:
            if layer in model.kv_source_layers:
                seen = (0,) * WORLD_SIZE
            for choice in range(2):
                _, current, seconds, resources = _index_step(
                    options[layer][choice], seen
                )
                next_states.append(
                    (
                        (*choices, choice),
                        serial + seconds,
                        tuple(a + b for a, b in zip(total, resources, strict=True)),
                        current,
                    )
                )
        states = next_states
    target_rows = sum(row["count"] for row in records[0]["request_rows"])
    source = (
        layers.index(model.candidate_source_layer)
        if model.candidate_source_layer in layers
        else None
    )
    handoff = candidate_handoff(records, model) if source is not None else []
    handoff_seconds = arithmetic.serial_seconds(handoff)
    handoff_resources = _resource_vector([handoff] * WORLD_SIZE)

    def price(state):
        choices, serial, resources, _ = state
        needs_handoff = (
            source is not None
            and choices[source] == 1
            and any(choice == 0 for choice in choices[source + 1 :])
        )
        if needs_handoff:
            serial += handoff_seconds
            resources = tuple(
                a + b for a, b in zip(resources, handoff_resources, strict=True)
            )
        return (
            min(serial, max(_resource_price(resources), serial / 2))
            if target_rows >= 2
            else serial
        )

    choices, _, _, _ = min(states, key=price)
    seen = (0,) * WORLD_SIZE
    for layer, choice in zip(layers, choices, strict=True):
        if layer in model.kv_source_layers:
            seen = (0,) * WORLD_SIZE
        groups, seen, _, _ = _index_step(options[layer][choice], seen)
        for rank in range(WORLD_SIZE):
            common[rank].extend(
                PooledHistoryGroup(f"{layer}.{group.name}", group.work)
                if isinstance(group, PooledHistoryGroup)
                else Group(f"{layer}.{group.name}", group.work)
                for group in groups[rank]
            )
    if (
        source is not None
        and choices[source] == 1
        and any(choice == 0 for choice in choices[source + 1 :])
    ):
        for groups in common:
            groups.extend(handoff)
    return common


def overlap_seconds(rank_groups, target_rows):
    """Two balanced microbatches: resource totals and half the dependency chain."""
    serial = critical_seconds(rank_groups).seconds
    if target_rows < 2:
        return serial
    resource = _resource_price(_resource_vector(rank_groups))
    return min(serial, max(resource, serial / 2))


def reference_trace(
    trace,
    *,
    strategy,
    model=DEEPSEEK_V41_MODEL,
    expected_binding=None,
):
    target = validate_trace(
        trace, strategy=strategy, model=model, expected_binding=expected_binding
    )
    calls = target.calls
    dense_seen, head_seen, key_seen = set(), set(), set()
    _collect.cache_clear()
    touched = [[0] * model.num_layers for _ in range(WORLD_SIZE)]
    total = 0.0
    for records in calls:
        prefill = records[0]["phase"] == "prefill"
        weights, head_weights, expert_weights, key_weights = [], [], [], []
        touched_credit = []
        for rank, record in enumerate(records):
            active = bool(record["request_rows"])
            logits = any(row["logit_rows"] for row in record["request_rows"])
            key_flags = []
            for layer in arithmetic.layer_table(model):
                new_keys = (
                    layer.owns_compressed_kv
                    and layer.owns_index_keys
                    and any(
                        (row["start"] + row["count"]) // layer.compress_ratio
                        > row["start"] // layer.compress_ratio
                        for row in record["request_rows"]
                    )
                )
                key_flags.append(
                    not prefill or (new_keys and (rank, layer.layer_id) not in key_seen)
                )
                if prefill and new_keys:
                    key_seen.add((rank, layer.layer_id))
            key_weights.append(key_flags)
            current = record["touched_experts"]
            delta = [
                max(0, value - prior)
                for value, prior in zip(current, touched[rank], strict=True)
            ]
            touched_credit.append(delta if prefill else current)
            expert_weights.append(
                [value > 0 for value in delta] if prefill else [True] * model.num_layers
            )
            if prefill:
                touched[rank] = [
                    max(value, prior)
                    for value, prior in zip(current, touched[rank], strict=True)
                ]
            weights.append(not prefill or (active and rank not in dense_seen))
            head_weights.append(not prefill or (logits and rank not in head_seen))
            if prefill and active:
                dense_seen.add(rank)
            if prefill and logits:
                head_seen.add(rank)
        groups = call_groups(
            records,
            model,
            strategy,
            weights=weights,
            head_weights=head_weights,
            touched_weights=touched_credit,
            expert_weights=expert_weights,
            key_weights=key_weights,
        )
        count = sum(
            row["count"]
            for record in (records if strategy == "dep4" else records[:1])
            for row in record["request_rows"]
        )
        total += overlap_seconds(groups, count)
    return ReferenceResult(
        version=VERSION,
        seconds=total,
        acceptance="trace",
        strategy=strategy,
        calls=len(calls),
        communication_model="lower_bound",
        profile_digest=collective_profile_digest(),
        communication_rule_version=RULE_VERSION,
        communication_is_modeled=strategy == "dep4",
        communication_is_measured=False,
    )


def _ideal_record(rank, call, phase, rows, global_rows, model, strategy):
    local = (
        rows
        if strategy != "dep4"
        else [row for row in rows if (row["id"] - 1) % WORLD_SIZE == rank]
    )
    count = sum(row["count"] for row in local)
    assignments = list(_route_row(count * model.experts_per_token))
    dispatch = (
        [min(count, a) for a in assignments] if strategy == "dep4" else [0] * WORLD_SIZE
    )
    received = (
        count * model.experts_per_token
        if strategy == "tp4"
        else _route_row(global_rows * model.experts_per_token)[rank]
    )
    bins = (
        model.routed_experts
        if strategy == "tp4"
        else model.routed_experts // WORLD_SIZE
    )
    return {
        "rank": rank,
        "call": call,
        "phase": phase,
        "request_rows": local,
        "expert_assignments": [assignments.copy() for _ in range(model.num_layers)],
        "dispatch_rows": [dispatch.copy() for _ in range(model.num_layers)],
        "touched_experts": [_expected_distinct(bins, received)] * model.num_layers,
    }


def ideal_trace(case, *, strategy=None, model=DEEPSEEK_V41_MODEL):
    strategy = strategy or case.get("strategy", "tp4")
    arithmetic.validate_execution_strategy(strategy, strategy)
    manifest = {
        "mode": case["mode"],
        "requests": [
            {
                "ordinal": index,
                "id": index + 1,
                "response_id": f"chatcmpl_{index + 1}",
                "input_tokens": case["input_tokens"],
                "output_tokens": case["output_tokens"],
                "owner_rank": index % WORLD_SIZE,
            }
            for index in range(case["batch"])
        ],
    }
    _manifest(manifest, model)
    calls = []
    start, remaining, call, accepted = 0, case["output_tokens"] - 1, 0, 0
    count = case["input_tokens"]
    while True:
        phase = "prefill" if call == 0 else "decode"
        a = 0 if call == 0 else count - 1
        rows = [
            {
                "id": index + 1,
                "start": start,
                "count": count,
                "accepted": a,
                "proposed": a,
                "logit_rows": 1 if call == 0 else count,
            }
            for index in range(case["batch"])
        ]
        for rank in range(WORLD_SIZE):
            calls.append(
                _ideal_record(
                    rank, call, phase, rows, case["batch"] * count, model, strategy
                )
            )
        if strategy == "dep4":
            records = calls[-WORLD_SIZE:]
            bins = model.routed_experts // WORLD_SIZE
            for rank, record in enumerate(records):
                n = received_assignments(records, rank, 0, strategy)
                record["touched_experts"] = [
                    _expected_distinct(bins, n)
                ] * model.num_layers
        accepted += a * case["batch"]
        start += count
        if call:
            remaining -= count
        if not remaining:
            break
        count = (
            min(model.draft_tokens + 1, remaining) if case["mode"] == "dspark5" else 1
        )
        call += 1
    binding = build_binding(
        source="ideal",
        native_modules=["ideal"],
        checkpoint_config="ideal",
        checkpoint_index="ideal",
        tokenizer="ideal",
        gpu_uuids=[f"ideal{rank}" for rank in range(WORLD_SIZE)],
        strategy=strategy,
        effective_strategy=strategy,
        owner_plan="manifest.owner_rank",
        sampling="greedy",
        case_id=str(case),
        accepted=accepted,
        proposed=accepted,
        manifest_sha256=digest(manifest),
    )
    return {"binding": binding, "manifest": manifest, "calls": calls}


def reference_case(case, *, strategy=None, model=DEEPSEEK_V41_MODEL):
    strategy = strategy or case.get("strategy", "tp4")
    if case.get("strategy", strategy) != strategy:
        raise ValueError("strategy differs from the case strategy")
    result = reference_trace(
        ideal_trace(case, strategy=strategy, model=model),
        strategy=strategy,
        model=model,
    )
    return replace(result, acceptance="ideal")


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    """The reference seconds and calculation identity for one case or trace."""

    version: str
    seconds: float
    acceptance: str
    strategy: str
    calls: int
    communication_model: str
    profile_digest: str | None
    communication_rule_version: str
    communication_is_modeled: bool
    communication_is_measured: bool


def efficiency_percent(reference_seconds, http_seconds):
    """Return the reference seconds as a percentage of the full HTTP seconds."""
    if not math.isfinite(http_seconds) or http_seconds <= 0:
        raise ValueError("full HTTP seconds must be finite and positive")
    if not math.isfinite(reference_seconds) or reference_seconds < 0:
        raise ValueError("reference seconds must be finite and nonnegative")
    return 100 * reference_seconds / http_seconds


def _route_row(assignments):
    base, remainder = divmod(assignments, WORLD_SIZE)
    return tuple(base + (destination < remainder) for destination in range(WORLD_SIZE))


def _expected_distinct(bins, draws):
    if draws <= 0:
        return 0
    return min(bins, round(bins * (1 - (1 - 1 / bins) ** draws)))
