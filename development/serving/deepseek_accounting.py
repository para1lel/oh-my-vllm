"""Accepted-output target reference and authenticated four-rank trace contract.

The collective profile is an estimate, not a physical latency proof."""

import math
from dataclasses import dataclass, replace

from . import deepseek_accounting_math as arithmetic
from .deepseek_accounting_contract import BINDING_FIELDS as BINDING_FIELDS
from .deepseek_accounting_contract import HISTORY_CACHE_BYTES as HISTORY_CACHE_BYTES
from .deepseek_accounting_contract import RULE_VERSION as RULE_VERSION
from .deepseek_accounting_contract import VERSION as VERSION
from .deepseek_accounting_contract import BindingError as BindingError
from .deepseek_accounting_contract import TargetTrace as TargetTrace
from .deepseek_accounting_contract import TraceError as TraceError
from .deepseek_accounting_contract import _manifest as _manifest
from .deepseek_accounting_contract import _matrix as _matrix
from .deepseek_accounting_contract import _request_rows as _request_rows
from .deepseek_accounting_contract import _uint as _uint
from .deepseek_accounting_contract import build_binding as build_binding
from .deepseek_accounting_contract import digest as digest
from .deepseek_accounting_contract import rate_profile_digest as rate_profile_digest
from .deepseek_accounting_contract import received_assignments as received_assignments
from .deepseek_accounting_contract import validate_binding as validate_binding
from .deepseek_accounting_contract import validate_trace as validate_trace
from .deepseek_accounting_groups import call_groups as call_groups
from .deepseek_accounting_history import HistoryGroup as HistoryGroup
from .deepseek_accounting_history import PooledHistoryGroup as PooledHistoryGroup
from .deepseek_accounting_history import _choose as _choose
from .deepseek_accounting_history import _collect as _collect
from .deepseek_accounting_history import _gather as _gather
from .deepseek_accounting_history import _route_work as _route_work
from .deepseek_accounting_history import engram_alternatives as engram_alternatives
from .deepseek_accounting_history import historical_reads as historical_reads
from .deepseek_accounting_history import index_alternatives as index_alternatives
from .deepseek_accounting_history import valid_index_pairs as valid_index_pairs
from .deepseek_accounting_index import _cached_index_step as _cached_index_step
from .deepseek_accounting_index import _credit_index as _credit_index
from .deepseek_accounting_index import _index_step as _index_step
from .deepseek_accounting_index import _resource_price as _resource_price
from .deepseek_accounting_index import _resource_times as _resource_times
from .deepseek_accounting_index import _resource_vector as _resource_vector
from .deepseek_accounting_index import candidate_handoff as candidate_handoff
from .deepseek_accounting_index import choose_index_groups as choose_index_groups
from .deepseek_accounting_index import overlap_seconds as overlap_seconds
from .deepseek_accounting_math import (
    DEEPSEEK_V41_MODEL,
    WORLD_SIZE,
    collective_profile_digest,
)


def _append_key_weights(record, model, prefill, rank, key_seen, key_weights):
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
            _append_key_weights(record, model, prefill, rank, key_seen, key_weights)
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
