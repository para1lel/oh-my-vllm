"""Conservative whole-HTTP floor for a fixed DeepSeek target algorithm.

This partial resource model does not describe every roofline bottleneck. It
is independent of the formal harness default. It validates v3.1 source traces,
but actual call boundaries and empirical collectives do not price the workload.
Unproved HBM retention, host transfers, and collectives receive no credit.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

from . import deepseek_accounting as accounting
from . import deepseek_accounting_math as arithmetic

VERSION = "deepseek-http-resource-floor-v4"
RATES = MappingProxyType(dict(arithmetic.RATES))
ASSUMPTIONS = MappingProxyType(
    {
        "algorithm": "released-target-matrices-formats-order-and-placement",
        "credit": "unique-manifest-prompt-and-accepted-output-rows",
        "overlap": "whole-http-resource-conservation",
        "query_score_placement": "fractional-global-pool-for-tp4-and-tep4",
        "routed_union": "assignment-count-lower-bound-without-expert-identities",
        "weight_reads": "union-once-diagnostic-no-unproved-reloads",
        "hbm_credit": "zero-without-reviewed-on-chip-retention-bound",
        "host_credit": "zero-without-mandatory-fresh-transfer-identities",
        "tp_tep_wire": "fixed-shards-full-bf16-rows-without-projection-recomputation",
        "embedding_wire": "zero-without-token-identity-reuse-proof",
        "wo_b_wire": "zero-unproved-full-fp32-allreduce",
        "routed_output_wire": "bf16-final-output-information-floor",
        "dep4_wire": "strict-zero-direct-owner-protocol-is-diagnostic-only",
        "candidate_block_size": 8,
        "candidate_pairs": "latest-block-is-mandatory-and-causally-partial",
        "greedy_head_wire": "zero-full-vocabulary-gather-is-not-required",
        "index_engram_wire": "zero-unproved-communication-alternatives",
        "empirical_latency": "excluded",
        "serial_chain": "excluded",
        "mtp_and_rejected_work": "excluded",
        "nonlinear_scalar_estimates": "excluded",
    }
)


@dataclass(frozen=True, slots=True)
class RankResources:
    fp8_flops: float = 0
    bf16_flops: float = 0
    fp32_flops: float = 0
    weight_union_lower_bytes: int = 0
    hbm_bytes: int = 0
    wire_sent_bytes: int = 0
    wire_received_bytes: int = 0
    direct_protocol_sent_bytes: int = 0
    direct_protocol_received_bytes: int = 0
    host_h2d_bytes: int = 0
    host_d2h_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    version: str
    seconds: float
    strategy: str
    acceptance: str
    calls: int
    accepted: int
    proposed: int
    rate_profile: str
    assumptions_sha256: str
    source_trace_version: str
    ranks: tuple[RankResources, ...]
    pooled_index_bf16_flops: float
    pooled_wire_sent_bytes: int
    pooled_wire_received_bytes: int
    expert_union_lower: tuple[tuple[int, ...], ...]


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def assumptions_digest():
    return _digest(dict(ASSUMPTIONS))


def rate_profile_digest(model=arithmetic.DEEPSEEK_V41_MODEL):
    return _digest([VERSION, dict(RATES), dict(ASSUMPTIONS), asdict(model)])


def _projection(rows, outputs, width, storage, strategy, *, sharded=False):
    if not rows:
        return arithmetic.Work(), 0
    shards = 4 if sharded and strategy != "dep4" else 1
    options = dict(
        storage=storage,
        shard_axis=0 if shards > 1 else None,
        shards=shards,
    )
    work = arithmetic.projection_work(
        rows, outputs, width, include_weights=False, **options
    )
    first = arithmetic.projection_work(1, outputs, width, **options)
    empty = arithmetic.projection_work(
        1, outputs, width, include_weights=False, **options
    )
    return work, int(first.hbm_bytes - empty.hbm_bytes)


def _dense(rows, logits, model, strategy):
    if not rows:
        return arithmetic.Work(), 0
    functions = (
        (arithmetic.attention_input_work, True),
        (arithmetic.attention_output_work, True),
        (arithmetic.shared_expert_work, True),
        (arithmetic.router_work, False),
        (arithmetic.engram_projection_work, False),
    )
    work = arithmetic.Work()
    weights = 0
    for operation, parallel in functions:
        options = (model, strategy) if parallel else (model,)
        extra = operation(rows, *options, include_weights=False)
        first = operation(1, *options)
        empty = operation(1, *options, include_weights=False)
        work = arithmetic.add_work(work, extra)
        weights += int(first.hbm_bytes - empty.hbm_bytes)
    hc_width = model.hc_mult * model.hidden_size
    work = arithmetic.add_work(
        work,
        arithmetic.Work(
            fp32_flops=4 * rows * model.hc_mix * hc_width * model.num_layers
        ),
    )
    weights += model.num_layers * 2 * model.hc_mix * hc_width * 4
    if logits:
        head = arithmetic.output_head_work(
            logits, model, strategy, include_weights=False
        )
        first = arithmetic.output_head_work(1, model, strategy)
        empty = arithmetic.output_head_work(1, model, strategy, include_weights=False)
        work = arithmetic.add_work(work, head)
        weights += int(first.hbm_bytes - empty.hbm_bytes)
    return work, weights


def _index(requests, layer, model, strategy):
    ratio = layer.compress_ratio
    queries = sum(max(0, length - ratio + 1) for length in requests)
    new_keys = (
        sum(length // ratio for length in requests) if layer.owns_compressed_kv else 0
    )
    query, query_weights = _projection(
        queries,
        model.index_heads * model.index_head_dim,
        model.q_lora_rank,
        "fp8",
        strategy,
        sharded=True,
    )
    head, head_weights = _projection(
        queries, model.index_heads, model.hidden_size, "bf16", strategy, sharded=True
    )
    key, key_weights = _projection(
        new_keys, model.index_head_dim, model.head_dim, "bf16", strategy
    )
    pairs = sum(
        arithmetic.candidate_pairs(length, ratio, model.candidate_positions)
        if layer.uses_candidate_mask
        else arithmetic.index_score_pairs(0, length, ratio)
        for length in requests
    )
    score = 2 * pairs * model.index_heads * model.index_head_dim
    return (
        arithmetic.add_work(query, head, key),
        query_weights + head_weights + key_weights,
        score,
    )


def _compressor(lengths, layer, model, strategy):
    ratio = layer.compress_ratio
    rows = sum(length // ratio * ratio for length in lengths)
    work, weights = _projection(
        rows,
        model.head_dim,
        model.hidden_size,
        "fp32" if ratio > 1 else "bf16",
        strategy,
    )
    if ratio > 1:
        work = arithmetic.scale_work(work, 2)
        weights *= 2
    return work, weights


def _target_work(requests, model, strategy):
    lengths = [row["input_tokens"] + row["output_tokens"] - 1 for row in requests]
    rows = sum(lengths)
    logits = sum(row["output_tokens"] for row in requests)
    work, weights = _dense(rows, logits, model, strategy)
    score_flops = 0
    for layer in arithmetic.layer_table(model):
        pairs = sum(
            arithmetic.window_pairs(0, length, model.window_size)
            + (
                arithmetic.compressed_pairs(
                    0, length, layer.compress_ratio, model.index_topk
                )
                if layer.compress_ratio
                else 0
            )
            for length in lengths
        )
        shards = 1 if strategy == "dep4" else 4
        work = arithmetic.add_work(
            work,
            arithmetic.Work(
                bf16_flops=4 * pairs * model.num_heads * model.head_dim / shards
            ),
        )
        if layer.owns_compressed_kv:
            compressor, weight_bytes = _compressor(lengths, layer, model, strategy)
            work = arithmetic.add_work(work, compressor)
            weights += weight_bytes
        if layer.owns_index_keys:
            projection, weight_bytes, score = _index(lengths, layer, model, strategy)
            work = arithmetic.add_work(work, projection)
            weights += weight_bytes
            score_flops += score
    if strategy == "dep4":
        work = arithmetic.add_work(work, arithmetic.Work(bf16_flops=score_flops))
        score_flops = 0
    return work, weights, score_flops


def _route_totals(calls, model, strategy):
    assignments = [[0] * 4 for _ in range(model.num_layers)]
    dispatch = [[[0] * 4 for _ in range(4)] for _ in range(model.num_layers)]
    for records in calls:
        sources = records if strategy == "dep4" else records[:1]
        for record in sources:
            source = record["rank"]
            for layer in range(model.num_layers):
                for destination in range(4):
                    assignments[layer][destination] += record["expert_assignments"][
                        layer
                    ][destination]
                    dispatch[layer][source][destination] += record["dispatch_rows"][
                        layer
                    ][destination]
    return assignments, dispatch


def _routed(assignments, rank, rows, model, strategy):
    work = arithmetic.Work()
    weights = 0
    unions = []
    for buckets in assignments:
        if strategy == "tp4":
            received = sum(buckets)
            union = sum((value + rows - 1) // rows for value in buckets)
        else:
            received = buckets[rank]
            union = (received + rows - 1) // rows
        extra = arithmetic.routed_expert_work(
            received, 0, model, strategy, include_weights=False
        )
        weight = arithmetic.routed_expert_work(0, union, model, strategy)
        work = arithmetic.add_work(work, extra)
        weights += int(weight.hbm_bytes)
        unions.append(union)
    return work, weights, tuple(unions)


def _dep4_wire(dispatch, model):
    sent = [0] * 4
    received = [0] * 4
    for layer in dispatch:
        for source in range(4):
            for destination in range(4):
                if source == destination:
                    continue
                count = layer[source][destination] * model.hidden_size
                sent[source] += count * 2
                received[destination] += count * 2
                sent[destination] += count * 4
                received[source] += count * 4
    return sent, received


def _pooled_wire(rows, model, strategy):
    if strategy == "dep4":
        return 0
    # Fixed MoE shards deliver complete BF16 rows without recomputing expert
    # projections. Embedding reuse and alternative wo_b reductions are unproved.
    routed = 3 * rows * model.hidden_size * 2 * model.num_layers
    if strategy == "tp4":
        routed += (
            3
            * rows
            * model.experts_per_token
            * model.moe_intermediate
            * 2
            * model.num_layers
        )
    return routed


def _seconds(ranks, pooled_score, pooled_wire):
    tensor = [
        row.fp8_flops / RATES["fp8_flops"] + row.bf16_flops / RATES["bf16_flops"]
        for row in ranks
    ]
    fixed = max(
        max(
            row.hbm_bytes / RATES["hbm_bytes"],
            tensor[rank],
            row.fp32_flops / RATES["fp32_flops"],
            max(row.wire_sent_bytes, row.wire_received_bytes) / RATES["nvlink_bytes"],
            row.host_h2d_bytes / RATES["host_bytes"],
            row.host_d2h_bytes / RATES["host_bytes"],
        )
        for rank, row in enumerate(ranks)
    )
    pooled_tensor = (sum(tensor) + pooled_score / RATES["bf16_flops"]) / 4
    return max(fixed, pooled_tensor, pooled_wire / (4 * RATES["nvlink_bytes"]))


def reference_trace(
    trace, *, strategy, model=arithmetic.DEEPSEEK_V41_MODEL, expected_binding=None
):
    """Validate a v3.1 source trace and price whole-HTTP necessary resources.

    HBM weight-union bounds are reported for review, but receive zero latency
    credit until a complete on-chip retention upper bound is authenticated.
    """
    target = accounting.validate_trace(
        trace, strategy=strategy, model=model, expected_binding=expected_binding
    )
    return reference_resources(target, model=model)


def reference_resources(
    target: accounting.TargetTrace, *, model=arithmetic.DEEPSEEK_V41_MODEL
):
    """Price whole-HTTP resources from an authenticated target trace."""
    strategy = target.binding["strategy"]
    calls = target.calls
    requests = target.manifest["requests"]
    rows = sum(row["input_tokens"] + row["output_tokens"] - 1 for row in requests)
    assignments, dispatch = _route_totals(calls, model, strategy)
    sent, received = _dep4_wire(dispatch, model)
    ranks, unions = [], []
    pooled_score = 0
    for rank in range(4):
        owned = (
            [row for row in requests if row["owner_rank"] == rank]
            if strategy == "dep4"
            else requests
        )
        dense, dense_weights, score = _target_work(owned, model, strategy)
        routed, expert_weights, union = _routed(
            assignments, rank, rows, model, strategy
        )
        work = arithmetic.add_work(dense, routed)
        ranks.append(
            RankResources(
                fp8_flops=work.fp8_flops,
                bf16_flops=work.bf16_flops,
                fp32_flops=work.fp32_flops + work.fp8_flops / 32,
                weight_union_lower_bytes=dense_weights + expert_weights,
                direct_protocol_sent_bytes=sent[rank],
                direct_protocol_received_bytes=received[rank],
            )
        )
        unions.append(union)
        if rank == 0:
            pooled_score = score
    wire = _pooled_wire(rows, model, strategy)
    return ReferenceResult(
        version=VERSION,
        seconds=_seconds(ranks, pooled_score, wire),
        strategy=strategy,
        acceptance="trace-resource-floor",
        calls=len(calls),
        accepted=target.binding["accepted"],
        proposed=target.binding["proposed"],
        rate_profile=rate_profile_digest(model),
        assumptions_sha256=assumptions_digest(),
        source_trace_version=target.binding["version"],
        ranks=tuple(ranks),
        pooled_index_bf16_flops=pooled_score,
        pooled_wire_sent_bytes=wire,
        pooled_wire_received_bytes=wire,
        expert_union_lower=tuple(unions),
    )


def reference_case(case, *, model=arithmetic.DEEPSEEK_V41_MODEL):
    """Return a favorable balanced-routing scale, not runtime acceptance."""
    trace = accounting.ideal_trace(case, model=model)
    result = reference_trace(trace, strategy=case["strategy"], model=model)
    return replace(result, acceptance="ideal-routing-resource-floor")
