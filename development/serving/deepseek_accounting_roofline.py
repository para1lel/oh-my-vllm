"""Whole-HTTP resource floor for the fixed target algorithm and placement."""

from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

from . import deepseek_accounting as source
from . import deepseek_accounting_floor as floor
from . import deepseek_accounting_math as arithmetic

VERSION = "deepseek-http-roofline-v6"
FAST_MEMORY_BYTES = 1 << 30
ASSUMPTIONS = MappingProxyType(
    {
        "scope": "fixed-target-algorithm-and-four-rank-placement-resource-floor",
        "arithmetic": "released-matrices-formats-k32-order-and-replicated-projections",
        "credit": "whole-http-prompt-and-accepted-output-no-observed-call-pricing",
        "rates": "dense-machine-peaks-no-empirical-latency",
        "overlap": "maximum-cross-row-and-cross-resource-overlap",
        "fast_memory": "at-most-1gib-all-on-chip-storage-per-b200-rank",
        "weights": "fixed-weight-shards-no-extra-copies-or-projection-recomputation",
        "weight_reuse": "whole-http-union-or-mandatory-dense-chain-take-maximum",
        "target_round": "ordinary-at-most-1-or-dspark5-at-most-6-new-target-rows",
        "history": "kv-fp4-g16-e4m3-288-index-fp4-g32-ue8m0-68-window-bf16-1024",
        "cache": "initially-empty-publish-full-encoded-history-without-recomputation",
        "cache_retention": "allow-1gib-unwritten-history-per-request",
        "history_reads": "zero-no-proof-of-mandatory-reloads-after-maximal-retention",
        "engram": "initially-host-resident-packed-264-byte-rows-full-token-reuse",
        "engram_reuse": "24-disjoint-columns-only-once-no-hash-identity-reloads",
        "tp_tep_wire": "fixed-moe-output-shards-complete-bf16-information-delivery",
        "dep4_wire": "direct-owner-bf16-dispatch-reverse-fp32-local-sum-no-compression",
        "wire_reuse": "no-cross-token-value-reuse-or-unmodeled-wire-compression",
        "index": "fractional-global-query-pool-for-tp4-and-tep4",
        "zero_wire": "embedding-wo-b-head-index-engram",
        "excluded": (
            "mtp-rejected-padding-copies-metadata-launch-scalar-estimates-s-over-2"
        ),
        "protocol": "ordered-logical-payload-diagnostics-not-a-serial-latency-floor",
    }
)


@dataclass(frozen=True, slots=True)
class RankResources:
    fp8_flops: float
    bf16_flops: float
    fp32_flops: float
    hbm_bytes: int
    wire_sent_bytes: int
    wire_received_bytes: int
    host_h2d_bytes: int
    host_d2h_bytes: int
    weight_union_lower_bytes: int
    dense_weight_bytes: int
    mandatory_target_rounds: int
    weight_read_lower_bytes: int
    cache_payload_bytes: int
    cache_write_lower_bytes: int


@dataclass(frozen=True, slots=True)
class ProtocolStage:
    name: str
    after: str
    before: str
    sent_bytes: tuple[float, ...]
    received_bytes: tuple[float, ...]


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
    pooled_host_h2d_bytes: int
    dependency_seconds: float
    protocol_stages: tuple[ProtocolStage, ...]


def assumptions_digest():
    return source.digest(dict(ASSUMPTIONS))


def rate_profile_digest(model=arithmetic.DEEPSEEK_V41_MODEL):
    return source.digest(
        [
            VERSION,
            dict(arithmetic.RATES),
            dict(ASSUMPTIONS),
            floor.assumptions_digest(),
            FAST_MEMORY_BYTES,
            asdict(model),
        ]
    )


def _owned(requests, rank, strategy):
    return (
        [row for row in requests if row["owner_rank"] == rank]
        if strategy == "dep4"
        else requests
    )


def _weight_reads(base, requests, model, strategy, mode):
    if not requests:
        return 0, 0, max(0, base.weight_union_lower_bytes - FAST_MEMORY_BYTES)
    _, dense = floor._dense(1, 1, model, strategy)
    limit = model.draft_tokens + 1 if mode == "dspark5" else 1
    rounds = max(
        1 + (row["output_tokens"] - 1 + limit - 1) // limit for row in requests
    )
    # Each interval on one request's dependency chain needs every dense matrix.
    # At most FAST_MEMORY_BYTES of those weights can survive from outside it.
    # Union and chain witnesses can describe the same reads; they must not add.
    reads = max(
        0,
        base.weight_union_lower_bytes - FAST_MEMORY_BYTES,
        rounds * max(0, dense - FAST_MEMORY_BYTES),
    )
    return dense, rounds, reads


def _cache_writes(requests, model):
    payload = writes = 0
    for row in requests:
        length = row["input_tokens"] + row["output_tokens"] - 1
        current = 0
        for layer in arithmetic.layer_table(model):
            if not layer.owns_compressed_kv:
                continue
            row_bytes = model.head_dim // 2 + model.head_dim // 16
            if layer.owns_index_keys:
                row_bytes += model.index_head_dim // 2 + model.index_head_dim // 32
            current += length // layer.compress_ratio * row_bytes
        payload += current
        writes += max(0, current - FAST_MEMORY_BYTES)
    return payload, writes


def _seconds(ranks, pooled_score, pooled_wire, pooled_host):
    tensor = [
        row.fp8_flops / arithmetic.RATES["fp8_flops"]
        + row.bf16_flops / arithmetic.RATES["bf16_flops"]
        for row in ranks
    ]
    fixed = max(
        max(
            tensor[rank],
            row.fp32_flops / arithmetic.RATES["fp32_flops"],
            row.hbm_bytes / arithmetic.RATES["hbm_bytes"],
            row.wire_sent_bytes / arithmetic.RATES["nvlink_bytes"],
            row.wire_received_bytes / arithmetic.RATES["nvlink_bytes"],
            row.host_h2d_bytes / arithmetic.RATES["host_bytes"],
            row.host_d2h_bytes / arithmetic.RATES["host_bytes"],
        )
        for rank, row in enumerate(ranks)
    )
    return max(
        fixed,
        (sum(tensor) + pooled_score / arithmetic.RATES["bf16_flops"]) / 4,
        pooled_wire / (4 * arithmetic.RATES["nvlink_bytes"]),
        pooled_host / (4 * arithmetic.RATES["host_bytes"]),
    )


def _expert_protocol(strategy, prefix, layer, model, rows, dispatch, collective, add):
    if strategy == "tp4":
        collective(
            f"{prefix}.expert.activation_gather",
            f"{prefix}.expert.up_gate",
            f"{prefix}.expert.down",
            [
                count * model.experts_per_token * model.moe_intermediate * 2
                for count in rows
            ],
            gather=True,
        )
        collective(
            f"{prefix}.expert.output_gather",
            f"{prefix}.expert.down",
            f"{prefix}.shared_expert",
            [count * model.hidden_size * 4 for count in rows],
            gather=True,
        )
    elif strategy == "tep4":
        collective(
            f"{prefix}.expert.reduce",
            f"{prefix}.expert.down",
            f"{prefix}.shared_expert",
            [count * model.hidden_size * 4 for count in rows],
        )
    else:
        matrix = dispatch[layer.layer_id]
        sent = [
            sum(matrix[rank][other] for other in range(4) if other != rank)
            for rank in range(4)
        ]
        received = [
            sum(matrix[other][rank] for other in range(4) if other != rank)
            for rank in range(4)
        ]
        add(
            f"{prefix}.dispatch",
            f"{prefix}.router",
            f"{prefix}.expert.up_gate",
            [count * model.hidden_size * 2 for count in sent],
            [count * model.hidden_size * 2 for count in received],
        )
        add(
            f"{prefix}.combine",
            f"{prefix}.expert.down",
            f"{prefix}.shared_expert",
            [count * model.hidden_size * 4 for count in received],
            [count * model.hidden_size * 4 for count in sent],
        )


def _index_protocol(owned, layer, model, prefix, collective):
    pairs = [
        sum(
            arithmetic.candidate_pairs(
                length, layer.compress_ratio, model.candidate_positions
            )
            if layer.uses_candidate_mask
            else arithmetic.index_score_pairs(0, length, layer.compress_ratio)
            for row in local
            for length in [row["input_tokens"] + row["output_tokens"] - 1]
        )
        for local in owned
    ]
    collective(
        f"{prefix}.index.reduce",
        f"{prefix}.index.score",
        f"{prefix}.sparse_attention",
        [count * 4 for count in pairs],
    )


def protocol_stages(target, *, model=arithmetic.DEEPSEEK_V41_MODEL):
    """Ordered, unpadded payload estimates; none adds serial floor seconds."""
    strategy = target.binding["strategy"]
    requests = target.manifest["requests"]
    owned = [_owned(requests, rank, strategy) for rank in range(4)]
    rows = [
        sum(row["input_tokens"] + row["output_tokens"] - 1 for row in local)
        for local in owned
    ]
    logits = [sum(row["output_tokens"] for row in local) for local in owned]
    _, dispatch = floor._route_totals(target.calls, model, strategy)
    stages = []

    def add(name, after, before, sent, received):
        stages.append(ProtocolStage(name, after, before, tuple(sent), tuple(received)))

    def collective(name, after, before, sizes, *, gather=False):
        operation = (
            arithmetic.all_gather_wire_bytes
            if gather
            else arithmetic.all_reduce_wire_bytes
        )
        wire = [operation(4, size) for size in sizes]
        add(name, after, before, wire, wire)

    if strategy != "dep4":
        collective(
            "embedding.reduce",
            "embedding",
            "layer.0",
            [count * model.hidden_size * 2 for count in rows],
        )
    for layer in arithmetic.layer_table(model):
        prefix = str(layer.layer_id)
        if strategy != "dep4" and layer.has_engram:
            collective(
                f"{prefix}.engram.reduce",
                f"{prefix}.engram.lookup",
                f"{prefix}.engram.projection",
                [
                    count * model.engram_columns * model.engram_head_dim * 2
                    for count in rows
                ],
            )
        if strategy != "dep4" and layer.owns_index_keys:
            _index_protocol(owned, layer, model, prefix, collective)
        if strategy != "dep4":
            collective(
                f"{prefix}.attention.reduce",
                f"{prefix}.attention.output",
                f"{prefix}.router",
                [count * model.hidden_size * 4 for count in rows],
            )
        _expert_protocol(
            strategy, prefix, layer, model, rows, dispatch, collective, add
        )
    if strategy != "dep4":
        collective(
            "head.gather",
            "head.projection",
            "sampling",
            [count * model.vocab_size * 4 for count in logits],
            gather=True,
        )
    return tuple(stages)


def reference_trace(
    trace, *, strategy, expected_binding=None, model=arithmetic.DEEPSEEK_V41_MODEL
):
    target = source.validate_trace(
        trace, strategy=strategy, expected_binding=expected_binding, model=model
    )
    return reference_resources(target, model=model)


def reference_resources(target, *, model=arithmetic.DEEPSEEK_V41_MODEL):
    base = floor.reference_resources(target, model=model)
    strategy = base.strategy
    requests = target.manifest["requests"]
    host_rows = model.engram_columns * len(model.engram_layers)
    host = host_rows * (model.engram_head_dim + model.engram_scale_bytes)
    ranks = []
    for rank, resource in enumerate(base.ranks):
        local = _owned(requests, rank, strategy)
        dense, rounds, reads = _weight_reads(
            resource, local, model, strategy, target.manifest["mode"]
        )
        payload, writes = _cache_writes(local, model)
        ranks.append(
            RankResources(
                resource.fp8_flops,
                resource.bf16_flops,
                resource.fp32_flops,
                reads + writes,
                resource.direct_protocol_sent_bytes if strategy == "dep4" else 0,
                resource.direct_protocol_received_bytes if strategy == "dep4" else 0,
                host if strategy == "dep4" and local else 0,
                0,
                resource.weight_union_lower_bytes,
                dense,
                rounds,
                reads,
                payload,
                writes,
            )
        )
    pooled_host = 0 if strategy == "dep4" else host
    seconds = _seconds(
        ranks, base.pooled_index_bf16_flops, base.pooled_wire_sent_bytes, pooled_host
    )
    return ReferenceResult(
        VERSION,
        seconds,
        strategy,
        "trace-roofline",
        base.calls,
        base.accepted,
        base.proposed,
        rate_profile_digest(model),
        assumptions_digest(),
        target.binding["version"],
        tuple(ranks),
        base.pooled_index_bf16_flops,
        base.pooled_wire_sent_bytes,
        base.pooled_wire_received_bytes,
        pooled_host,
        0.0,
        protocol_stages(target, model=model),
    )


def reference_case(case, *, model=arithmetic.DEEPSEEK_V41_MODEL):
    trace = source.ideal_trace(case, model=model)
    result = reference_trace(trace, strategy=case["strategy"], model=model)
    return replace(result, acceptance="ideal-routing-roofline")
