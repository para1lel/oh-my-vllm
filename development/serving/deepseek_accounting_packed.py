"""Packed-history and collective protocol estimate, not a physical lower bound."""

from dataclasses import asdict, dataclass
from types import MappingProxyType

from . import deepseek_accounting as source
from . import deepseek_accounting_floor as floor
from . import deepseek_accounting_math as arithmetic
from .deepseek_accounting_packed_stages import _append_experts as _append_experts
from .deepseek_accounting_packed_stages import _append_index as _append_index
from .deepseek_accounting_packed_stages import _append_layer as _append_layer
from .deepseek_accounting_packed_stages import _append_sparse as _append_sparse
from .deepseek_accounting_packed_stages import _expert_projection as _expert_projection
from .deepseek_accounting_packed_stages import call_groups as call_groups
from .deepseek_accounting_packed_stages import collective_bytes as collective_bytes
from .deepseek_accounting_packed_stages import directed_bytes as directed_bytes
from .deepseek_accounting_packed_stages import history_work as history_work
from .deepseek_accounting_packed_stages import packed_row_bytes as packed_row_bytes

VERSION = "deepseek-http-protocol-estimate-v5"


ASSUMPTIONS = MappingProxyType(
    {
        "scope": "packed-aware-project-reference-not-universal-physical-minimum",
        "history": "bf16-window-fp4-g16-e4m3-kv-fp4-g32-ue8m0-index",
        "history_reads": "one-union-per-consumer-no-cross-consumer-retention-credit",
        "index_workspace": "decode-write-once-per-physical-source-per-call",
        "weights": "stream-per-call-no-unproved-inter-call-retention",
        "communication": "fixed-ring-volume-and-direct-owner-bf16-fp32-payload",
        "metadata": "counts-ids-weights-are-protocol-diagnostics-not-reference-credit",
        "dependencies": "ordered-stages-resource-and-two-microbatch-relaxation",
        "tensor": "fp8-and-bf16-share-resource-k32-fp32-accumulation",
        "latency": "wire-bandwidth-only-no-empirical-nccl-profile",
        "pending": "kernel-padding-cache-hit-rates-launch-latency-overlap-and-mtp-cost",
    }
)


@dataclass(frozen=True, slots=True)
class RankResources:
    work: arithmetic.Work
    sent_bytes: int
    received_bytes: int
    history_union_bytes: int
    protocol_metadata_pressure_bytes: float


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    version: str
    seconds: float
    strategy: str
    acceptance: str
    source_trace_version: str
    rate_profile: str
    assumptions_sha256: str
    ranks: tuple[RankResources, ...]
    call_seconds: tuple[float, ...]
    conditional_partial_floor_seconds: float
    protocol_serial_seconds: float


def assumptions_digest():
    return source.digest(dict(ASSUMPTIONS))


def rate_profile_digest(model=arithmetic.DEEPSEEK_V41_MODEL):
    return source.digest(
        [VERSION, dict(arithmetic.RATES), dict(ASSUMPTIONS), asdict(model)]
    )


def reference_trace(
    trace, *, strategy, expected_binding=None, model=arithmetic.DEEPSEEK_V41_MODEL
):
    target = source.validate_trace(
        trace, strategy=strategy, model=model, expected_binding=expected_binding
    )
    calls = target.calls
    totals = [arithmetic.Work() for _ in range(4)]
    sent = [0] * 4
    received = [0] * 4
    unions = [0] * 4
    seconds = []
    metadata = [0.0] * 4
    protocol_serial = 0.0
    for records in calls:
        ranks, directions, unique = call_groups(records, strategy=strategy, model=model)
        protocol_serial += arithmetic.critical_seconds(ranks).seconds
        credited = [
            [
                group
                for group in groups
                if not group.name.endswith(
                    (".dispatch.counts", ".dispatch.routing_metadata")
                )
            ]
            for groups in ranks
        ]
        rows = sum(
            row["count"]
            for record in (records if strategy == "dep4" else records[:1])
            for row in record["request_rows"]
        )
        seconds.append(source.overlap_seconds(credited, rows))
        for rank, groups in enumerate(ranks):
            totals[rank] = arithmetic.add_work(
                totals[rank],
                *(
                    arithmetic.scale_accumulation_work(group.work)
                    for group in credited[rank]
                ),
            )
            sent[rank] += directions[rank][0]
            received[rank] += directions[rank][1]
            unions[rank] += unique[rank]
            metadata[rank] += sum(
                group.work.nvlink_bytes
                for group in groups
                if group.name.endswith(
                    (".dispatch.counts", ".dispatch.routing_metadata")
                )
            )
    partial = floor.reference_resources(target, model=model)
    return ReferenceResult(
        VERSION,
        sum(seconds),
        strategy,
        "trace-project-estimate",
        trace["binding"]["version"],
        rate_profile_digest(model),
        assumptions_digest(),
        tuple(
            RankResources(
                totals[rank], sent[rank], received[rank], unions[rank], metadata[rank]
            )
            for rank in range(4)
        ),
        tuple(seconds),
        partial.seconds,
        protocol_serial,
    )
