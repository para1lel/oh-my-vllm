"""Rank identities and fixed device storage for credited target routes."""

import json
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

from ...operators.runtime.native import (
    cache_directory,
    file_sha256,
    source_fingerprint,
)
from ...source import source_identity
from .worker_limits import DEFAULT_DECODE_GRAPHS

NATIVE_MODULES = (
    "device",
    "paged",
    "activation",
    "hyperconnection",
    "projection",
    "routing",
    "sparse_attention",
    "transform",
)


def rank_trace_path(value, rank):
    if "{rank}" in value:
        return value.format(rank=rank)
    if rank == 0:
        return value
    path = Path(value)
    return str(path.with_name(f"{path.stem}-rank{rank}{path.suffix}"))


def execution_modes(
    strategy=None,
    *,
    batch_decode=None,
    decode_graph=None,
    decode_graph_limit=None,
    proposal_graph=None,
):
    from .kernel import native_sparse_graph_enabled

    strategy = (
        strategy or os.environ.get("OH_MY_VLLM_DEEPSEEK_STRATEGY", "tep4")
    ).lower()
    if batch_decode is None:
        batch_decode = os.environ.get("OH_MY_VLLM_DEEPSEEK_BATCH_DECODE") == "1"
    if decode_graph is None:
        decode_graph = (
            os.environ.get("OH_MY_VLLM_DEEPSEEK_DECODE_GRAPH", "1") != "0"
            and strategy == "tp4"
            and native_sparse_graph_enabled()
        )
    if decode_graph_limit is None:
        decode_graph_limit = int(
            os.environ.get("OH_MY_VLLM_DEEPSEEK_DECODE_GRAPHS", DEFAULT_DECODE_GRAPHS)
        )
    if type(decode_graph_limit) is not int or decode_graph_limit < 0:
        raise ValueError("DeepSeek decode graph limit must be nonnegative")
    if proposal_graph is None:
        proposal_graph = (
            decode_graph and os.environ.get("OH_MY_VLLM_DEEPSEEK_MTP_GRAPH", "0") == "1"
        )
    return {
        "batch_decode": batch_decode,
        "decode_graph": decode_graph,
        "decode_graph_limit": decode_graph_limit,
        "proposal_graph": proposal_graph,
    }


def runtime_identity(
    rank,
    *,
    strategy=None,
    batch_decode=None,
    decode_graph=None,
    decode_graph_limit=None,
    proposal_graph=None,
):
    modules = {}
    for name in NATIVE_MODULES:
        metadata = json.loads((cache_directory() / f"{name}.json").read_text())
        if (
            metadata["source_fingerprint"] != source_fingerprint(name)
            or file_sha256(metadata["library"]) != metadata["binary_id"]
        ):
            raise RuntimeError(f"DeepSeek trace native identity differs: {name}")
        modules[name] = {
            "source": metadata["source_id"],
            "binary": metadata["binary_id"],
        }
    source = os.environ.get("OH_MY_VLLM_SOURCE_COMMIT")
    if source is None:
        source = subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[4]),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
    gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not gpu_uuid.startswith("GPU-"):
        gpu_uuid = f"GPU-{gpu_uuid}"
    return {
        "rank": rank,
        "gpu_uuid": gpu_uuid,
        "source_commit": source,
        "source_digest": source_identity(),
        "native_modules": modules,
        "execution_modes": execution_modes(
            strategy,
            batch_decode=batch_decode,
            decode_graph=decode_graph,
            decode_graph_limit=decode_graph_limit,
            proposal_graph=proposal_graph,
        ),
    }


def target_call_id(worker, fallback):
    provider = getattr(worker, "target_call_id_provider", None)
    call = fallback if provider is None else provider()
    if type(call) is not int or call < 0:
        raise ValueError("target call ID must be a nonnegative integer")
    return call


def record_accounting_call(worker, *, call, phase, request_rows, counters):
    worker.trace.record(
        "accounting_call",
        rank=worker.rank,
        call=call,
        phase=phase,
        request_rows=request_rows,
        **counters,
    )
    if getattr(worker, "strategy", None) != "dep4":
        if not hasattr(worker, "target_trace_frontiers"):
            worker.target_trace_frontiers = {}
        for row in request_rows:
            worker.target_trace_frontiers[row["id"]] = row["start"] + row["count"]


class TargetTrace:
    def __init__(self, worker, requests, *, records=None):
        self.worker = worker
        self.requests = tuple(requests)
        self.records = [] if records is None else records
        if not hasattr(worker, "target_trace_frontiers"):
            worker.target_trace_frontiers = {}

    @classmethod
    def for_requests(cls, worker, requests):
        if (
            not getattr(worker, "routing_trace_enabled", False)
            or worker.strategy == "dep4"
        ):
            return None
        return cls(worker, requests)

    def subset(self, first, last):
        return TargetTrace(self.worker, self.requests[first:last], records=self.records)

    def routes(self, length):
        return RoutingTrace(
            len(self.worker.model.layers),
            len(self.requests) * length,
            rank=self.worker.rank,
            strategy=self.worker.strategy,
            device=self.worker.device,
        )

    def append(self, routes, start, length, computes_logits, *, verification=False):
        worker = self.worker
        worker.target_call = getattr(worker, "target_call", 0) + 1
        call = target_call_id(worker, worker.target_call)
        self.records.append(
            (call, self.requests, routes, start, length, computes_logits, verification)
        )
        if not verification:
            self.finish()

    def _credit(
        self,
        request_id,
        prompt,
        start,
        length,
        logits,
        verification,
        accepted,
        proposed,
    ):
        first = max(start, self.worker.target_trace_frontiers.get(request_id, 0))
        end = start + length
        if first < min(end, prompt):
            if end > prompt:
                return "mixed", None
            return "prefill", dict(
                id=request_id,
                start=first,
                count=min(end, prompt) - first,
                accepted=0,
                proposed=0,
                logit_rows=int(logits and end == prompt),
            )
        if not logits:
            return "decode", None
        if verification:
            prediction_start = end - proposed - 1
            end = prediction_start + accepted + 1
            first = max(first, prediction_start)
            drafts = max(0, proposed - (first - prediction_start))
        else:
            first, drafts = max(first, end - 1), 0
        if first >= end:
            return "decode", None
        return "decode", dict(
            id=request_id,
            start=first,
            count=end - first,
            accepted=end - first - 1,
            proposed=drafts,
            logit_rows=end - first,
        )

    def finish(self, *, accepted=0, proposed=0):
        for call, requests, routes, start, length, logits, verification in self.records:
            rows, phases, errors = [], set(), []
            mask = torch.zeros(
                (len(requests), length), dtype=torch.bool, device=self.worker.device
            )
            for index, (request_id, prompt) in enumerate(requests):
                phase, row = self._credit(
                    request_id,
                    prompt,
                    start,
                    length,
                    logits,
                    verification,
                    accepted,
                    proposed,
                )
                if phase == "mixed":
                    errors.append(f"request {request_id} crosses the prompt boundary")
                if row is not None:
                    rows.append(row)
                    phases.add(phase)
                    first = row["start"] - start
                    mask[index, first : first + row["count"]] = True
            if len(phases) > 1:
                errors.append("mixed prefill and decode requests")
            if errors:
                self.worker.trace.record(
                    "accounting_error",
                    rank=self.worker.rank,
                    call=call,
                    reason="; ".join(errors),
                )
                continue
            record_accounting_call(
                self.worker,
                call=call,
                phase="prefill" if "prefill" in phases else "decode",
                request_rows=rows,
                counters=routes.counters(mask.flatten()),
            )
        self.records.clear()


def dep4_expert_union(presence):
    if not dist.is_initialized() or dist.get_world_size() != 4:
        raise ValueError("DEP4 expert presence needs a four-rank process group")
    dist.all_reduce(presence, op=dist.ReduceOp.MAX)


class RoutingTrace:
    def __init__(self, layers, rows, *, rank, strategy, device):
        if layers < 1 or rows < 1 or rank not in range(4):
            raise ValueError("invalid DeepSeek routing trace dimensions")
        if strategy not in ("tp4", "tep4", "dep4"):
            raise ValueError("invalid DeepSeek routing trace strategy")
        self.rank = rank
        self.strategy = strategy
        self.indices = torch.full(
            (layers, rows, 6), -1, dtype=torch.int32, device=device
        )

    def record(self, layer, indices, first=0):
        if indices.ndim != 2 or indices.size(1) != 6:
            raise ValueError("DeepSeek target routing must have six experts per row")
        last = first + indices.size(0)
        if not 0 <= layer < self.indices.size(
            0
        ) or not 0 <= first < last <= self.indices.size(1):
            raise ValueError("DeepSeek routing trace write is out of bounds")
        self.indices[layer, first:last].copy_(indices)

    def device_counters(self, credited_rows):
        if credited_rows.shape != (self.indices.size(1),):
            raise ValueError("credited row mask differs from the routing trace")
        if (
            credited_rows.dtype != torch.bool
            or credited_rows.device != self.indices.device
        ):
            raise ValueError(
                "credited row mask must use the trace device and boolean type"
            )
        valid = credited_rows[None, :, None] & (self.indices >= 0)
        ids = self.indices.clamp_min(0).to(torch.int64)
        counts = torch.zeros(
            (self.indices.size(0), 384), dtype=torch.int32, device=ids.device
        )
        counts.scatter_add_(1, ids.flatten(1), valid.to(torch.int32).flatten(1))
        assignments = counts.view(-1, 4, 96).sum(-1, dtype=torch.int32)
        presence = (counts > 0).to(torch.uint8)
        dispatch = torch.zeros_like(assignments)
        if self.strategy == "dep4":
            destinations = torch.zeros(
                (*self.indices.shape[:2], 4), dtype=torch.int32, device=ids.device
            )
            destinations.scatter_add_(
                2, ids.div(96, rounding_mode="floor"), valid.to(torch.int32)
            )
            dispatch = (destinations > 0).sum(1, dtype=torch.int32)
        return assignments, dispatch, presence

    def counters(self, credited_rows, *, expert_union=None):
        assignments, dispatch, presence = self.device_counters(credited_rows)
        if self.strategy == "dep4" and getattr(self, "deferred", False):
            return {
                "_deferred_routes": (
                    assignments,
                    dispatch,
                    presence,
                    credited_rows.clone(),
                )
            }
        if self.strategy == "dep4":
            (dep4_expert_union if expert_union is None else expert_union)(presence)
        local = (
            presence
            if self.strategy == "tp4"
            else presence[:, self.rank * 96 : (self.rank + 1) * 96]
        )
        touched = local.sum(-1, dtype=torch.int32)
        packed = torch.cat((assignments, dispatch, touched[:, None]), dim=1)
        values = packed.cpu().tolist()
        return {
            "expert_assignments": [row[:4] for row in values],
            "dispatch_rows": [row[4:8] for row in values],
            "touched_experts": [row[8] for row in values],
        }
