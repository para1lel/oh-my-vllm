import hashlib
import json
import math
from dataclasses import dataclass

from . import deepseek_accounting_math as arithmetic
from .deepseek_accounting_math import DEEPSEEK_V41_MODEL, WORLD_SIZE

VERSION = "deepseek-http-reference-v3.1"


RULE_VERSION = "k32-fp32-scale-accumulation-v3.1"


HISTORY_CACHE_BYTES = 128 * 1024 * 1024


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


def _validated_calls(records, model):
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
    return calls


def _validate_expert_counters(record, records, rank, model, strategy):
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
                        count / max(1, min(local_experts, model.experts_per_token))
                    )
                    <= distinct
                    <= min(count, token_rows)
                ):
                    raise TraceError(
                        "dispatch rows differ from distinct token destinations"
                    )
            elif distinct:
                raise TraceError("non-DEP4 trace has dispatch rows")
        received = received_assignments(records, rank, layer, strategy)
        bins = model.routed_experts if strategy == "tp4" else local_experts
        touched = record["touched_experts"][layer]
        global_tokens = (
            sum(sum(row["count"] for row in r["request_rows"]) for r in records)
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
                    for source in records
                ),
            )
        if not minimum_touched <= touched <= min(bins, received):
            raise TraceError("touched expert count exceeds assignment bounds")


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
    calls = _validated_calls(records, model)
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
            _validate_expert_counters(
                record, list(ranks.values()), rank, model, strategy
            )
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
