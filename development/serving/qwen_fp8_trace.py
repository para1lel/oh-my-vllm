"""Validate archived execution wire data before constructing useful work."""

import hashlib
import json
import math
from dataclasses import dataclass, field

from .qwen_fp8_evidence import (
    FORMAT as FORMAT,
)
from .qwen_fp8_evidence import (
    LAYOUT as LAYOUT,
)
from .qwen_fp8_evidence import (
    VERSION as VERSION,
)
from .qwen_fp8_evidence import (
    Blocked as Blocked,
)
from .qwen_fp8_evidence import (
    digest as digest,
)
from .qwen_fp8_evidence import (
    execution_key,
    forward_precisions,
    require,
    tokens,
    validate_cache_result,
    validate_checkpoint_sources,
    validate_identity,
    validate_model_state,
    validate_prepared,
)
from .qwen_fp8_math import integer
from .workload import PROFILE, SAMPLING_FIELDS, validate_tools


def validate_sampling(events, manifest, artifacts, tokenizer):
    require(tokenizer is not None, "missing loaded tokenizer sampling evidence")
    raw = (tokenizer.path / "generation_config.json").read_bytes()
    require(
        hashlib.sha256(raw).hexdigest()
        == artifacts["tokenizer"]["generation_config.json"],
        "sampling generation bytes differ from the loaded tokenizer identity",
    )
    generation = json.loads(raw)
    require(
        isinstance(generation, dict) and generation == tokenizer.generation,
        "sampling generation differs from the loaded tokenizer",
    )
    expected = {
        name: value
        for name, value in (
            generation | {"presence_penalty": 0.5, "repetition_guard": True}
        ).items()
        if name in SAMPLING_FIELDS
    }
    actual = manifest["sampling"]
    require(
        isinstance(actual, dict)
        and actual == expected
        and set(actual) == set(expected),
        "effective sampling differs from checkpoint and approved service defaults",
    )
    defaults = dict(
        temperature=1,
        top_k=20,
        top_p=0.95,
        min_p=0,
        frequency_penalty=0,
        presence_penalty=0,
        repetition_penalty=1,
        repetition_guard=True,
    )
    effective = defaults | actual
    for name, value in effective.items():
        if name == "repetition_guard":
            require(type(value) is bool, "invalid effective repetition guard")
        else:
            require(
                type(value) in (int, float) and math.isfinite(value),
                "invalid effective sampling parameter",
            )
    require(effective["temperature"] > 0, "workload requires stochastic sampling")
    integer(effective["top_k"] + 1, "effective top-k plus one")
    for row in manifest["requests"]:
        require(
            all(name not in row["body"] for name in SAMPLING_FIELDS),
            "request sampling differs from the approved default workload",
        )
    validate_prepared(events, manifest, effective)


def validate_requests(manifest, responses, tokenizer=None):
    require(manifest["workload"] == PROFILE, "workload differs from serving-tools-v1")
    require(manifest["mode"] in ("ordinary", "mtp4"), "unknown Qwen mode")
    rows = manifest["requests"]
    require(isinstance(rows, list) and rows, "empty request manifest")
    require(len(rows) == len(responses), "HTTP response inventory differs")
    require(len({r["id"] for r in rows}) == len(rows), "duplicate trace request ID")
    require(len({r["response_id"] for r in rows}) == len(rows), "duplicate response ID")
    mapped = {response["id"]: response for response in responses}
    require(len(mapped) == len(responses), "duplicate HTTP response ID")
    states = {}
    for ordinal, row in enumerate(sorted(rows, key=lambda r: r["ordinal"])):
        require(
            type(row["ordinal"]) is int
            and row["ordinal"] == ordinal
            and type(row["seed"]) is int
            and row["seed"] == ordinal,
            "ordinal or seed differs from the frozen workload",
        )
        key = integer(row["id"], "request ID", 1)
        require(
            row["response_id"].startswith("chatcmpl_")
            and row["response_id"].rsplit("_", 1)[-1] == str(key),
            "unbound response ID",
        )
        response = mapped[row["response_id"]]
        usage = response["usage"]
        prompt = integer(row["input_tokens"], "prompt length", 1)
        output = integer(row["output_tokens"], "output length", 1)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            integer(usage[name], f"HTTP {name}", 1)
        require(
            usage["prompt_tokens"] == prompt
            and usage["completion_tokens"] == output
            and usage["total_tokens"] == prompt + output,
            "HTTP usage differs",
        )
        require(
            validate_tools(response, tokenizer) == row["tools"],
            "tool validation differs",
        )
        require(
            type(usage["prompt_tokens_details"]["cached_tokens"]) is int
            and usage["prompt_tokens_details"]["cached_tokens"] == 0,
            "AgentX/prefix accounting is blocked without a full state inventory",
        )
        require(
            len(tokens(row["prompt_token_ids"])) == prompt,
            "prompt token identity differs",
        )
        require(
            row["body_sha256"] == digest(row["body"]), "request body identity differs"
        )
        require(
            row["body"]["seed"] == ordinal
            and row["body"]["min_tokens"] == output
            and row["body"]["max_tokens"] == output
            and row["body"]["tools"] == PROFILE["tools"]
            and row["body"]["tool_choice"] == "auto"
            and row["body"]["reasoning_effort"] == "off",
            "request settings differ",
        )
        require(
            response["choices"][0]["finish_reason"] == "length",
            "HTTP response is incomplete",
        )
        states[key] = RequestState(row, list(row["prompt_token_ids"]))
    return states


@dataclass
class RequestState:
    manifest: dict
    history: list
    computed: int = 0
    drafts: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    accepted_end: int | None = None
    complete: bool = False
    last_key: tuple | None = None
    last_result: dict | None = None
    endpoints: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)

    def apply(self, row, result, key, mtp, identity, precision):
        require(
            not self.complete and result["error"] is None,
            "failed or already completed request",
        )
        require(
            result["request_id"] == row["request_id"]
            and result["lease_id"] == row["lease_id"],
            "scheduled/result lease or request differs",
        )
        start, end = row["forward_start"], row["forward_end"]
        integer(start, "forward start")
        integer(end, "forward end", start)
        require(
            end - start == len(tokens(row["token_ids"])),
            "forwarded token count differs",
        )
        purpose, action, kind = row["purpose"], row["action"], row["sampling"]["kind"]
        require(
            purpose in ("advance", "finalize")
            and action in ("forward", "head_only", "export_only")
            and kind in ("none", "target", "verify"),
            "unknown execution work",
        )
        require(
            result["purpose"] == purpose and result["action"] == action,
            "result work differs",
        )
        validate_model_state(row, not self.calls, identity)
        require(
            self.last_key is None or key[:2] > self.last_key[:2],
            "stale or reordered execution",
        )
        if not self.calls:
            require(
                row["prefill_token_ids"] == self.history,
                "prepared prompt token identity differs",
            )
        validate_checkpoint_sources(row, self.endpoints, self.last_result)
        origin = dict(row=row, ba_precision=precision)
        if action != "forward":
            source = row["model_state"]["endpoint_source"]
            slot = source.get("slot")
            if source["kind"] == "forward":
                slot = next(
                    value["slot"]
                    for value in self.last_result["validated_exports"]
                    if value["end"] == row["endpoint"]
                )
            origin = self.sources[slot]
        emitted = tokens(result["token_ids"])
        if purpose == "finalize":
            self.finalize(row, result, emitted)
        else:
            self.advance(row, result, emitted, mtp)
        if emitted:
            self.steps[-1]["forward"] = origin
        for at, slot, version in validate_cache_result(row, result):
            self.endpoints[slot] = version, at
            self.sources[slot] = origin
        self.calls.append(dict(row=row, result=result, ba_precision=precision))
        self.computed = result["computed_end"]
        self.last_key, self.last_result = key, result

    def advance(self, row, result, emitted, mtp):
        require(self.accepted_end is None, "advance after parser acceptance was frozen")
        start, end, kind = (
            row["forward_start"],
            row["forward_end"],
            row["sampling"]["kind"],
        )
        if kind == "verify":
            count = integer(row["sampling"]["drafts"], "active drafts", 1)
            require(
                mtp
                and row["action"] == "forward"
                and start == self.computed
                and start == len(self.history) - 1
                and count <= 4
                and end - start == count + 1
                and self.drafts == row["token_ids"][1:],
                "verification proposal dependency differs",
            )
            require(
                row["token_ids"][:1] == self.history[start : start + 1],
                "verification history differs",
            )
            accepted = integer(result["num_accepted_draft_tokens"], "accepted drafts")
            require(
                accepted <= count
                and len(emitted) == accepted + 1
                and emitted[:accepted] == self.drafts[:accepted],
                "invalid accepted proposal prefix",
            )
            computed = start + accepted + 1
        else:
            require(kind in ("target", "none"), "invalid advance sampling")
            require(
                start <= self.computed
                and end <= len(self.history)
                and (kind != "target" or end == row["endpoint"] == len(self.history))
                and row["token_ids"] == self.history[start:end],
                "prefill/replay token identity differs",
            )
            require(
                len(emitted) == int(kind == "target")
                and result["num_accepted_draft_tokens"] == 0,
                "target output count differs",
            )
            require(action_valid(row), "invalid forward or head-only action")
            computed = end
        require(
            result["computed_end"] == computed
            and result["accepted_end"] == len(self.history) + len(emitted),
            "result endpoints differ",
        )
        if emitted:
            self.steps.append(dict(row=row, tokens=list(emitted), result=result))
        self.history.extend(emitted)
        self.drafts = tokens(result["new_draft_token_ids"])
        require(
            (mtp or not self.drafts) and len(self.drafts) <= 4,
            "invalid proposal suffix",
        )
        if mtp:
            require(
                result["mtp_accepted_endpoint"]["end"] == max(0, computed - 1),
                "MTP accepted context does not end at F-1",
            )

    def finalize(self, row, result, emitted):
        end = self.accepted_end
        require(
            end is not None and row["accepted_end"] == end,
            "missing parser-frozen accepted endpoint",
        )
        require(
            row["sampling"]["kind"] == "none"
            and not emitted
            and not result["new_draft_token_ids"]
            and result["num_accepted_draft_tokens"] == 0,
            "finalization sampled or changed usage",
        )
        require(
            result["accepted_end"]
            == result["computed_end"]
            == row["forward_end"]
            == end,
            "finalization did not produce the exact endpoint",
        )
        require(
            row["token_ids"] == self.history[row["forward_start"] : end],
            "finalization replay differs",
        )
        require(
            row["action"] != "head_only", "finalization cannot run the sampling head"
        )
        require(
            row["action"] == "forward"
            or row["forward_start"] == end == row["endpoint"],
            "invalid zero-row endpoint export",
        )
        require(
            any(export["end"] == end for export in result["validated_exports"]),
            "exact endpoint has no validated export",
        )

    def accept(self, end):
        require(
            self.accepted_end is None and not self.complete,
            "duplicate or late parser acceptance",
        )
        require(
            type(end) is int
            and self.manifest["input_tokens"] <= end <= len(self.history),
            "invalid parser accepted endpoint",
        )
        require(
            end == self.manifest["input_tokens"] + self.manifest["output_tokens"],
            "accepted endpoint differs from full HTTP usage",
        )
        self.accepted_end = end
        self.history = self.history[:end]
        self.drafts = []

    def finish(self, row, mtp):
        end = self.accepted_end
        require(
            not self.complete and end is not None and self.computed == end,
            "request success precedes exact finalization",
        )
        execution = row["execution"]
        key = tuple(
            execution[name]
            for name in ("worker_epoch", "step_id", "request_id", "lease_id")
        )
        require(
            row["status"] == "completed"
            and row["accepted_end"] == end
            and key == self.last_key,
            "success has no matching validated execution",
        )
        require(
            row["token_sha256"] == digest(self.history),
            "accepted token identity differs",
        )
        require(
            not mtp or self.last_result["mtp_accepted_endpoint"]["end"] == end - 1,
            "exact final MTP endpoint differs",
        )
        require(
            self.calls[-1]["row"]["purpose"]
            == self.last_result["purpose"]
            == "finalize"
            and self.calls[-1]["row"]["sampling"]["kind"] == "none",
            "success has no matching finalize execution",
        )
        require(self.last_result["validated_exports"], "missing final export evidence")
        self.complete = True


def action_valid(row):
    if row["action"] == "forward":
        return row["forward_end"] > row["forward_start"]
    return (
        row["action"] == "head_only"
        and row["sampling"]["kind"] == "target"
        and row["forward_start"] == row["forward_end"] == row["endpoint"]
    )


def validate_private_owners(row, states):
    refs = {
        (p["slot"], p["version"]) for p in row["model_state"]["private_checkpoints"]
    }
    for key, state in states.items():
        if not state.calls or state.complete:
            continue
        previous = {
            (p["slot"], p["version"])
            for p in state.calls[-1]["row"]["model_state"]["private_checkpoints"]
        }
        require(
            refs == previous
            if key == row["request_id"]
            else not ({p[0] for p in refs} & {p[0] for p in previous}),
            "private checkpoint ownership changed or aliases another active request",
        )


def validate_call(event, states, mtp, seen, identity):
    require(event["validated"] is True, "worker results are not Rust-validated")
    epoch = integer(event["worker_epoch"], "worker epoch", 1)
    step = integer(event["step_id"], "step ID", 1)
    require((epoch, step) not in seen, "duplicate execution step")
    seen.add((epoch, step))
    scheduled, outputs = event["scheduled"], event["outputs"]
    require(scheduled and len(scheduled) == len(outputs), "execution inventory differs")
    results = {row["request_id"]: row for row in outputs}
    require(
        len(results) == len(outputs)
        and len({r["request_id"] for r in scheduled}) == len(scheduled),
        "duplicate scheduled/result request",
    )
    precisions = forward_precisions(event, mtp)
    for row in scheduled:
        key = execution_key(row, epoch, step)
        require(key[2] in states and key[2] in results, "unknown execution request ID")
        validate_private_owners(row, states)
        states[key[2]].apply(
            row, results[key[2]], key, mtp, identity, precisions.get(key[2])
        )


def validate_execution(events, manifest, artifacts, responses, tokenizer=None):
    try:
        identity = validate_identity(events, manifest, artifacts)
        require(
            tokenizer is None or tokenizer.identity == artifacts["tokenizer"],
            "tool validator tokenizer identity differs",
        )
        states = validate_requests(manifest, responses, tokenizer)
        mtp, seen = manifest["mode"] == "mtp4", set()
        for event in events:
            operation = event["operation"]
            if operation in ("identity", "prepared"):
                continue
            if operation == "accounting_call":
                require(
                    event["worker_epoch"] == identity["config"]["worker_epoch"],
                    "execution came from a different loaded worker epoch",
                )
                validate_call(event, states, mtp, seen, identity)
            elif operation in ("accepted", "complete"):
                for row in event["requests"]:
                    require(row["id"] in states, "unknown lifecycle request ID")
                    if operation == "accepted":
                        states[row["id"]].accept(row["accepted_end"])
                    else:
                        states[row["id"]].finish(row, mtp)
            else:
                raise Blocked(
                    f"unsupported or insufficient trace operation: {operation}"
                )
        require(
            all(state.complete for state in states.values()),
            "incomplete validated request inventory",
        )
        return states
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise Blocked(f"missing or invalid runtime evidence: {error}") from error


def validate(events, manifest, artifacts, responses, tokenizer=None):
    states = validate_execution(events, manifest, artifacts, responses, tokenizer)
    try:
        validate_sampling(events, manifest, artifacts, tokenizer)
    except (KeyError, TypeError, AttributeError, OSError) as error:
        raise Blocked(f"missing or invalid sampling evidence: {error}") from error
    return states
