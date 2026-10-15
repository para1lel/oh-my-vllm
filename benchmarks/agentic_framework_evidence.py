"""Check serving contracts independently of native agent task outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.agentic_native_goals import collection_record
from benchmarks.agentic_session_evidence import terminal_usage
from benchmarks.agentic_soak_analysis import file_identity, request_metric
from benchmarks.agentic_soak_protocol import StreamCompletion


def response_protocol(path: Path, api: str) -> tuple[dict, list[dict]]:
    """Validate SSE before a canonical serving cancellation/error frame."""
    parser, errors = StreamCompletion(api, 16 << 20), []
    frame = bytearray()
    response_done = False

    def complete_frame(raw):
        nonlocal response_done
        fields = [line.rstrip(b"\r") for line in raw.split(b"\n")]
        data = b"\n".join(
            line[5:].removeprefix(b" ") for line in fields if line.startswith(b"data:")
        )
        event_names = [
            line[6:].removeprefix(b" ") for line in fields if line.startswith(b"event:")
        ]
        if not data:
            parser.feed(raw)
            return
        if errors:
            raise ValueError("model data followed a terminal serving error")
        if parser.snapshot()["success"]:
            if api == "responses" and data == b"[DONE]" and not response_done:
                response_done = True
                return
            raise ValueError("model data followed successful SSE completion")
        event = None if data == b"[DONE]" else json.loads(data)
        if (
            isinstance(event, dict)
            and "error" in event
            and (api != "chat" or set(event) != {"error"})
        ):
            raise ValueError("serving error envelope differs from its API")
        error = event.get("error") if isinstance(event, dict) else None
        if (
            api == "responses"
            and isinstance(event, dict)
            and event.get("type") == "error"
        ):
            if (
                event_names != [b"error"]
                or event.get("param") is not None
                or type(event.get("sequence_number")) is not int
                or event["sequence_number"] < 0
                or set(event) != {"type", "code", "message", "param", "sequence_number"}
            ):
                raise ValueError("invalid Responses serving error frame")
            error = {name: event[name] for name in ("code", "message")}
            error["type"] = "server_error"
        if isinstance(error, dict):
            if (
                error.get("code") != "server_error"
                or error.get("message") != "request cancelled or timed out"
                or error.get("type", "server_error") != "server_error"
                or set(error) - {"code", "message", "type"}
                or parser.snapshot()["success"]
            ):
                raise ValueError("unexpected serving SSE error")
            errors.append(error)
        else:
            parser.feed(raw)

    with path.open("rb") as source:
        for line in source:
            frame.extend(line)
            if len(frame) > 16 << 20:
                raise ValueError("SSE evidence frame exceeds its bound")
            if line in (b"\n", b"\r\n"):
                complete_frame(bytes(frame))
                frame.clear()
    # A cancelled transport can end within an SSE frame. Retain this fact.
    if frame:
        if (errors or parser.snapshot()["success"]) and any(
            line.startswith(b"data:") for line in frame.splitlines()
        ):
            raise ValueError("model data followed a terminal serving error")
        parser.feed(bytes(frame))
    return dict(
        parser.snapshot(),
        response_ids=sorted(parser.ids),
        trailing_frame_bytes=len(frame),
    ), errors


def audit_framework_session(
    directory: Path, session: dict, metrics: dict, measured_start: float
) -> tuple[list[dict], dict]:
    """Bind all requests; report cancellations without asserting their cause."""
    states = json.loads((directory / "states.json").read_text())
    if not states or any(
        state.get("sessionId") != session["session_id"] for state in states
    ):
        raise ValueError("native workload session identity changed")
    wire, seen, points, outcomes = directory / "wire", set(), [], []
    with (wire / "http.jsonl").open() as source:
        for line in source:
            record = json.loads(line)
            identities = record.get("response_ids", [])
            if record.get("response_status") != 200 or len(identities) != 1:
                raise ValueError("model request lacks HTTP success/response identity")
            identity = identities[0]
            if identity in seen or identity not in metrics:
                raise ValueError("missing or duplicate serving request identity")
            seen.add(identity)
            metric = request_metric(metrics[identity])
            if (
                metric["response_id"] != identity
                or metric["api"] != session["api"]
                or record.get("session_id") != session["session_id"]
                or metric["status"] == "failed"
            ):
                raise ValueError("model request failed or changed API/session")
            proof = record["context_proof"]
            if (
                not all(
                    type(proof[name]) is int and proof[name] >= 0
                    for name in ("input_tokens", "output_budget", "total_budget")
                )
                or proof["output_budget"] == 0
                or proof["total_budget"]
                != proof["input_tokens"] + proof["output_budget"]
                or proof["total_budget"] > 262144
                or proof["reasoning_effort"] != "xhigh"
            ):
                raise ValueError("invalid actual model context/reasoning budget")
            if metric["output_tokens"] > proof["output_budget"]:
                raise ValueError("usage differs from actual model output budget")
            for name in ("request", "response"):
                path = wire / record[f"{name}_file"]
                if not path.resolve().is_relative_to(wire.resolve()):
                    raise ValueError("wire evidence path escaped its directory")
                if (
                    name == "response"
                    and file_identity(path)["sha256"] != record["response_sha256"]
                ):
                    raise ValueError("wire evidence bytes changed")
            body = json.loads((wire / record["request_file"]).read_text())
            api = session["api"]
            budget = (
                body.get("max_completion_tokens", body.get("max_tokens"))
                if api == "chat"
                else body.get("max_output_tokens")
            )
            if (
                body.get("model") != "qwen3.8-27b-fp8"
                or body.get("stream") is not True
                or record.get("path")
                != ("/v1/chat/completions" if api == "chat" else "/v1/responses")
                or budget != proof["output_budget"]
                or not isinstance(
                    body.get("messages" if api == "chat" else "input"), list
                )
            ):
                raise ValueError(
                    "original model request differs from its preflight contract"
                )
            response = wire / record["response_file"]
            protocol, errors = response_protocol(response, session["api"])
            if (
                protocol["response_ids"] != [identity]
                or protocol["model"] != "qwen3.8-27b-fp8"
                or protocol["errors"]
                or len(errors) > 1
            ):
                raise ValueError("model response violates its SSE contract")
            outcome = {
                "response_id": identity,
                "request_file": record["request_file"],
                "status": metric["status"],
                "wire_error": record.get("error"),
                "error_direction": record.get("error_direction"),
                "transport_error_kind": record.get("transport_error_kind"),
                "protocol": protocol,
                "server_error": errors,
            }
            outcomes.append(outcome)
            if metric["status"] == "cancelled":
                # A cancellation is reported without asserting its cause unless
                # the original canonical abort evidence binds this request, in
                # which case its own failure direction and window travel with it.
                # The proof is optional evidence here, so an unavailable or
                # malformed record leaves the cancellation unproved with an
                # explicit diagnostic instead of failing the whole audit.
                if collection_record(session):
                    from benchmarks.agentic_deadline_evidence import deadline_abort

                    try:
                        proof = deadline_abort(
                            directory, session, record, metric, measured_start
                        )
                    except Exception as error:
                        outcome["cancellation_evidence"] = (
                            f"{type(error).__name__}: {error}"
                        )
                    else:
                        outcome["client_completion"] = "collection_deadline_aborted"
                        outcome["deadline_abort_proof"] = proof
                continue
            if errors or not protocol["success"]:
                if not errors and collection_record(session):
                    from benchmarks.agentic_deadline_evidence import deadline_abort

                    outcome["client_completion"] = "collection_deadline_aborted"
                    outcome["deadline_abort_proof"] = deadline_abort(
                        directory, session, record, metric, measured_start
                    )
                    continue
                raise ValueError("completed request lacks successful SSE completion")
            if terminal_usage(response, session["api"]) != {
                "input": proof["input_tokens"],
                "output": metric["output_tokens"],
            }:
                raise ValueError("completed usage differs from CPU/server counts")
            if (
                protocol["finish_reason"] == "length"
                and metric["output_tokens"] != proof["output_budget"]
            ):
                raise ValueError("length completion differs from output budget")
            points.append(
                {
                    **metric,
                    "lane": session["lane"],
                    "session_id": session["session_id"],
                    "purpose": record.get("purpose_hint", "task"),
                    "input_tokens": proof["input_tokens"],
                    "output_budget": proof["output_budget"],
                    "elapsed_minutes": (
                        metric["processing_started_unix_s"] - measured_start
                    )
                    / 60,
                    "output_tps": metric["output_tokens"]
                    / metric["processing_seconds"],
                }
            )
    return points, {
        "lane": session["lane"],
        "api": session["api"],
        "session_id": session["session_id"],
        "response_ids": sorted(seen),
        "outcomes": outcomes,
        "completed_requests": len(points),
        "cancelled_requests": sum(
            outcome["status"] == "cancelled" for outcome in outcomes
        ),
        "deadline_aborted_completed_requests": sum(
            outcome.get("client_completion") == "collection_deadline_aborted"
            for outcome in outcomes
        ),
        "cancellation_cause": "not inferred from a shared cancel/deadline outcome",
    }


def audit_oneoff_requests(
    requests: list[dict], api: str, metrics: list[dict], raw_directory: Path
) -> dict:
    """Check original native OMP streams without a final-answer requirement."""
    by_id = {metric["response_id"]: request_metric(metric) for metric in metrics}
    if len(by_id) != len(metrics) or not by_id:
        raise ValueError("oneoff workload has no unambiguous model execution")
    seen, outcomes = set(), []
    for request in requests:
        identities = request.get("response_ids", [])
        if request.get("response_status") != 200 or len(identities) != 1:
            raise ValueError("oneoff request has no valid HTTP/response identity")
        identity = identities[0]
        if identity in seen or identity not in by_id:
            raise ValueError("oneoff serving request identity is missing or repeated")
        seen.add(identity)
        metric, body = by_id[identity], request["request"]
        if (
            metric["response_id"] != identity
            or metric["api"] != api
            or metric["status"] == "failed"
        ):
            raise ValueError("oneoff request failed or uses a different API")
        if body.get("model") != "qwen3.8-27b-fp8" or body.get("stream") is not True:
            raise ValueError("oneoff uses a different model or response contract")
        path = raw_directory / request["response_file"]
        if not path.resolve().is_relative_to(raw_directory.resolve()):
            raise ValueError("oneoff response path escaped evidence directory")
        if file_identity(path)["sha256"] != request["response_sha256"]:
            raise ValueError("oneoff original SSE bytes changed")
        protocol, errors = response_protocol(path, api)
        if (
            protocol["response_ids"] != [identity]
            or protocol["model"] != "qwen3.8-27b-fp8"
            or protocol["errors"]
            or len(errors) > 1
        ):
            raise ValueError("oneoff SSE contract is invalid")
        if metric["status"] == "completed":
            if errors or not protocol["success"]:
                raise ValueError("oneoff completion lacks a successful SSE terminal")
            usage = terminal_usage(path, api)
            if usage["output"] != metric["output_tokens"]:
                raise ValueError("oneoff SSE and server usage differ")
            budget = (
                body.get("max_completion_tokens", body.get("max_tokens"))
                if api == "chat"
                else body.get("max_output_tokens")
            )
            if protocol["finish_reason"] == "length" and usage["output"] != budget:
                raise ValueError("oneoff length completion differs from its budget")
        outcomes.append({"response_id": identity, "status": metric["status"]})
    if seen != set(by_id) or len({metric["epoch"] for metric in metrics}) != 1:
        raise ValueError("oneoff request coverage or worker epoch changed")
    return {
        "requests": len(seen),
        "outcomes": outcomes,
        "task_outcomes_are_gates": False,
    }
