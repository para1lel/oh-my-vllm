"""Check target-model reasoning budgets through both HTTP APIs.

Run against each UUID-pinned ordinary, MTP4 and DSpark service. Keep original
request/response evidence. Scripted workers are not target-model acceptance.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oh_my_vllm.performance.coverage import runtime_contract

from benchmarks.agentic_soak_analysis import read_metrics
from benchmarks.agentic_soak_protocol import StreamCompletion
from benchmarks.evidence import (
    observed_configuration,
    read_service_evidence,
    runtime_source_identity,
)
from scripts.serving_evidence import (
    ServiceAttempt,
    response_text,
    validate_constraint_response,
)


def stream_result(attempt, url, body, api):
    events = []
    proof = StreamCompletion(api, 4 << 20)
    with attempt.stream(url, body, timeout=1800) as lines:
        for line in lines:
            proof.feed(line)
            if line.startswith(b"data: "):
                data = line[6:].strip()
                events.append("[DONE]" if data == b"[DONE]" else json.loads(data))
    completion = proof.snapshot()
    if (
        not completion["success"]
        or completion["model"] != body["model"]
        or proof.pending
        or proof.data_lines
    ):
        raise ValueError("stream protocol or model identity differs")
    if api == "responses":
        terminal = [
            event["response"]
            for event in events
            if isinstance(event, dict)
            and event.get("type") in ("response.completed", "response.incomplete")
        ]
        if len(terminal) != 1:
            raise ValueError("Responses stream has no unique terminal")
        return terminal[0]
    if not events or events[-1] != "[DONE]":
        raise ValueError("Chat stream has no terminal")
    chunks = events[:-1]
    choices = [chunk["choices"][0] for chunk in chunks if chunk.get("choices")]
    reasons = [
        choice["finish_reason"] for choice in choices if choice.get("finish_reason")
    ]
    usage = [chunk["usage"] for chunk in chunks if chunk.get("usage")]
    if len(reasons) != 1 or len(usage) != 1:
        raise ValueError("Chat stream has no unique finish reason and usage")
    return {
        "id": chunks[0]["id"],
        "model": completion["model"],
        "usage": usage[0],
        "choices": [
            {
                "finish_reason": reasons[0],
                "message": {
                    "content": "".join(
                        choice["delta"].get("content", "") for choice in choices
                    ),
                },
            }
        ],
    }


def server_identity(diagnostics, log, source):
    """Bind the live process and its startup source record, not only a local build."""
    configured = observed_configuration(log)
    if configured.get("rust_sources_sha256") != source["rust_sources_sha256"]:
        raise ValueError("running Rust build stamp differs")
    pid = diagnostics["server_pid"]
    stat_path = Path("/proc", str(pid), "stat")
    before = int(stat_path.read_text().rsplit(") ", 1)[1].split()[19])
    binary = hashlib.sha256(Path("/proc", str(pid), "exe").read_bytes()).hexdigest()
    after = int(stat_path.read_text().rsplit(") ", 1)[1].split()[19])
    if before != after or binary != source["binary_sha256"]:
        raise ValueError("live server process or executable differs")
    runtimes = []
    for line in log.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("message") == "Independent runtime":
            runtimes.append(row)
    if (
        len(runtimes) != 1
        or runtimes[0].get("sources_sha256") != runtime_contract()["sources"]
    ):
        raise ValueError("worker startup source identities differ")
    worker_pid = runtimes[0]["pid"]
    if diagnostics.get("worker_pid") != worker_pid:
        raise ValueError("worker diagnostics and startup identities differ")
    worker_stat = (
        Path("/proc", str(worker_pid), "stat").read_text().rsplit(") ", 1)[1].split()
    )
    if int(worker_stat[1]) != pid:
        raise ValueError("worker does not belong to the live server")
    return {
        "pid": pid,
        "start_ticks": before,
        "binary_sha256": binary,
        "worker_pid": worker_pid,
        "worker_start_ticks": int(worker_stat[19]),
        "worker_sources": runtimes[0]["sources_sha256"],
        "configuration": configured,
    }


def run(args, attempt):
    source = runtime_source_identity(args.binary)
    if source["git_status"]:
        raise ValueError("reasoning acceptance requires committed unchanged source")
    diagnostics_url = args.base_url.rstrip("/").removesuffix("/v1") + "/diagnostics"
    baseline = attempt.request(diagnostics_url)
    deadline = time.monotonic() + 120
    while baseline.get("worker_pid") is None and time.monotonic() < deadline:
        time.sleep(1)
        baseline = attempt.request(diagnostics_url)
    identity = server_identity(baseline, args.server_log.read_text(), source)
    default = baseline["limits"]["thinking_token_budget"]
    if not isinstance(default, int) or default < 0:
        raise ValueError(
            "start this acceptance service with a nonnegative default budget"
        )
    offset = args.server_log.stat().st_size
    log_stat = args.server_log.stat()
    attempt.update(
        source=source, observed_server=identity, initial_diagnostics=baseline
    )
    cases = [
        ("zero", 0, "text", False, 512, "xhigh"),
        ("stream", 2, "text", True, 512, "xhigh"),
        ("json", 0, "json_schema", False, 512, "xhigh"),
        ("tool", 2, "tool", False, 512, "xhigh"),
        ("default", None, "text", False, 512, "xhigh"),
        ("off", 0, "text", False, 512, "off"),
        ("total", 8, "text", True, 1, "xhigh"),
        ("unlimited", -1, "text", False, 512, "xhigh"),
        ("stochastic", 2, "text", False, 512, "xhigh"),
        ("stochastic-stream", 8, "text", True, 512, "xhigh"),
    ]
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "const": 123},
            "label": {"type": "string", "const": "verified"},
        },
        "required": ["n", "label"],
        "additionalProperties": False,
    }
    checks, identities = [], set()
    for api in ("chat", "responses"):
        for name, budget, kind, stream, total, effort in cases:
            prompt = (
                "Return JSON with n=123 and label=verified. For tool use, call report "
                "with those arguments. Think carefully before answering."
                if kind != "text"
                else "Think carefully, then answer with the word READY."
            )
            body = {
                "model": args.model,
                "temperature": 1 if name.startswith("stochastic") else 0,
                "stream": stream,
                "seed": 42,
            }
            if budget is not None:
                body["thinking_token_budget"] = budget
            if api == "chat":
                path = "/chat/completions"
                body.update(
                    messages=[{"role": "user", "content": prompt}],
                    reasoning_effort=effort,
                    max_completion_tokens=total,
                )
                if stream:
                    body["stream_options"] = {"include_usage": True}
            else:
                path = "/responses"
                body.update(
                    input=prompt,
                    reasoning={"effort": effort},
                    max_output_tokens=total,
                    store=False,
                )
            if kind == "tool":
                function = {"name": "report", "strict": True, "parameters": schema}
                body["tools"] = (
                    [{"type": "function", "function": function}]
                    if api == "chat"
                    else [{"type": "function", **function}]
                )
                body.update(tool_choice="required", parallel_tool_calls=False)
            elif kind == "json_schema":
                details = {"name": "result", "schema": schema, "strict": True}
                if api == "chat":
                    body["response_format"] = {"type": kind, "json_schema": details}
                else:
                    body["text"] = {"format": {"type": kind, **details}}
            url = args.base_url.rstrip("/") + path
            result = (
                stream_result(attempt, url, body, api)
                if stream
                else attempt.request(url, body, timeout=1800)
            )
            effective = default if budget is None else budget
            if result.get("model") != args.model:
                raise ValueError("response model identity differs")
            usage = result["usage"]
            output = usage["completion_tokens" if api == "chat" else "output_tokens"]
            reasoning = usage[
                "completion_tokens_details"
                if api == "chat"
                else "output_tokens_details"
            ]["reasoning_tokens"]
            if (
                type(reasoning) is not int
                or type(output) is not int
                or not 0 <= reasoning <= output <= total
            ):
                raise ValueError("reasoning, output and total token counts differ")
            if effective >= 0 and reasoning > effective + 1:
                raise ValueError(
                    "reasoning exceeds its content budget plus closing token"
                )
            if effort == "off" and reasoning != 0:
                raise ValueError("off produced reasoning tokens")
            if budget == 0 and effort != "off" and reasoning != 1:
                raise ValueError("zero budget did not force the closing token")
            finish = (
                result["choices"][0]["finish_reason"]
                if api == "chat"
                else result["status"]
            )
            if name == "total":
                if output != total or finish not in ("length", "incomplete"):
                    raise ValueError("total budget did not stop generation")
            elif kind != "text":
                validate_constraint_response(result, api, kind)
            elif not response_text(result, api).strip() or finish not in (
                "stop",
                "completed",
            ):
                raise ValueError("answer did not continue after reasoning")
            rid = int(result["id"].rsplit("_", 1)[1])
            if rid in identities:
                raise ValueError("duplicate response identity")
            identities.add(rid)
            checks.append(
                {
                    "case": name,
                    "api": api,
                    "response_id": result["id"],
                    "effective_budget": effective,
                    "reasoning_tokens": reasoning,
                    "output_tokens": output,
                    "passed": True,
                }
            )
            attempt.update(checks=checks)
            print(api, name, "PASS", reasoning, output, flush=True)
    if (args.server_log.stat().st_dev, args.server_log.stat().st_ino) != (
        log_stat.st_dev,
        log_stat.st_ino,
    ):
        raise ValueError("server log identity changed")
    modes = re.findall(
        r'\bspeculative_mode="?(none|mtp|dspark)"?\b', args.server_log.read_text()
    )
    if not modes or any(mode != args.mode for mode in modes):
        raise ValueError("server mode identity differs")
    metrics = read_metrics(args.server_log, offset)
    by_id = {metric["response_id"]: metric for metric in metrics}
    if len(metrics) != len(checks):
        raise ValueError("server request count differs")
    for check in checks:
        metric = by_id[check["response_id"]]
        if (
            metric["status"] != "completed"
            or metric["output_tokens"] != check["output_tokens"]
            or metric["thinking_token_budget"] != check["effective_budget"]
        ):
            raise ValueError("server and client budget evidence differ")
    if args.mode != "none":
        speculative = read_service_evidence(
            args.server_log,
            offset,
            args.mode,
            request_ids=identities,
            expected=len(checks),
        )
        by_request = {row["request_id"]: row for row in speculative}
        for check in checks:
            if check["case"].startswith("stochastic"):
                row = by_request[int(check["response_id"].rsplit("_", 1)[1])]
                if row["verified_draft_tokens"] <= 0:
                    raise ValueError("stochastic budget request did not verify drafts")
        attempt.update(speculative_requests=speculative)
    final = attempt.request(diagnostics_url)
    if final["epoch"] != baseline["epoch"] or final["restarts"] != baseline["restarts"]:
        raise ValueError("worker restarted during acceptance")
    if server_identity(final, args.server_log.read_text(), source) != identity:
        raise ValueError("running process identity changed during acceptance")
    if runtime_source_identity(args.binary) != source:
        raise ValueError("source or binary changed during acceptance")
    attempt.update(passed=True, server_metrics=metrics, final_diagnostics=final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "target/release/oh-my-vllm-zmq-worker",
    )
    parser.add_argument("--mode", choices=["none", "mtp", "dspark"], required=True)
    parser.add_argument("--model", default="qwen3.8-27b-fp8")
    args = parser.parse_args()
    with ServiceAttempt(
        args.output, {"kind": "thinking_budget", "mode": args.mode}
    ) as attempt:
        run(args, attempt)


if __name__ == "__main__":
    main()
