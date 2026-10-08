#!/usr/bin/env python3
"""Run the documented read-only OMP task against an already running local service.

Launch that service with scripts/with-gpu.sh and --num-speculative-tokens 4.
This script records client evidence; real MTP counters must also be checked in
server logs. It never substitutes a scripted worker for inference acceptance.
"""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.context_boundary import _reap_on_termination
from benchmarks.speculative import read_service_evidence
from scripts.acceptance_proxy import recording_proxy

ROOT = Path(__file__).resolve().parents[1]

TASK = (
    "Read this repository's README, architecture documentation and necessary source. "
    "Introduce the project goals, architecture, how to run it and its current "
    "completion status. Cite the files supporting your answer. Do not modify files."
    " Read at least crates/scheduler/src/lib.rs and "
    "python/oh_my_vllm/worker/model_runner.py with the read tool before writing "
    "the final answer; directory listings alone do not count."
    " Cite these two source filenames in the final answer."
)


def source_read_path(argument: str) -> dict:
    """Separate OMP's positive line-range suffix from a repository filename."""
    if not isinstance(argument, str):
        raise ValueError("source read path must be a string")
    match = re.fullmatch(
        r"(?P<path>[^:\x00]+)(?::(?P<start>[1-9][0-9]*)(?:-(?P<end>[1-9][0-9]*))?)?",
        argument,
    )
    if match is None:
        raise ValueError("source read has an invalid line-range suffix")
    start = int(match["start"]) if match["start"] else None
    end = int(match["end"]) if match["end"] else start
    if start is not None and end < start:
        raise ValueError("source read line range is reversed")
    resolved = (ROOT / match["path"]).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("source read is outside the repository")
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "original_path": argument,
        "line_range": [start, end] if start is not None else None,
    }


def validate_agent_evidence(events: list[dict], requests: list[dict], api: str) -> dict:
    """Prove successful source reads and their use in the final model request."""
    required = {
        "crates/scheduler/src/lib.rs": r"\bpub\s+fn\s+schedule\s*\(",
        "python/oh_my_vllm/worker/model_runner.py": r"\bclass\s+OhMyVllmWorker\b",
    }
    starts, successful, results, assistants = {}, {}, {}, []
    for event in events:
        kind = event.get("type")
        if kind == "tool_execution_start" and event.get("toolName") == "read":
            starts[event["toolCallId"]] = event.get("args", {}).get("path", "")
        elif kind == "tool_execution_end" and event.get("isError") is False:
            call = event.get("toolCallId")
            path = starts.get(call)
            if not path:
                continue
            try:
                source = source_read_path(path)
            except ValueError:
                continue
            text = "\n".join(
                item.get("text", "")
                for item in event.get("result", {}).get("content", [])
            )
            for relative, anchor in required.items():
                if source["path"] == relative and re.search(anchor, text):
                    successful[call] = source | {"text": text}
        elif kind == "message_end":
            message = event.get("message", {})
            if message.get("role") == "toolResult" and message.get("isError") is False:
                results[message.get("toolCallId")] = message
            elif message.get("role") == "assistant":
                assistants.append(message)
    success = {call: record for call, record in successful.items() if call in results}
    if {record["path"] for record in success.values()} != set(required):
        raise ValueError("OMP must successfully read both required source files")
    if not assistants:
        raise ValueError("OMP has no subsequent model answer")
    final = assistants[-1]
    expected_api = "openai-completions" if api == "chat" else "openai-responses"
    answer = "\n".join(item.get("text", "") for item in final.get("content", []))
    if (
        final.get("api") != expected_api
        or final.get("model") != "qwen3.8-27b-fp8"
        or not answer.strip()
        or final.get("stopReason") != "stop"
    ):
        raise ValueError("OMP final answer is missing or uses a different API/model")
    if any(name not in answer for name in required):
        raise ValueError("OMP final answer must cite both required source filenames")
    for request in requests:
        if request.get("response_status") != 200 or final.get(
            "responseId"
        ) not in request.get("response_ids", []):
            continue
        body = request.get("request", {})
        if body.get("model") != "qwen3.8-27b-fp8":
            continue
        history = body.get("messages" if api == "chat" else "input", [])
        if not isinstance(history, list):
            continue
        forwarded = {}
        forwarded_reads = []
        for item in history:
            if api == "chat" and item.get("role") == "tool":
                call, content = item.get("tool_call_id"), item.get("content", "")
            elif api == "responses" and item.get("type") == "function_call_output":
                call, content = item.get("call_id"), item.get("output", "")
            else:
                continue
            serialized = (
                content
                if isinstance(content, str)
                else json.dumps(content, ensure_ascii=False)
            )
            if isinstance(content, list):
                forwarded_text = "\n".join(item.get("text", "") for item in content)
            else:
                forwarded_text = str(content)
            if call in success and success[call]["text"] in forwarded_text:
                read = {
                    "tool_call_id": call,
                    "path": success[call]["path"],
                    "original_path": success[call]["original_path"],
                    "line_range": success[call]["line_range"],
                    "forwarded_output_sha256": hashlib.sha256(
                        serialized.encode()
                    ).hexdigest(),
                }
                forwarded[success[call]["path"]] = read
                forwarded_reads.append(read)
        if set(forwarded) == set(required):
            return {
                "source_reads": forwarded,
                "successful_source_reads": forwarded_reads,
                "final_response_id": final["responseId"],
                "final_answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                "tool_results_in_final_model_request": True,
                "requires_semantic_answer_review": True,
            }
    raise ValueError(
        "successful tool results were not forwarded to the final model request"
    )


def run_client(command, state: Path, output, errors, timeout=660):
    process = None
    with _reap_on_termination() as cancelled:
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=os.environ | {"PI_CODING_AGENT_DIR": str(state)},
                stdout=output,
                stderr=errors,
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if cancelled() or time.monotonic() >= deadline:
                    return 124
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=1)
            return process.returncode
        finally:
            if process is not None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def configuration(base_url):
    providers = {}
    for name, api in [
        ("chat", "openai-completions"),
        ("responses", "openai-responses"),
    ]:
        providers[f"oh-my-vllm-{name}"] = {
            "baseUrl": base_url,
            "api": api,
            "apiKey": "local-test-only",
            "models": [
                {
                    "id": "qwen3.8-27b-fp8",
                    "name": "Local Qwen3.8 27B FP8",
                    "reasoning": True,
                    "input": ["text"],
                    "contextWindow": 65536,
                    "maxTokens": 8192,
                    "thinking": {
                        "mode": "effort",
                        "efforts": ["minimal", "low", "medium", "high", "xhigh", "max"],
                        "defaultLevel": "medium",
                    },
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "compat": {
                        "supportsReasoningEffort": True,
                        "supportsReasoningParams": True,
                        "supportsReasoningSummary": True,
                        "thinkingFormat": "openai",
                        "qwenTemplateReasoningEffort": False,
                        # Strict normalization adds nullable XML strings.
                        # Use native schemas; test strict API cases separately.
                        "supportsStrictMode": False,
                        "supportsConfigurationUpdate": False,
                        "reasoningEffortMap": {
                            "minimal": "off",
                            "low": "low",
                            "medium": "medium",
                            "high": "xhigh",
                            "xhigh": "xhigh",
                            "max": "xhigh",
                        },
                    },
                }
            ],
        }
    return {"providers": providers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", choices=["chat", "responses"], required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--omp", default="omp")
    parser.add_argument(
        "--thinking",
        default="medium",
        choices=["off", "minimal", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument("--configure-only", action="store_true")
    parser.add_argument("--speculative-mode", choices=["mtp", "dspark"], default="mtp")
    parser.add_argument("--server-log", type=Path)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if (
        args.speculative_mode == "dspark"
        and not args.server_log
        and not args.configure_only
    ):
        parser.error("DSpark OMP acceptance requires --server-log")
    if not args.configure_only and (directory / f"{args.api}.jsonl").exists():
        parser.error("keep every OMP attempt; choose an unused output directory")
    directory.mkdir(parents=True, exist_ok=True)
    state = directory / "state"
    state.mkdir(exist_ok=True)
    (state / "models.yml").write_text(
        json.dumps(configuration(args.base_url), indent=2)
    )
    if args.configure_only:
        print(state)
        return
    command = [
        args.omp,
        "--cwd",
        str(ROOT),
        "--model",
        f"oh-my-vllm-{args.api}/qwen3.8-27b-fp8",
        "--thinking",
        args.thinking,
        "--tools",
        "read,grep,glob",
        "--no-lsp",
        "--no-extensions",
        "--no-skills",
        "--no-rules",
        "--no-title",
        "--no-session",
        "--mode",
        "json",
        "--max-time",
        "10m",
        "--print",
        TASK,
    ]
    started = time.monotonic()
    log_offset = args.server_log.stat().st_size if args.server_log else None
    log_identity = (
        (args.server_log.stat().st_dev, args.server_log.stat().st_ino)
        if args.server_log
        else None
    )
    with recording_proxy(args.base_url) as (proxy_url, requests):
        (state / "models.yml").write_text(
            json.dumps(configuration(proxy_url), indent=2)
        )
        with (
            (directory / f"{args.api}.jsonl").open("w") as output,
            (directory / f"{args.api}.stderr").open("w") as errors,
        ):
            try:
                exit_code = run_client(command, state, output, errors)
            except InterruptedError:
                exit_code = 124
    (directory / f"{args.api}-http.json").write_text(
        json.dumps(requests, indent=2) + "\n"
    )
    summary = {
        "api": args.api,
        "command": command,
        "exit_code": exit_code,
        "elapsed_s": time.monotonic() - started,
        "task": TASK,
        "speculative_mode": args.speculative_mode,
        "passed": False,
        "requires_review": (
            "Verify real tool execution, tool-result follow-up requests, "
            "the final answer claims and model execution evidence; "
            "exit zero alone is not acceptance."
        ),
    }
    try:
        if exit_code:
            raise ValueError(f"OMP exited with status {exit_code}")
        events = [
            json.loads(line)
            for line in (directory / f"{args.api}.jsonl").read_text().splitlines()
        ]
        summary["client_verification"] = validate_agent_evidence(
            events, requests, args.api
        )
        if args.server_log:
            if log_identity != (
                args.server_log.stat().st_dev,
                args.server_log.stat().st_ino,
            ):
                raise ValueError("server log changed identity during OMP acceptance")
            ids = {
                int(rid.rsplit("_", 1)[1])
                for request in requests
                for rid in request.get("response_ids", [])
            }
            summary["server_verification"] = read_service_evidence(
                args.server_log, log_offset, args.speculative_mode, request_ids=ids
            )
        summary["passed"] = bool(args.server_log)
    except ValueError as error:
        summary["error"] = str(error)
    (directory / f"{args.api}-run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    raise SystemExit(exit_code or (1 if "error" in summary else 0))


if __name__ == "__main__":
    main()
