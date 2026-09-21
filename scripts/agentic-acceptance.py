#!/usr/bin/env python3
"""Run the documented read-only OMP task against an already running local service.

Launch that service with scripts/with-gpu.sh and --num-speculative-tokens 4.
This script records client evidence; real MTP counters must also be checked in
server logs. It never substitutes a scripted worker for inference acceptance.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

TASK = (
    "Read this repository's README, architecture documentation and necessary source. "
    "Introduce the project goals, architecture, how to run it and its current "
    "completion status. Cite the files supporting your answer. Do not modify files."
)


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
                    "id": "qwen3.5-27b-fp8",
                    "name": "Local Qwen3.5 27B FP8",
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
    parser.add_argument("--omp", default="/home/dongwu.chen/.local/bin/omp")
    parser.add_argument(
        "--thinking",
        default="medium",
        choices=["off", "minimal", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument("--configure-only", action="store_true")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    state = directory / "state"
    state.mkdir(exist_ok=True)
    (state / "models.yml").write_text(
        json.dumps(configuration(args.base_url), indent=2)
    )
    if args.configure_only:
        print(state)
        return
    root = Path(__file__).resolve().parents[1]
    command = [
        args.omp,
        "--cwd",
        str(root),
        "--model",
        f"oh-my-vllm-{args.api}/qwen3.5-27b-fp8",
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
    with (
        (directory / f"{args.api}.jsonl").open("w") as output,
        (directory / f"{args.api}.stderr").open("w") as errors,
    ):
        process = subprocess.Popen(
            command,
            cwd=root,
            env=os.environ | {"PI_CODING_AGENT_DIR": str(state)},
            stdout=output,
            stderr=errors,
            start_new_session=True,
        )
        try:
            exit_code = process.wait(timeout=660)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            exit_code = 124
    summary = {
        "api": args.api,
        "command": command,
        "exit_code": exit_code,
        "elapsed_s": time.monotonic() - started,
        "task": TASK,
        "requires_review": (
            "Verify real tool execution, tool-result follow-up requests, "
            "grounded final answer, and MTP counters in server logs; "
            "exit zero alone is not acceptance."
        ),
    }
    (directory / f"{args.api}-run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
