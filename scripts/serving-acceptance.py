"""Exercise real JSON and strict tool generation through both HTTP APIs.

Start the UUID-pinned service with MTP4 first. This records request/response
evidence; also inspect matching request IDs and draft counters in server logs.
Passing against a scripted worker is not real model acceptance.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.evidence import read_service_evidence
from scripts.serving_evidence import ServiceAttempt, validate_constraint_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speculative-mode", choices=["mtp", "dspark"], default="mtp")
    parser.add_argument("--server-log", type=Path)
    args = parser.parse_args()
    if args.speculative_mode == "dspark" and not args.server_log:
        parser.error("DSpark constrained acceptance requires --server-log")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with ServiceAttempt(
        args.output_dir / "attempt.json",
        {"speculative_mode": args.speculative_mode, "kind": "constraints"},
    ) as attempt:
        run(args, attempt)
        if args.server_log:
            attempt.update(passed=True)


def run(args, attempt):
    OUT = args.output_dir
    log_offset = args.server_log.stat().st_size if args.server_log else None
    log_identity = (
        (args.server_log.stat().st_dev, args.server_log.stat().st_ino)
        if args.server_log
        else None
    )
    request_ids = set()
    SCHEMA = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "enum": [123]},
            "label": {"type": "string", "const": "verified"},
        },
        "required": ["n", "label"],
        "additionalProperties": False,
    }
    for api in ["chat", "responses"]:
        for effort in ["off", "medium"]:
            for kind in ["json_object", "json_schema", "tool"]:
                key = f"{api}-{effort}-{kind}"
                prompt = (
                    "Return JSON with n=123 and label=verified. "
                    "For tool use, call report with those arguments."
                )
                body = {"model": "qwen3.8-27b-fp8", "temperature": 0}
                if api == "chat":
                    path = "chat/completions"
                    body.update(
                        messages=[{"role": "user", "content": prompt}],
                        reasoning_effort=effort,
                        max_completion_tokens=4096,
                    )
                else:
                    path = "responses"
                    body.update(
                        input=prompt,
                        reasoning={"effort": effort},
                        max_output_tokens=4096,
                        store=False,
                    )
                if kind == "tool":
                    function = {
                        "name": "report",
                        "description": "Report the result.",
                        "strict": True,
                        "parameters": SCHEMA,
                    }
                    body["tools"] = (
                        [{"type": "function", "function": function}]
                        if api == "chat"
                        else [{"type": "function", **function}]
                    )
                    body.update(tool_choice="required", parallel_tool_calls=False)
                else:
                    fmt = {"type": kind}
                    if kind == "json_schema":
                        details = {"name": "result", "schema": SCHEMA, "strict": True}
                        fmt.update(
                            {"json_schema": details} if api == "chat" else details
                        )
                    body.update(
                        {"response_format": fmt}
                        if api == "chat"
                        else {"text": {"format": fmt}}
                    )
                started = time.monotonic()
                result = attempt.request(
                    args.base_url.rstrip("/") + "/" + path, body, timeout=600
                )
                (OUT / (key + ".json")).write_text(
                    json.dumps(
                        {
                            "request": body,
                            "response": result,
                            "elapsed_s": time.monotonic() - started,
                            "speculative_mode": args.speculative_mode,
                        },
                        indent=2,
                    )
                )
                validate_constraint_response(result, api, kind)
                request_ids.add(int(result["id"].rsplit("_", 1)[1]))
                print(
                    key,
                    "PASS",
                    round(time.monotonic() - started, 3),
                    "seconds",
                    "id=",
                    result["id"],
                    flush=True,
                )

    if args.server_log:
        if log_identity != (
            args.server_log.stat().st_dev,
            args.server_log.stat().st_ino,
        ):
            raise ValueError(
                "server log changed identity during constrained acceptance"
            )
        evidence = read_service_evidence(
            args.server_log,
            log_offset,
            args.speculative_mode,
            request_ids=request_ids,
            expected=12,
        )
        (OUT / "server-verification.json").write_text(
            json.dumps(
                {
                    "speculative_mode": args.speculative_mode,
                    "requests": evidence,
                    "passed": True,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
