"""Exercise real serving thinking, state, mixed batching and disconnect recovery.

Run against a UUID-pinned MTP4 service. Check server cancellation/MTP logs too.
"""

import argparse
import concurrent.futures
import json
import re
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.speculative import read_service_evidence
from scripts.serving_evidence import ServiceAttempt, response_text, stream_identity


def generation_body(api, prompt, effort="off", max_tokens=128):
    body = {"model": "qwen3.8-27b-fp8"}
    if api == "chat":
        body.update(
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort=effort,
            max_tokens=max_tokens,
        )
    else:
        body.update(
            input=prompt,
            reasoning={"effort": effort},
            max_output_tokens=max_tokens,
            store=False,
        )
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api", choices=["chat", "responses"], default="chat")
    parser.add_argument("--speculative-mode", choices=["mtp", "dspark"], default="mtp")
    parser.add_argument(
        "--server-log",
        type=Path,
        required=True,
        help="Serving DEBUG log, used to prove batching and worker cleanup",
    )
    args = parser.parse_args()
    with ServiceAttempt(
        args.output,
        {
            "speculative_mode": args.speculative_mode,
            "api": args.api,
            "kind": "lifecycle",
        },
    ) as attempt:
        run(args, attempt)
        attempt.update(passed=True)


def run(args, attempt):
    log_offset = args.server_log.stat().st_size
    log_identity = (args.server_log.stat().st_dev, args.server_log.stat().st_ino)
    endpoint = "/chat/completions" if args.api == "chat" else "/responses"

    def request(path, body=None, method=None):
        return attempt.request(args.base_url + path, body, method)

    results = {}
    attempt.update(results=results)
    for effort in ["off", "low", "medium", "high", "xhigh"]:
        result = request(
            endpoint,
            generation_body(args.api, "计算 12+13,只回答结果。", effort, 1024),
        )
        if "25" not in response_text(result, args.api):
            raise ValueError("thinking response has incorrect content")
        results[effort] = result

    first = request(
        "/responses",
        {
            "model": "qwen3.8-27b-fp8",
            "input": "Remember the number 731. Reply OK.",
            "reasoning": {"effort": "off"},
            "max_output_tokens": 128,
        },
    )
    if request("/responses/" + first["id"])["id"] != first["id"]:
        raise ValueError("stored response retrieval returned a different ID")
    second = request(
        "/responses",
        {
            "model": "qwen3.8-27b-fp8",
            "previous_response_id": first["id"],
            "input": "What number did I ask you to remember?",
            "reasoning": {"effort": "off"},
            "max_output_tokens": 128,
        },
    )
    if "731" not in json.dumps(second["output"]):
        raise ValueError("stored response continuation lost the remembered value")
    request("/responses/" + first["id"], method="DELETE")
    try:
        request("/responses/" + first["id"])
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise ValueError("deleted response did not return HTTP 404") from error
    else:
        raise AssertionError("deleted response still exists")
    results["stored"] = [first, second]

    def mixed(constrained):
        body = generation_body(
            args.api,
            "Return a JSON object with n equal to 123."
            if constrained
            else "计算 12+13,只回答结果。",
        )
        if constrained:
            if args.api == "chat":
                body["response_format"] = {"type": "json_object"}
            else:
                body["text"] = {"format": {"type": "json_object"}}
        result = request(endpoint, body)
        content = response_text(result, args.api)
        if constrained:
            if json.loads(content).get("n") != 123:
                raise ValueError("mixed constrained response has incorrect content")
        else:
            if "25" not in content:
                raise ValueError("mixed plain response has incorrect content")
        return result

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results["mixed"] = list(pool.map(mixed, [True, False]))
    mixed_ids = {int(result["id"].rsplit("_", 1)[1]) for result in results["mixed"]}
    body = generation_body(
        args.api, "Write a very long essay about mathematics.", max_tokens=8192
    )
    body["stream"] = True
    canceled_id = None
    with attempt.stream(args.base_url + endpoint, body) as response:
        for line in response:
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                event = json.loads(line[6:])
                canceled_id, has_output = stream_identity(event, args.api, canceled_id)
                if has_output:
                    break
        else:
            raise AssertionError("stream ended before output")
    results["after_disconnect"] = mixed(False)
    deadline = time.monotonic() + 10
    while True:
        with args.server_log.open("rb") as log:
            log.seek(log_offset)
            lines = log.read().decode().splitlines()
        batches, released, canceled = [], set(), set()
        for line in lines:
            ids = re.search(r"request_ids=(\[[^\]]*\])", line)
            if ids:
                values = set(json.loads(ids.group(1)))
                if "scheduled mixed batch" in line:
                    batches.append(values)
                elif "worker requests released" in line:
                    released.update(values)
            rid = re.search(r"request_id=(\d+)\b", line)
            if rid and ("request cancelled" in line or "request aborted" in line):
                canceled.add(int(rid.group(1)))
        if (
            any(mixed_ids <= batch for batch in batches)
            and canceled_id in canceled
            and canceled_id in released
        ):
            break
        if time.monotonic() >= deadline:
            raise AssertionError(
                {
                    "mixed_ids": sorted(mixed_ids),
                    "batches": [sorted(batch) for batch in batches],
                    "canceled_id": canceled_id,
                    "canceled": sorted(canceled),
                    "released": sorted(released),
                }
            )
        time.sleep(0.1)
    results["log_verification"] = {
        "mixed_request_ids": sorted(mixed_ids),
        "canceled_request_id": canceled_id,
        "same_batch": True,
        "worker_released": True,
    }
    if log_identity != (args.server_log.stat().st_dev, args.server_log.stat().st_ino):
        raise ValueError("server log changed identity during lifecycle acceptance")
    completed = [
        results[effort] for effort in ("off", "low", "medium", "high", "xhigh")
    ]
    completed.extend([first, second, *results["mixed"], results["after_disconnect"]])
    ids = {int(response["id"].rsplit("_", 1)[1]) for response in completed}
    results["draft_verification"] = read_service_evidence(
        args.server_log, log_offset, args.speculative_mode, request_ids=ids
    )
    results["speculative_mode"] = args.speculative_mode
    results["api"] = args.api
    attempt.update(results=results)
    print(
        "thinking, stored continuation/delete, mixed batch, disconnect follow-up PASS"
    )


if __name__ == "__main__":
    main()
