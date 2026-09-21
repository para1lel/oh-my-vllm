"""Exercise real serving thinking, state, mixed batching and disconnect recovery.

Run against a UUID-pinned MTP4 service. Check server cancellation/MTP logs too.
"""

import argparse
import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--server-log",
        type=Path,
        required=True,
        help="Serving DEBUG log, used to prove batching and worker cleanup",
    )
    args = parser.parse_args()
    log_offset = args.server_log.stat().st_size

    def request(path, body=None, method=None):
        req = urllib.request.Request(
            args.base_url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)

    results = {}
    for effort in ["off", "low", "medium", "high", "xhigh"]:
        result = request(
            "/chat/completions",
            {
                "model": "qwen3.5-27b-fp8",
                "messages": [{"role": "user", "content": "计算 12+13,只回答结果。"}],
                "reasoning_effort": effort,
                "max_tokens": 1024,
            },
        )
        assert "25" in result["choices"][0]["message"]["content"], result
        results[effort] = result

    first = request(
        "/responses",
        {
            "model": "qwen3.5-27b-fp8",
            "input": "Remember the number 731. Reply OK.",
            "reasoning": {"effort": "off"},
            "max_output_tokens": 128,
        },
    )
    assert request("/responses/" + first["id"])["id"] == first["id"]
    second = request(
        "/responses",
        {
            "model": "qwen3.5-27b-fp8",
            "previous_response_id": first["id"],
            "input": "What number did I ask you to remember?",
            "reasoning": {"effort": "off"},
            "max_output_tokens": 128,
        },
    )
    assert "731" in json.dumps(second["output"]), second
    request("/responses/" + first["id"], method="DELETE")
    try:
        request("/responses/" + first["id"])
    except urllib.error.HTTPError as error:
        assert error.code == 404
    else:
        raise AssertionError("deleted response still exists")
    results["stored"] = [first, second]

    def mixed(constrained):
        body = {
            "model": "qwen3.5-27b-fp8",
            "messages": [
                {
                    "role": "user",
                    "content": "Return a JSON object with n equal to 123."
                    if constrained
                    else "计算 12+13,只回答结果。",
                }
            ],
            "reasoning_effort": "off",
            "max_tokens": 128,
        }
        if constrained:
            body["response_format"] = {"type": "json_object"}
        result = request("/chat/completions", body)
        content = result["choices"][0]["message"]["content"]
        if constrained:
            assert json.loads(content)["n"] == 123, result
        else:
            assert "25" in content, result
        return result

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results["mixed"] = list(pool.map(mixed, [True, False]))
    mixed_ids = {int(result["id"].rsplit("_", 1)[1]) for result in results["mixed"]}
    req = urllib.request.Request(
        args.base_url + "/chat/completions",
        data=json.dumps(
            {
                "model": "qwen3.5-27b-fp8",
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a very long essay about mathematics.",
                    }
                ],
                "stream": True,
                "max_tokens": 8192,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        for line in response:
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                event = json.loads(line[6:])
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content"):
                        canceled_id = int(event["id"].rsplit("_", 1)[1])
                        break
                else:
                    continue
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
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(
        "thinking, stored continuation/delete, mixed batch, disconnect follow-up PASS"
    )


if __name__ == "__main__":
    main()
