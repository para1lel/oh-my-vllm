"""Real MTP constrained generation and prefix recovery beyond 131072 input tokens."""

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path


def read_draft_counts(server_log: Path, offset: int, expected: int = 4) -> list[int]:
    """Require one nonzero MTP proposal count per request in a dedicated server log."""
    with server_log.open("rb") as source:
        source.seek(0, 2)
        if source.tell() < offset:
            raise ValueError("server log was truncated during acceptance")
        source.seek(offset)
        tail = source.read().decode("utf-8")
    completed = [line for line in tail.splitlines() if "request completed" in line]
    if len(completed) != expected:
        raise ValueError(
            f"expected {expected} completed requests, found {len(completed)}"
        )
    counts = []
    for line in completed:
        match = re.search(r"\bproposed_draft_tokens=(\d+)\b", line)
        if match is None or int(match.group(1)) <= 0:
            raise ValueError("completed request has no positive proposed_draft_tokens")
        counts.append(int(match.group(1)))
    return counts


def validate_response(text: str, count: int, cached: int, repeat: int) -> None:
    """Reject invalid content, short prompts, or missing repeat prefix hits."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("long-context response is not JSON") from exc
    if value != {"n": 123, "label": "verified"}:
        raise ValueError("long-context JSON response has incorrect content")
    if not isinstance(count, int) or count < 131072:
        raise ValueError("long-context prompt token count is below 131072")
    if repeat and (not isinstance(cached, int) or cached < 130000):
        raise ValueError("long-context repeat has fewer than 130000 cached tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18015/v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    if not args.server_log.is_file():
        raise FileNotFoundError(
            f"dedicated MTP server log is missing: {args.server_log}"
        )
    log_identity = (args.server_log.stat().st_dev, args.server_log.stat().st_ino)
    log_offset = args.server_log.stat().st_size
    base = args.base_url.rstrip("/") + "/"
    tok = AutoTokenizer.from_pretrained("/data0/shared/Qwen3.8-27B-FP8")
    unit = tok.encode(
        "参考资料:北京是中国的首都,拥有丰富的历史文化。", add_special_tokens=False
    )
    prompt = (
        tok.decode((unit * (131072 // len(unit) + 1))[:131072])
        + "\nReturn only JSON with n=123 and label=verified."
    )
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "enum": [123]},
            "label": {"type": "string", "const": "verified"},
        },
        "required": ["n", "label"],
        "additionalProperties": False,
    }
    rows = []
    for api in ["chat/completions", "responses"]:
        for repeat in range(2):
            body = {"model": "qwen3.8-27b-fp8", "temperature": 0}
            if api == "chat/completions":
                body.update(
                    messages=[{"role": "user", "content": prompt}],
                    reasoning_effort="off",
                    max_completion_tokens=256,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "result",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )
            else:
                body.update(
                    input=prompt,
                    reasoning={"effort": "off"},
                    max_output_tokens=256,
                    store=False,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "result",
                            "strict": True,
                            "schema": schema,
                        }
                    },
                )
            start = time.monotonic()
            with urllib.request.urlopen(
                urllib.request.Request(
                    base + api,
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=600,
            ) as response:
                out = json.load(response)
            if api == "chat/completions":
                text = out["choices"][0]["message"]["content"]
                usage = out["usage"]
                count = usage["prompt_tokens"]
                cached = usage["prompt_tokens_details"]["cached_tokens"]
            else:
                text = "".join(
                    content.get("text", "")
                    for item in out["output"]
                    for content in item.get("content", [])
                    if content.get("type") == "output_text"
                )
                usage = out["usage"]
                count = usage["input_tokens"]
                cached = usage["input_tokens_details"]["cached_tokens"]
            validate_response(text, count, cached, repeat)
            rows.append(
                {
                    "api": api,
                    "repeat": repeat,
                    "elapsed_s": time.monotonic() - start,
                    "response": out,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )
            print(api, repeat, count, cached, flush=True)
    current_identity = (args.server_log.stat().st_dev, args.server_log.stat().st_ino)
    if current_identity != log_identity:
        raise ValueError("server log changed identity during acceptance")
    counts = read_draft_counts(args.server_log, log_offset, expected=len(rows))
    for row, proposed in zip(rows, counts, strict=True):
        row["proposed_draft_tokens"] = proposed
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
