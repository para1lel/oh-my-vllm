"""Exercise real JSON and strict tool generation through both HTTP APIs.

Start the UUID-pinned service with MTP4 first. This records request/response
evidence; also inspect matching request IDs and draft counters in server logs.
Passing against a scripted worker is not real model acceptance.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
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
            body = {"model": "qwen3.5-27b-fp8", "temperature": 0}
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
                    fmt.update({"json_schema": details} if api == "chat" else details)
                body.update(
                    {"response_format": fmt}
                    if api == "chat"
                    else {"text": {"format": fmt}}
                )
            started = time.monotonic()
            req = urllib.request.Request(
                args.base_url.rstrip("/") + "/" + path,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as response:
                    result = json.load(response)
                (OUT / (key + ".json")).write_text(
                    json.dumps(
                        {
                            "request": body,
                            "response": result,
                            "elapsed_s": time.monotonic() - started,
                        },
                        indent=2,
                    )
                )
                if api == "chat":
                    choice = result["choices"][0]
                    assert choice["finish_reason"] == (
                        "tool_calls" if kind == "tool" else "stop"
                    ), choice
                    if kind == "tool":
                        calls = choice["message"]["tool_calls"]
                        assert (
                            len(calls) == 1 and calls[0]["function"]["name"] == "report"
                        )
                        output = calls[0]["function"]["arguments"]
                    else:
                        output = choice["message"]["content"]
                else:
                    assert result["status"] == "completed", result
                    if kind == "tool":
                        calls = [
                            item
                            for item in result["output"]
                            if item["type"] == "function_call"
                        ]
                        assert len(calls) == 1 and calls[0]["name"] == "report"
                        output = calls[0]["arguments"]
                    else:
                        output = "".join(
                            part["text"]
                            for item in result["output"]
                            if item["type"] == "message"
                            for part in item["content"]
                            if part["type"] == "output_text"
                        )
                decoded = json.loads(output)
                assert isinstance(decoded, dict), decoded
                if kind != "json_object":
                    assert decoded == {"n": 123, "label": "verified"}, decoded
                print(
                    key,
                    "PASS",
                    round(time.monotonic() - started, 3),
                    "seconds",
                    "id=",
                    result["id"],
                    flush=True,
                )
            except Exception as error:
                if hasattr(error, "read"):
                    print(error.read().decode(), flush=True)
                raise
