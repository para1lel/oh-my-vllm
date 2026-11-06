import ast
import http.client
import json
import time
from contextlib import closing
from urllib.parse import quote

from evidence import save_json, timestamp


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def events(text):
    return [
        json.loads(line[5:])
        for line in text.splitlines()
        if line.startswith("data:") and line[5:].strip() != "[DONE]"
    ]


class Http:
    def __init__(self, port, artifacts, processes):
        self.port = port
        self.artifacts = artifacts
        self.processes = processes
        self.counter = 0
        self.model = None

    def request(self, label, method, path, body=None, expected=200):
        self.counter += 1
        directory = self.artifacts / "http" / f"{self.counter:03d}-{label}"
        headers = {"Content-Type": "application/json"}
        save_json(
            directory / "request.json",
            {
                "method": method,
                "url": f"http://127.0.0.1:{self.port}{path}",
                "headers": headers,
                "body": body,
                "started_at": timestamp(),
            },
        )
        self.processes.observe()
        start = time.monotonic()
        try:
            with closing(
                http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            ) as connection:
                connection.request(
                    method, path, None if body is None else json.dumps(body), headers
                )
                response = connection.getresponse()
                content = response.read()
                (directory / "response.body").write_bytes(content)
                save_json(
                    directory / "response.json",
                    {
                        "status": response.status,
                        "headers": response.getheaders(),
                        "elapsed_seconds": time.monotonic() - start,
                    },
                )
            require(
                response.status == expected,
                f"{label}: HTTP {response.status}, expected {expected}",
            )
            return content.decode()
        except Exception as error:
            save_json(directory / "error.json", {"error": str(error)})
            raise

    def json(self, label, method, path, body=None, expected=200):
        return json.loads(self.request(label, method, path, body, expected))

    def chat(self, **extra):
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 128,
            "reasoning_effort": "off",
            "seed": 7,
            **extra,
        }

    def response(self, **extra):
        return {
            "model": self.model,
            "input": "hello",
            "max_output_tokens": 128,
            "reasoning": {"effort": "off"},
            "seed": 7,
            **extra,
        }


def drive_chat(http):
    plain = http.json("chat-json", "POST", "/v1/chat/completions", http.chat())
    stream = http.request(
        "chat-sse",
        "POST",
        "/v1/chat/completions",
        http.chat(stream=True, stream_options={"include_usage": True}),
    )
    chunks = events(stream)
    text = "".join(
        choice["delta"].get("content", "")
        for chunk in chunks
        for choice in chunk["choices"]
    )
    require("[DONE]" in stream, "Chat SSE has no completion marker")
    require(text == "test answer", "Chat fixture text differs")
    require(
        text == plain["choices"][0]["message"]["content"], "Chat JSON/SSE text differs"
    )
    finishes = [
        choice["finish_reason"]
        for chunk in chunks
        for choice in chunk["choices"]
        if choice.get("finish_reason") is not None
    ]
    require(
        finishes == [plain["choices"][0]["finish_reason"]],
        "Chat termination differs",
    )
    require(plain["usage"]["completion_tokens"] > 0, "Chat has no output token usage")
    require(
        chunks[-1]["usage"]["completion_tokens"] == plain["usage"]["completion_tokens"],
        "Chat completion token count differs",
    )
    return {
        "text": text,
        "json_usage": plain["usage"],
        "sse_usage": chunks[-1]["usage"],
    }


def drive_responses(http):
    parent = http.json("response-create", "POST", "/v1/responses", http.response())
    parent_path = "/v1/responses/" + quote(parent["id"], safe="")
    saved = http.json("response-get", "GET", parent_path)
    require(saved == parent, "Stored response differs")
    child = http.json(
        "response-continue",
        "POST",
        "/v1/responses",
        http.response(previous_response_id=parent["id"], input="continue"),
    )
    require(
        child["previous_response_id"] == parent["id"], "Continuation lost its parent"
    )
    http.json("response-delete", "DELETE", parent_path)
    http.json("response-missing", "GET", parent_path, expected=404)
    child_path = "/v1/responses/" + quote(child["id"], safe="")
    require(
        http.json("response-child", "GET", child_path) == child, "Child was removed"
    )
    http.json("response-delete-child", "DELETE", child_path)
    stream = http.request(
        "responses-sse",
        "POST",
        "/v1/responses",
        http.response(stream=True, store=False),
    )
    chunks = events(stream)
    require(
        chunks[-1]["type"] == "response.completed", "Responses SSE did not complete"
    )
    require(
        [chunk["sequence_number"] for chunk in chunks] == list(range(len(chunks))),
        "Responses event sequence is not continuous",
    )
    return {"parent_id": parent["id"], "child_id": child["id"], "events": len(chunks)}


def drive_tools(http):
    schema = {
        "type": "object",
        "properties": {"key": {"type": "string", "enum": ["hello"]}},
        "required": ["key"],
        "additionalProperties": False,
    }
    function = {"name": "lookup", "parameters": schema, "strict": True}
    chat = http.json(
        "chat-tools",
        "POST",
        "/v1/chat/completions",
        http.chat(
            tools=[{"type": "function", "function": function}], tool_choice="required"
        ),
    )
    call = chat["choices"][0]["message"]["tool_calls"][0]["function"]
    require(call["name"] == "lookup", "Tool name differs")
    require(json.loads(call["arguments"]) == {"key": "hello"}, "Tool arguments differ")
    response = http.json(
        "responses-tools",
        "POST",
        "/v1/responses",
        http.response(
            tools=[{"type": "function", **function}],
            tool_choice="required",
            max_output_tokens=256,
            store=False,
        ),
    )
    require(
        response["output"][0]["type"] == "function_call", "Responses has no tool call"
    )
    result = http.json(
        "chat-schema",
        "POST",
        "/v1/chat/completions",
        http.chat(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": schema},
            }
        ),
    )
    content = json.loads(result["choices"][0]["message"]["content"])
    require(content == {"key": "hello"}, "Structured output differs")
    return {"tool": call, "structured_output": content}


def drive_operations(http, processes, command):
    http.json(
        "invalid-request", "POST", "/v1/chat/completions", http.chat(n=2), expected=400
    )
    diagnostics = http.json("diagnostics", "GET", "/diagnostics")
    require(diagnostics["active_requests"] == 0, "Service is not idle")
    offline = {
        "run": ["run", "--tokens", "1", "2", "--max-tokens", "8"],
        "bench": [
            "bench",
            "--batch-size",
            "2",
            "--input-len",
            "786",
            "--output-len",
            "8",
            "--prefix-hit",
            "--warmup",
            "0",
            "--repetitions",
            "1",
        ],
    }
    for label, arguments in offline.items():
        processes.start(label, command(label) + arguments)
        directory = processes.wait(label, 10)
        output = (directory / "stdout.txt").read_text()
        if label == "run":
            line = next(
                line
                for line in output.splitlines()
                if line.startswith("output token ids:")
            )
            require(
                len(ast.literal_eval(line.split(":", 1)[1])) == 8,
                "Run output length differs",
            )
        else:
            line = next(
                line for line in output.splitlines() if line.startswith("BENCH_RESULT ")
            )
            row = json.loads(line.split(" ", 1)[1])
            require(
                row["initial_prefix_hit_tokens"] == 1570, "Bench prefix reuse differs"
            )
    return {"diagnostics": diagnostics, "offline_scope": "fixture protocol only"}


def drive(feature, http, processes, command):
    actions = {"chat": drive_chat, "responses": drive_responses, "tools": drive_tools}
    selected = [*actions, "operations"] if feature == "all" else [feature]
    results = {}
    for name in selected:
        results[name] = (
            drive_operations(http, processes, command)
            if name == "operations"
            else actions[name](http)
        )
        save_json(http.artifacts / "features" / f"{name}.json", results[name])
    return results
