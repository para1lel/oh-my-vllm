"""CPU HTTP and evidence checks; actual model acceptance needs the B200 service."""

import copy
import importlib.util
import io
import json
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.acceptance_proxy import recording_proxy, response_ids
from scripts.serving_evidence import (
    ServiceAttempt,
    stream_identity,
    validate_constraint_response,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "agentic_acceptance", ROOT / "scripts/agentic-acceptance.py"
)
agentic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agentic)


def agent_evidence(api="chat"):
    events, history = [], []
    for index, (path, text) in enumerate(
        (
            ("crates/scheduler/src/lib.rs", "pub fn schedule() {\n  real_source\n}"),
            (
                "python/oh_my_vllm/worker/model_runner.py",
                "class OhMyVllmWorker:\n  real_source",
            ),
        )
    ):
        call = f"call_{index}"
        events.extend(
            [
                {
                    "type": "tool_execution_start",
                    "toolName": "read",
                    "toolCallId": call,
                    "args": {"path": path},
                },
                {
                    "type": "tool_execution_end",
                    "toolCallId": call,
                    "isError": False,
                    "result": {"content": [{"type": "text", "text": text}]},
                },
                {
                    "type": "message_end",
                    "message": {
                        "role": "toolResult",
                        "toolCallId": call,
                        "isError": False,
                        "content": [{"type": "text", "text": text}],
                    },
                },
            ]
        )
        history.append(
            {"role": "tool", "tool_call_id": call, "content": text}
            if api == "chat"
            else {"type": "function_call_output", "call_id": call, "output": text}
        )
    rid = "chatcmpl_test_17" if api == "chat" else "resp_test_17"
    events.append(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "api": "openai-completions" if api == "chat" else "openai-responses",
                "model": "qwen3.8-27b-fp8",
                "responseId": rid,
                "stopReason": "stop",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "See crates/scheduler/src/lib.rs and "
                            "python/oh_my_vllm/worker/model_runner.py."
                        ),
                    }
                ],
            },
        }
    )
    requests = [
        {
            "response_status": 200,
            "response_ids": [rid],
            "request": {
                "model": "qwen3.8-27b-fp8",
                "messages" if api == "chat" else "input": history,
            },
        }
    ]
    return events, requests


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_omp_source_reads_are_forwarded_to_the_matching_final_model_request(api):
    events, requests = agent_evidence(api)
    result = agentic.validate_agent_evidence(events, requests, api)
    assert result["tool_results_in_final_model_request"]
    assert result["requires_semantic_answer_review"]
    assert len(result["source_reads"]) == 2


def composite_agent_evidence():
    events, requests = agent_evidence("responses")
    history = requests[0]["request"]["input"]
    for index in range(2):
        call, item = f"call_{index}", f"fc_{index}"
        for event in events[index * 3 : index * 3 + 3]:
            target = event.get("message", event)
            target["toolCallId"] = f"{call}|{item}"
        history.insert(
            index * 2,
            {
                "type": "function_call",
                "call_id": call,
                "id": item,
                "name": "read",
                "arguments": json.dumps({"path": events[index * 3]["args"]["path"]}),
            },
        )
    events.insert(
        0,
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": f"call_{index}|fc_{index}",
                        "name": "read",
                        "arguments": {"path": events[index * 3]["args"]["path"]},
                    }
                    for index in range(2)
                ],
            },
        },
    )
    return events, requests


def test_omp_responses_compound_identity_is_bound_to_both_recorded_wire_ids():
    events, requests = composite_agent_evidence()
    result = agentic.validate_agent_evidence(events, requests, "responses")
    read = result["source_reads"]["crates/scheduler/src/lib.rs"]
    assert read["tool_call_id"] == "call_0"
    assert read["client_tool_call_id"] == "call_0|fc_0"
    assert result["tool_results_in_final_model_request"]


def test_omp_responses_omitted_item_ids_are_bound_to_recorded_client_tool_calls():
    events, requests = composite_agent_evidence()
    for item in requests[0]["request"]["input"]:
        if item.get("type") == "function_call":
            del item["id"]
    result = agentic.validate_agent_evidence(events, requests, "responses")
    assert result["source_reads"]["crates/scheduler/src/lib.rs"][
        "client_tool_call_id"
    ] == ("call_0|fc_0")


@pytest.mark.parametrize("failure", ["unissued_item", "wrong_source", "wrong_tool"])
def test_omp_responses_omitted_item_ids_cannot_substitute_other_client_evidence(
    failure,
):
    events, requests = composite_agent_evidence()
    history = requests[0]["request"]["input"]
    del history[0]["id"]
    if failure == "unissued_item":
        events[0]["message"]["content"][0]["id"] = "call_0|fc_unrelated"
    elif failure == "wrong_source":
        history[0]["arguments"] = json.dumps({"path": "README.md"})
    else:
        history[0]["name"] = "grep"
    with pytest.raises(
        ValueError, match=r"not forwarded|issued source-read call differs"
    ):
        agentic.validate_agent_evidence(events, requests, "responses")


def test_omp_responses_wire_item_id_also_needs_the_actual_client_issued_call():
    events, requests = composite_agent_evidence()
    del events[0]
    with pytest.raises(ValueError, match="not forwarded"):
        agentic.validate_agent_evidence(events, requests, "responses")


@pytest.mark.parametrize("failure", ["wrong_source", "wrong_tool"])
def test_omp_responses_client_issued_call_name_and_arguments_must_match(failure):
    events, requests = composite_agent_evidence()
    issued = events[0]["message"]["content"][0]
    if failure == "wrong_source":
        issued["arguments"]["path"] = "README.md"
    else:
        issued["name"] = "grep"
    with pytest.raises(ValueError, match="issued source-read call differs"):
        agentic.validate_agent_evidence(events, requests, "responses")


@pytest.mark.parametrize("failure", ["wrong_source", "wrong_item", "wrong_tool"])
def test_omp_responses_conflicting_wire_call_definitions_are_rejected(failure):
    events, requests = composite_agent_evidence()
    history = requests[0]["request"]["input"]
    conflicting = copy.deepcopy(history[0])
    if failure == "wrong_source":
        conflicting["arguments"] = json.dumps({"path": "README.md"})
    elif failure == "wrong_item":
        conflicting["id"] = "fc_unrelated"
    else:
        conflicting["name"] = "grep"
    history.append(conflicting)
    with pytest.raises(ValueError, match="ambiguous wire"):
        agentic.validate_agent_evidence(events, requests, "responses")


def test_omp_responses_identical_wire_call_history_keeps_the_same_binding():
    events, requests = composite_agent_evidence()
    history = requests[0]["request"]["input"]
    history.append(copy.deepcopy(history[0]))
    assert agentic.validate_agent_evidence(events, requests, "responses")[
        "tool_results_in_final_model_request"
    ]


@pytest.mark.parametrize(
    "failure", ["wrong_item", "wrong_call", "missing_call", "modified_result"]
)
def test_omp_responses_compound_identity_rejects_unbound_or_changed_evidence(failure):
    events, requests = composite_agent_evidence()
    history = requests[0]["request"]["input"]
    if failure == "wrong_item":
        history[0]["id"] = "fc_unrelated"
    elif failure == "wrong_call":
        history[0]["call_id"] = "call_unrelated"
    elif failure == "missing_call":
        del history[0]
    else:
        history[1]["output"] = "changed source contents"
    with pytest.raises(ValueError, match="not forwarded"):
        agentic.validate_agent_evidence(events, requests, "responses")


def test_omp_compound_identity_is_not_stripped_for_chat():
    events, requests = agent_evidence("chat")
    for event in events[:3]:
        event.get("message", event)["toolCallId"] += "|fc_0"
    with pytest.raises(ValueError, match="not forwarded"):
        agentic.validate_agent_evidence(events, requests, "chat")


def test_omp_responses_ambiguous_client_identity_does_not_pass():
    source = {
        "path": "crates/scheduler/src/lib.rs",
        "original_path": "crates/scheduler/src/lib.rs",
        "text": "pub fn schedule() {}",
    }
    success = {"call_0": source, "call_0|fc_0": source}
    history = [
        {
            "type": "function_call",
            "call_id": "call_0",
            "id": "fc_0",
            "name": "read",
            "arguments": json.dumps({"path": source["original_path"]}),
        }
    ]
    with pytest.raises(ValueError, match="ambiguous"):
        agentic.wire_source_reads(
            success,
            history,
            "responses",
            {"call_0|fc_0": {("read", json.dumps({"path": source["original_path"]}))}},
        )


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("suffix", [":31-320", ":31"])
def test_omp_source_read_ranges_keep_identity_and_forwarded_result(api, suffix):
    events, requests = agent_evidence(api)
    original = "crates/scheduler/src/lib.rs" + suffix
    events[0]["args"]["path"] = original
    result = agentic.validate_agent_evidence(events, requests, api)
    read = result["source_reads"]["crates/scheduler/src/lib.rs"]
    assert read["original_path"] == original
    assert read["line_range"] == [31, 320 if "-" in suffix else 31]


@pytest.mark.parametrize(
    "path",
    [
        "crates/scheduler/src/lib.rs:0",
        "crates/scheduler/src/lib.rs:31-0",
        "crates/scheduler/src/lib.rs:320-31",
        "crates/scheduler/src/lib.rs:31-320,324-470",
        "crates/scheduler/src/lib.rs:31-320:1",
        "crates/scheduler/src/lib.rs:31oops",
        "../oh-my-vllm-other/crates/scheduler/src/lib.rs:31-320",
    ],
)
def test_omp_invalid_read_suffix_or_external_path_cannot_prove_source_read(path):
    events, requests = agent_evidence()
    events[0]["args"]["path"] = path
    with pytest.raises(ValueError, match="required source files"):
        agentic.validate_agent_evidence(events, requests, "chat")


def test_omp_absolute_source_read_range_stays_inside_repository():
    path = str(ROOT / "crates/scheduler/src/lib.rs") + ":31-320"
    parsed = agentic.source_read_path(path)
    assert parsed == {
        "path": "crates/scheduler/src/lib.rs",
        "original_path": path,
        "line_range": [31, 320],
    }


def test_omp_schedule_accessor_cannot_substitute_for_scheduler_function():
    events, requests = agent_evidence()
    events[1]["result"]["content"][0]["text"] = "pub fn scheduled_request() {}"
    with pytest.raises(ValueError, match="required source files"):
        agentic.validate_agent_evidence(events, requests, "chat")


def test_omp_multiple_forwarded_source_ranges_are_retained_for_review():
    events, requests = agent_evidence()
    events[0]["args"]["path"] += ":31-320"
    continuation = copy.deepcopy(events[:3])
    text = "pub fn schedule() { /* continuation */ }"
    for event in continuation:
        if "toolCallId" in event:
            event["toolCallId"] = "call_scheduler_continuation"
        else:
            event["message"]["toolCallId"] = "call_scheduler_continuation"
    continuation[0]["args"]["path"] = "crates/scheduler/src/lib.rs:324-470"
    continuation[1]["result"]["content"][0]["text"] = text
    events[-1:-1] = continuation
    requests[0]["request"]["messages"].append(
        {
            "role": "tool",
            "tool_call_id": "call_scheduler_continuation",
            "content": text,
        }
    )
    result = agentic.validate_agent_evidence(events, requests, "chat")
    reads = [
        item
        for item in result["successful_source_reads"]
        if item["path"] == "crates/scheduler/src/lib.rs"
    ]
    assert [item["line_range"] for item in reads] == [[31, 320], [324, 470]]


@pytest.mark.parametrize(
    "failure",
    [
        "failed_read",
        "directory_only",
        "modified_result",
        "wrong_call",
        "wrong_response",
        "wrong_model",
        "tool_use_without_answer",
    ],
)
@pytest.mark.parametrize("api", ["chat", "responses"])
def test_omp_partial_or_unforwarded_reads_do_not_pass(failure, api):
    events, requests = agent_evidence(api)
    key = "messages" if api == "chat" else "input"
    if failure == "failed_read":
        events[1]["isError"] = True
    elif failure == "directory_only":
        events[0]["args"]["path"] = "crates/scheduler/src"
    elif failure == "modified_result":
        requests[0]["request"][key][0]["content" if api == "chat" else "output"] = (
            "pub fn schedule different_body"
        )
    elif failure == "wrong_call":
        requests[0]["request"][key][0][
            "tool_call_id" if api == "chat" else "call_id"
        ] = "unrelated"
    elif failure == "wrong_response":
        requests[0]["response_ids"] = ["resp_unrelated_20"]
    elif failure == "wrong_model":
        events[-1]["message"]["model"] = "other-model"
    else:
        events[-1]["message"]["stopReason"] = "toolUse"
    with pytest.raises(ValueError):
        agentic.validate_agent_evidence(events, requests, api)


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_proxy_forwards_exact_request_and_sse_bytes_and_releases_listener(api):
    seen = []
    rid = "chatcmpl_test_17" if api == "chat" else "resp_test_17"
    payload = (
        {"id": rid}
        if api == "chat"
        else {"type": "response.created", "response": {"id": rid}}
    )
    body = b"data: " + json.dumps(payload).encode() + b"\n\ndata: [DONE]\n\n"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            seen.append(
                (self.path, self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    path = "/chat/completions" if api == "chat" else "/responses"
    sent = json.dumps(
        {"model": "qwen3.8-27b-fp8", "stream": True, "input": "完整正文"},
        ensure_ascii=False,
    ).encode()
    try:
        with recording_proxy(f"http://127.0.0.1:{server.server_port}/v1") as (
            url,
            records,
        ):
            with urllib.request.urlopen(
                urllib.request.Request(url + path, data=sent), timeout=5
            ) as response:
                assert response.read() == body
            assert seen == [("/v1" + path, sent)]
            assert records[0]["request"] == json.loads(sent)
            assert records[0]["response_ids"] == [rid]
            assert records[0]["response_status"] == 200
        with pytest.raises(urllib.error.URLError):
            urllib.request.urlopen(url + path, timeout=0.2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_response_sse_identity_precedes_delta_and_wrong_identity_fails():
    created = {"type": "response.created", "response": {"id": "resp_test_18"}}
    delta = {"type": "response.output_text.delta", "delta": "answer"}
    rid, output = stream_identity(created, "responses")
    assert (rid, output) == (18, False)
    assert stream_identity(delta, "responses", rid) == (18, True)
    with pytest.raises(ValueError, match="identity"):
        stream_identity(delta, "responses")
    assert response_ids(
        b'data: {"type":"response.created","response":{"id":"resp_test_18"}}\n'
    ) == ["resp_test_18"]


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_constraint_completion_and_exact_schema_payload_are_checked(api):
    text = '{"n":123,"label":"verified"}'
    result = (
        {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
        if api == "chat"
        else {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": text}]}
            ],
        }
    )
    validate_constraint_response(result, api, "json_schema")
    bad = copy.deepcopy(result)
    if api == "chat":
        bad["choices"][0]["message"]["content"] = '{"n":124,"label":"verified"}'
    else:
        bad["output"][0]["content"][0]["text"] = '{"n":124,"label":"verified"}'
    with pytest.raises(ValueError, match="incorrect content"):
        validate_constraint_response(bad, api, "json_schema")


@pytest.mark.parametrize("kind", ["later_validation", "http_error", "stream"])
def test_failed_attempt_keeps_received_http_and_refuses_overwrite(tmp_path, kind):
    body = b'{"id":"resp_test_9","text":"full received response"}'
    frame = b'data: {"id":"resp_test_9"}\n'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(503 if self.path == "/error" else 200)
            self.end_headers()
            self.wfile.write(frame + b"\n" if self.path == "/stream" else body)

        do_POST = do_GET

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "attempt.json"
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with (
            pytest.raises((ValueError, urllib.error.HTTPError)),
            ServiceAttempt(output, {"speculative_mode": "dspark"}) as attempt,
        ):
            if kind == "http_error":
                attempt.request(url + "/error")
            elif kind == "stream":
                with attempt.stream(url + "/stream", None) as response:
                    assert next(response) == frame
                raise ValueError("later stream cleanup evidence failed")
            else:
                assert attempt.request(url + "/json")["id"] == "resp_test_9"
                saved = json.loads(output.read_text())
                assert saved["responses"][0]["response_text"] == body.decode()
                assert not saved["passed"]
                raise ValueError("later server-log evidence failed")
        saved = json.loads(output.read_text())
        assert not saved["passed"]
        assert saved["error"]["type"]
        assert saved["responses"][0]["response_text"] == (
            frame.decode() if kind == "stream" else body.decode()
        )
        previous = output.read_bytes()
        with pytest.raises(FileExistsError):
            ServiceAttempt(output, {"speculative_mode": "mtp"})
        assert output.read_bytes() == previous
        assert list(tmp_path.iterdir()) == [output]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("failure", ["retrieval", "continuation", "delete"])
def test_lifecycle_storage_contracts_hold_with_python_assertions_off(tmp_path, failure):
    path = ROOT / "scripts/serving-lifecycle.py"
    module = {"__name__": "optimized_acceptance", "__file__": str(path)}
    exec(compile(path.read_text(), str(path), "exec", optimize=2), module)
    server_log = tmp_path / "server.log"
    server_log.write_text('BENCH_CONFIG speculative_mode="dspark"\n')

    class Attempt:
        def __init__(self):
            self.number = 0

        def update(self, **_fields):
            pass

        def request(self, _url, _body=None, method=None):
            self.number += 1
            if self.number <= 5:
                return {"choices": [{"message": {"content": "25"}}]}
            if self.number == 6:
                return {"id": "resp_test_9"}
            if self.number == 7:
                return {"id": "wrong" if failure == "retrieval" else "resp_test_9"}
            if self.number == 8:
                return {"output": "wrong" if failure == "continuation" else "731"}
            if method == "DELETE":
                return {"deleted": True}
            raise urllib.error.HTTPError(_url, 503, "unavailable", None, io.BytesIO())

    args = SimpleNamespace(
        server_log=server_log,
        api="chat",
        base_url="http://unused",
        speculative_mode="dspark",
    )
    with pytest.raises(
        ValueError,
        match={
            "retrieval": "different ID",
            "continuation": "remembered value",
            "delete": "HTTP 404",
        }[failure],
    ):
        module["run"](args, Attempt())


@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT])
def test_service_attempt_cancellation_saves_responses_and_survives_second_signal(
    tmp_path, number
):
    output, ready = tmp_path / "attempt.json", tmp_path / "ready"
    program = """
import time
from pathlib import Path
from scripts.serving_evidence import ServiceAttempt
with ServiceAttempt(Path(__import__('sys').argv[1]), {}) as attempt:
    attempt.record({'response': {'id': 'resp_before_cancel'}})
    original = attempt.update
    def slow_update(**fields):
        time.sleep(0.3)
        original(**fields)
    attempt.update = slow_update
    Path(__import__('sys').argv[2]).touch()
    time.sleep(30)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(output), str(ready)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and process.poll() is None:
            if time.monotonic() > deadline:
                pytest.fail("CPU cancellation fixture did not start")
            time.sleep(0.01)
        process.send_signal(number)
        time.sleep(0.05)
        if process.poll() is None:
            process.send_signal(number)
        process.communicate(timeout=5)
        assert process.returncode != 0
        saved = json.loads(output.read_text())
        assert not saved["passed"]
        assert saved["error"]["type"] == "InterruptedError"
        assert saved["responses"][0]["response"]["id"] == "resp_before_cancel"
        assert not any(path.name.startswith("tmp") for path in tmp_path.iterdir())
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


@pytest.mark.parametrize("previous_handler_raises", [False, True])
def test_service_attempt_first_cancellation_during_handler_restore_is_saved(
    tmp_path, monkeypatch, previous_handler_raises
):
    output = tmp_path / "attempt.json"
    previous = {
        number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)
    }
    original = signal.signal
    triggered = False

    def restore_with_cancel(number, handler):
        nonlocal triggered
        if attempt.cleaning and not triggered:
            triggered = True
            if previous_handler_raises:
                original(number, handler)
                raise KeyboardInterrupt("Previous SIGINT handler cancelled")
            attempt._terminate(signal.SIGTERM, None)
        return original(number, handler)

    attempt = ServiceAttempt(output, {})
    monkeypatch.setattr(signal, "signal", restore_with_cancel)
    error_type = KeyboardInterrupt if previous_handler_raises else InterruptedError
    with pytest.raises(error_type), attempt:
        attempt.record({"response": {"id": "resp_before_exit"}})
        attempt.update(passed=True)
    saved = json.loads(output.read_text())
    assert triggered
    assert not saved["passed"]
    assert saved["error"]["type"] == error_type.__name__
    assert saved["responses"][0]["response"]["id"] == "resp_before_exit"
    assert all(
        signal.getsignal(number) == handler for number, handler in previous.items()
    )
