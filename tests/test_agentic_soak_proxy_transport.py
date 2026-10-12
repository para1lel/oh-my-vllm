"""Stream transport regressions use captured bytes and deterministic exceptions."""

import copy
import hashlib
import io
import json

import pytest

from benchmarks.agentic_soak_protocol import StreamCompletion, attach_client_completions
from benchmarks.agentic_soak_proxy import record_transport_error, relay_stream

CHAT_ID = "chatcmpl_test_1"
RESPONSE_ID = "resp_test_1"


def chat_event(reason=None, identity=CHAT_ID):
    return (
        b"data: "
        + json.dumps(
            {
                "id": identity,
                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
            }
        ).encode()
        + b"\n\n"
    )


def chat_stream(reason="stop"):
    return chat_event() + chat_event(reason) + b"data: [DONE]\n\n"


def responses_stream(kind="response.completed", *, tool=False):
    response = {
        "id": RESPONSE_ID,
        "status": "completed",
        "output": [{"type": "function_call" if tool else "message"}],
    }
    if kind == "response.incomplete":
        response.update(
            status="incomplete", incomplete_details={"reason": "max_output_tokens"}
        )
    return (
        b"event: "
        + kind.encode()
        + b"\ndata: "
        + json.dumps({"type": kind, "response": response}).encode()
        + b"\n\n"
    )


def parse(api, body, fragment=1):
    proof = StreamCompletion(api, 1 << 20)
    for offset in range(0, len(body), fragment):
        proof.feed(body[offset : offset + fragment])
    return proof.snapshot()


@pytest.mark.parametrize("reason", ["stop", "tool_calls", "length"])
def test_fragmented_chat_terminal_requires_finish_and_done(reason):
    body = chat_stream(reason)
    proof = parse("chat", body)
    assert proof["success"]
    assert proof["response_id"] == CHAT_ID
    assert proof["finish_reason"] == reason
    assert proof["terminal_kind"] == "chat_done"
    assert 0 < proof["raw_terminal_offset"] <= len(body)
    assert proof["terminal_forwarded"] is False


@pytest.mark.parametrize(
    "kind,tool,reason,terminal",
    [
        ("response.completed", False, "stop", "responses_completed"),
        ("response.completed", True, "tool_calls", "responses_completed"),
        ("response.incomplete", False, "length", "responses_incomplete"),
    ],
)
def test_fragmented_responses_terminal_maps_actual_service_stop(
    kind, tool, reason, terminal
):
    proof = parse("responses", responses_stream(kind, tool=tool))
    assert proof["success"]
    assert proof["response_id"] == RESPONSE_ID
    assert proof["finish_reason"] == reason
    assert proof["terminal_kind"] == terminal


@pytest.mark.parametrize(
    "body",
    [
        chat_event("stop"),
        chat_event() + b"data: [DONE]\n\n",
        chat_event("stop") + chat_event("length") + b"data: [DONE]\n\n",
        chat_event() + chat_event("stop", "chatcmpl_foreign") + b"data: [DONE]\n\n",
        chat_event("stop") + b"data: {broken}\n\ndata: [DONE]\n\n",
        chat_event("stop")
        + b'data: {"error":{"message":"failure"}}\n\ndata: [DONE]\n\n',
        b'data: {"id":"chatcmpl_test_1","choices":null}\n\ndata: [DONE]\n\n',
        chat_stream() + b"data: [DONE]\n\n",
    ],
)
def test_invalid_or_incomplete_chat_cannot_prove_completion(body):
    assert parse("chat", body)["success"] is False


@pytest.mark.parametrize("mutation", ["status", "reason", "identity", "error", "event"])
def test_responses_rejects_invalid_terminal_and_error(mutation):
    body = responses_stream("response.incomplete")
    if mutation == "status":
        body = body.replace(b'"status": "incomplete"', b'"status": "completed"')
    elif mutation == "reason":
        body = body.replace(b"max_output_tokens", b"content_filter")
    elif mutation == "identity":
        body = responses_stream() + body.replace(RESPONSE_ID.encode(), b"resp_foreign")
    elif mutation == "error":
        body = b'data: {"type":"error","error":{"message":"failed"}}\n\n' + body
    elif mutation == "event":
        body = body.replace(b"event: response.incomplete", b"event: response.completed")
    assert parse("responses", body)["success"] is False


def test_unbounded_sse_line_is_rejected_without_claiming_terminal():
    proof = StreamCompletion("chat", 16)
    with pytest.raises(ValueError, match="bound"):
        proof.feed(b"data: " + b"x" * 20)
    assert proof.snapshot()["success"] is False


class Reader:
    def __init__(self, pieces):
        self.pieces = iter(pieces)
        self.calls = 0

    def read(self, size):
        self.calls += 1
        piece = next(self.pieces, b"")
        if isinstance(piece, BaseException):
            raise piece
        return piece

    read1 = read


def relay(pieces, *, api="chat", write=None):
    reader, output, record = Reader(pieces), io.BytesIO(), {}
    delivered = bytearray()
    try:
        relay_stream(reader, output, write or delivered.extend, record, api)
    except Exception as error:
        record_transport_error(record, error)
    return reader, output.getvalue(), record, bytes(delivered)


@pytest.mark.parametrize(
    "api,body", [("chat", chat_stream()), ("responses", responses_stream())]
)
def test_complete_terminal_stops_before_upstream_eof_or_reset(api, body):
    reader, raw, record, delivered = relay(
        [body, ConnectionResetError("after terminal")], api=api
    )
    assert reader.calls == 1
    assert raw == delivered == body
    assert record["response_sha256"] == hashlib.sha256(body).hexdigest()
    assert record["response_bytes"] == len(body)
    assert record["transport_outcome"] == "completed"
    assert "error" not in record
    assert record["protocol_completion"]["terminal_forwarded"] is True


@pytest.mark.parametrize("exception", [ConnectionResetError, BrokenPipeError])
def test_downstream_disconnect_after_captured_terminal_remains_candidate(exception):
    def write(chunk):
        raise exception("client closed after consuming finish")

    body = chat_stream("tool_calls")
    _reader, raw, record, _delivered = relay([body], write=write)
    assert raw == body
    assert record["error"].startswith(exception.__name__ + ":")
    assert record["transport_error_kind"] == exception.__name__
    assert record["error_direction"] == "downstream_write"
    assert record["transport_outcome"] == "terminal_disconnect"
    assert record["protocol_completion"]["success"] is True
    assert record["protocol_completion"]["terminal_forwarded"] is False


@pytest.mark.parametrize("direction", ["upstream", "downstream"])
def test_reset_before_terminal_is_always_failed(direction):
    pieces, write = [chat_event(), ConnectionResetError("truncated")], None
    if direction == "downstream":

        def write(chunk):
            raise ConnectionResetError("early close")

    _reader, raw, record, _delivered = relay(pieces, write=write)
    assert raw == chat_event()
    assert record["transport_outcome"] == "failed"
    assert record["protocol_completion"]["success"] is False
    assert record["error_direction"] == direction + (
        "_read" if direction == "upstream" else "_write"
    )


def test_clean_eof_without_protocol_terminal_is_a_preserved_failure():
    _reader, _raw, record, _delivered = relay([chat_event("stop"), b""])
    assert record["transport_outcome"] == "failed"
    assert record["transport_error_kind"] == "ValueError"
    assert "missing_successful_terminal" in record["protocol_completion"]["errors"]


@pytest.mark.parametrize(
    "reason", ["upstream_headers", "request_read", "protocol_parse"]
)
def test_reset_outside_stream_transport_is_never_a_terminal_disconnect(reason):
    record = {"error_direction": reason, "protocol_completion": {"success": True}}
    record_transport_error(record, ConnectionResetError("reset"))
    assert record["transport_outcome"] == "failed"


def client_message(api="chat", reason="stop"):
    return {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "responseId": CHAT_ID if api == "chat" else RESPONSE_ID,
            "api": "openai-completions" if api == "chat" else "openai-responses",
            "model": "qwen3.8-27b-fp8",
            "stopReason": reason,
            "content": [{"type": "text", "text": "complete"}],
        },
    }


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("reason", ["stop", "toolUse", "length"])
def test_client_completion_binds_unique_consumed_same_model_response(api, reason):
    event = client_message(api, reason)
    request = {"response_ids": [event["message"]["responseId"]]}
    attach_client_completions([event], [request], api)
    proof = request["client_completion"]
    assert proof["successful"] is True
    assert proof["message_count"] == 1
    assert proof["response_id"] == event["message"]["responseId"]
    assert proof["stop_reason"] == reason
    assert (
        proof["message_sha256"]
        == hashlib.sha256(
            json.dumps(event["message"], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "id",
        "api",
        "model",
        "stop",
        "error",
        "content",
        "request_ids",
    ],
)
def test_client_consumption_missing_duplicate_failed_or_unbound_is_not_success(
    mutation,
):
    event = client_message()
    events, request = [event], {"response_ids": [CHAT_ID]}
    if mutation == "missing":
        events = []
    elif mutation == "duplicate":
        events.append(copy.deepcopy(event))
    elif mutation == "request_ids":
        request["response_ids"].append("chatcmpl_foreign")
    else:
        key, value = {
            "id": ("responseId", "chatcmpl_foreign"),
            "api": ("api", "openai-responses"),
            "model": ("model", "foreign"),
            "stop": ("stopReason", "error"),
            "error": ("errorMessage", "failure"),
            "content": ("content", []),
        }[mutation]
        event["message"][key] = value
    attach_client_completions(events, [request], "chat")
    assert request["client_completion"]["successful"] is False


@pytest.mark.parametrize(
    "api,body", [("chat", chat_stream()), ("responses", responses_stream())]
)
def test_terminal_without_complete_sse_delimiter_is_not_completion(api, body):
    assert body.endswith(b"\n\n")
    assert parse(api, body[:-1])["success"] is False
    _reader, raw, record, _delivered = relay([body[:-1], b""], api=api)
    assert raw == body[:-1]
    assert record["transport_outcome"] == "failed"


@pytest.mark.parametrize(
    "api,body", [("chat", chat_stream()), ("responses", responses_stream())]
)
def test_sse_delimiter_split_across_reads_is_required_before_stopping(api, body):
    reader, raw, record, delivered = relay(
        [body[:-1], b"\n", ConnectionResetError("after terminal")], api=api
    )
    assert reader.calls == 2
    assert raw == delivered == body
    assert record["protocol_completion"]["raw_terminal_offset"] == len(body)
    assert record["transport_outcome"] == "completed"


@pytest.mark.parametrize(
    "api,body", [("chat", chat_stream()), ("responses", responses_stream())]
)
def test_crlf_delimiter_is_preserved_and_byte_fragmented(api, body):
    crlf = body.replace(b"\n", b"\r\n")
    proof = parse(api, crlf)
    assert proof["success"]
    assert proof["raw_terminal_offset"] == len(crlf)


def test_multiple_data_lines_are_dispatched_as_one_bounded_sse_event():
    body = (
        b'data: {"id":"chatcmpl_test_1",\n'
        b'data: "choices":[{"index":0,"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    assert parse("chat", body)["success"]
    proof = StreamCompletion("chat", 70)
    with pytest.raises(ValueError, match="bound"):
        proof.feed(body)
    assert proof.snapshot()["success"] is False
