"""Framework evidence rejects corrupt streams independently of agent outcomes."""

import hashlib
import json

import pytest

from benchmarks.agentic_framework_evidence import audit_framework_session
from tests.support.framework_evidence import request_fixture as fixture


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_valid_framework_outcomes_do_not_require_native_task_success(
    tmp_path, api, cancelled
):
    session, metrics, _ = fixture(tmp_path, api, cancelled=cancelled)
    points, audit = audit_framework_session(tmp_path, session, metrics, 100)
    assert len(points) == (0 if cancelled else 1)
    assert audit["cancelled_requests"] == int(cancelled)
    assert audit["outcomes"][0]["status"] == ("cancelled" if cancelled else "completed")


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("failure", ["usage", "hash", "failed", "identity", "context"])
def test_task_scope_change_preserves_framework_rejection(tmp_path, api, failure):
    session, metrics, record = fixture(tmp_path, api)
    metric = next(iter(metrics.values()))
    if failure == "usage":
        metric["output_tokens"] = 7
    elif failure == "hash":
        (tmp_path / "wire/response.bin").write_bytes(b"changed")
    elif failure == "failed":
        metric["status"] = "failed"
    elif failure == "identity":
        record["response_ids"].append("different")
    else:
        record["context_proof"]["total_budget"] = 262145
    (tmp_path / "wire/http.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        audit_framework_session(tmp_path, session, metrics, 100)


@pytest.mark.parametrize(
    "failure", ["sequence", "param", "event", "extra", "after_error"]
)
def test_cancelled_status_does_not_hide_invalid_serving_errors(tmp_path, failure):
    session, metrics, record = fixture(tmp_path, "responses", cancelled=True)
    path = tmp_path / "wire/response.bin"
    raw = path.read_bytes()
    error = {
        "type": "error",
        "code": "server_error",
        "message": "request cancelled or timed out",
        "param": None,
        "sequence_number": 1,
    }
    original = next(
        line
        for line in raw.splitlines()
        if line.startswith(b"data: ") and json.loads(line[6:]).get("type") == "error"
    )
    if failure == "sequence":
        error["sequence_number"] = True
    elif failure == "param":
        error["param"] = "bad"
    elif failure == "extra":
        error["foreign"] = "extra"
    elif failure == "event":
        raw = raw.replace(b"event: error", b"event: response.completed")
    else:
        raw += b'data: {"type":"response.output_text.delta","delta":"late"}\n\n'
    raw = raw.replace(original, b"data: " + json.dumps(error).encode())
    path.write_bytes(raw)
    record["response_sha256"] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "wire/http.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        audit_framework_session(tmp_path, session, metrics, 100)


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_oneoff_contract_has_raw_usage_and_no_final_answer_gate(tmp_path, api):
    from benchmarks.agentic_framework_evidence import audit_oneoff_requests

    _session, metrics, record = fixture(tmp_path, api)
    record["request"] = json.loads((tmp_path / "wire/request.json").read_text())
    result = audit_oneoff_requests(
        [record], api, list(metrics.values()), tmp_path / "wire"
    )
    assert result["requests"] == 1
    assert result["task_outcomes_are_gates"] is False
    metric = next(iter(metrics.values()))
    metric["output_tokens"] = 7
    with pytest.raises(ValueError, match="usage"):
        audit_oneoff_requests([record], api, list(metrics.values()), tmp_path / "wire")


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("space", [b" ", b""])
def test_successful_terminal_cannot_hide_later_model_data(tmp_path, api, space):
    session, metrics, record = fixture(tmp_path, api)
    path = tmp_path / "wire/response.bin"
    raw = (
        path.read_bytes() + b'data: {"id":"chatcmpl_fixture_1","choices":['
        b'{"index":0,"delta":{"content":"late"}}]}\n\n'
    )
    raw = raw.replace(b"data: ", b"data:" + space)
    path.write_bytes(raw)
    record["response_sha256"] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "wire/http.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="successful SSE completion"):
        audit_framework_session(tmp_path, session, metrics, 100)


def test_responses_cancellation_rejects_chat_error_envelope(tmp_path):
    session, metrics, record = fixture(tmp_path, "responses", cancelled=True)
    path = tmp_path / "wire/response.bin"
    raw = (
        path.read_bytes().split(b"event: error")[0]
        + b'data: {"error":{"code":"server_error",'
        b'"message":"request cancelled or timed out","type":"server_error"}}\n\n'
    )
    path.write_bytes(raw)
    record["response_sha256"] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "wire/http.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="envelope"):
        audit_framework_session(tmp_path, session, metrics, 100)
