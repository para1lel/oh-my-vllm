"""Native SDK spans keep real identities, cache usage, purpose and API provenance."""

import hashlib
import http.client
import json
import socket
import time
from urllib.parse import urlsplit

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)

from benchmarks import agentic_sdk_trace as sdk
from benchmarks.agentic_sdk_trace import SdkTrace, read_sdk_spans
from benchmarks.agentic_session_evidence import trace_messages


def export(attributes=None, **changes):
    fields = {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "qwen3.8-27b-fp8",
        "gen_ai.response.model": "qwen3.8-27b-fp8",
        "gen_ai.provider.name": "oh-my-vllm-chat",
        "gen_ai.conversation.id": "fixed-session",
        "gen_ai.response.id": "response-1",
        "gen_ai.response.finish_reasons": ["stop"],
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.output_tokens": 3,
        "gen_ai.usage.cache_read.input_tokens": 80,
        "gen_ai.usage.cache_creation.input_tokens": 0,
        "omp.gen_ai.usage.total_tokens": 103,
        "omp.gen_ai.request.reasoning.effort": "max",
        "omp.gen_ai.oneshot.kind": "compaction_summary",
        "gen_ai.request.max_tokens": 16384,
    }
    fields.update(attributes or {})
    request = ExportTraceServiceRequest()
    scope = request.resource_spans.add().scope_spans.add()
    scope.scope.name = "@oh-my-pi/pi-agent-core"
    span = scope.spans.add(
        trace_id=b"\x01" * 16,
        span_id=b"\x02" * 8,
        start_time_unix_nano=1791840000000000000,
        end_time_unix_nano=1791840001000000000,
    )
    for key, value in changes.items():
        setattr(span, key, value)
    for key, value in fields.items():
        item = span.attributes.add(key=key)
        if isinstance(value, str):
            item.value.string_value = value
        elif type(value) is int:
            item.value.int_value = value
        elif isinstance(value, list):
            for text in value:
                item.value.array_value.values.add(string_value=text)
        else:
            item.value.bool_value = value
    return request.SerializeToString()


def saved_trace(tmp_path, raw):
    directory = tmp_path / "sdk-trace"
    directory.mkdir()
    (directory / "0000.pb").write_bytes(raw)
    (directory / "exports.jsonl").write_text(
        json.dumps(
            {
                "file": "0000.pb",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        + "\n"
    )
    (directory / "closed.json").write_text(
        json.dumps({"closed": True, "exports": 1, "bytes": len(raw), "error": None})
    )
    state = tmp_path / "state"
    state.mkdir()
    (state / "models.yml").write_text(
        json.dumps({"providers": {"oh-my-vllm-chat": {"api": "openai-completions"}}})
    )
    (state / "settings.yml").write_text("{}")
    (tmp_path / "states.json").write_text(
        json.dumps(
            [
                {
                    "sessionId": "fixed-session",
                    "model": {
                        "id": "qwen3.8-27b-fp8",
                        "provider": "oh-my-vllm-chat",
                        "api": "openai-completions",
                    },
                }
            ]
        )
    )
    return {
        "api": "chat",
        "session_id": "fixed-session",
        "sdk_trace": {
            "directory": "sdk-trace",
            "source": "omp_sdk_otlp",
            "api_binding": "openai-completions",
            "api_binding_source": "isolated_provider_configuration",
            "configuration_sha256": {
                name: hashlib.sha256((state / name).read_bytes()).hexdigest()
                for name in ("models.yml", "settings.yml")
            },
        },
    }


@pytest.mark.parametrize(
    "purpose,budget",
    [
        ("compaction_summary", 16384),
        ("compaction_turn_prefix", 16384),
        ("compaction_short_summary", 512),
        ("task", 32768),
    ],
)
def test_native_cache_total_is_not_counted_twice_and_api_is_configuration_binding(
    tmp_path, purpose, budget
):
    session = saved_trace(
        tmp_path,
        export(
            {"omp.gen_ai.oneshot.kind": purpose, "gen_ai.request.max_tokens": budget}
        ),
    )
    [message] = trace_messages(tmp_path, session)
    assert message["usage"] == {"input": 100, "output": 3}
    assert message["purpose"] == purpose
    assert "api" not in message
    assert message["api_binding_source"] == "isolated_provider_configuration"
    assert (
        message["sdk_trace"]["attributes"]["gen_ai.usage.cache_read.input_tokens"] == 80
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("gen_ai.usage.cache_read.input_tokens", -1),
        ("gen_ai.usage.cache_read.input_tokens", 101),
        ("gen_ai.usage.output_tokens", True),
        ("omp.gen_ai.usage.total_tokens", 999),
        ("gen_ai.usage.reasoning.output_tokens", 4),
        ("gen_ai.request.model", "other"),
        ("gen_ai.response.model", "other"),
        ("gen_ai.request.max_tokens", -1),
        ("gen_ai.request.max_tokens", 32769),
        ("omp.gen_ai.request.reasoning.effort", "high"),
        ("omp.gen_ai.oneshot.kind", "compaction_unknown"),
        ("gen_ai.conversation.id", "other-session"),
        ("gen_ai.response.finish_reasons", ["error"]),
        ("gen_ai.provider.name", "other"),
    ],
)
def test_wrong_native_result_cannot_supply_independent_completion(tmp_path, key, value):
    session = saved_trace(tmp_path, export({key: value}))
    with pytest.raises(ValueError, match="SDK result trace"):
        trace_messages(tmp_path, session)


@pytest.mark.parametrize(
    "changes",
    [
        {"start_time_unix_nano": 0},
        {"trace_id": b""},
        {"span_id": b"\0" * 8},
        {"end_time_unix_nano": 1},
    ],
)
def test_sdk_identity_and_lifecycle_must_be_complete(tmp_path, changes):
    session = saved_trace(tmp_path, export(**changes))
    with pytest.raises(ValueError, match="SDK result trace"):
        trace_messages(tmp_path, session)


@pytest.mark.parametrize(
    "mutation", ["raw", "error", "closed", "api", "runtime_api", "count"]
)
def test_replay_rejects_modified_export_failed_close_and_wrong_api(tmp_path, mutation):
    session = saved_trace(tmp_path, export())
    directory = tmp_path / "sdk-trace"
    if mutation == "raw":
        (directory / "0000.pb").write_bytes(b"changed")
    elif mutation == "error":
        (directory / "error.json").write_text('{"error":"receiver failure"}')
    elif mutation in ("closed", "count"):
        receipt = json.loads((directory / "closed.json").read_text())
        receipt["closed" if mutation == "closed" else "exports"] = False
        (directory / "closed.json").write_text(json.dumps(receipt))
    elif mutation == "api":
        session["sdk_trace"]["api_binding"] = "openai-responses"
    else:
        states = json.loads((tmp_path / "states.json").read_text())
        states[0]["model"]["api"] = "openai-responses"
        (tmp_path / "states.json").write_text(json.dumps(states))
    with pytest.raises(ValueError, match="SDK"):
        trace_messages(tmp_path, session)


def test_duplicate_export_retry_preserves_originals_but_two_sdk_completions_fail(
    tmp_path,
):
    raw = export()
    session = saved_trace(tmp_path, raw)
    directory = tmp_path / "sdk-trace"
    record = json.loads((directory / "exports.jsonl").read_text())
    record["file"] = "0001.pb"
    (directory / "0001.pb").write_bytes(raw)
    with (directory / "exports.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")
    (directory / "closed.json").write_text(
        json.dumps({"closed": True, "exports": 2, "bytes": 2 * len(raw)})
    )
    assert len(trace_messages(tmp_path, session)) == 1
    different = export(span_id=b"\x03" * 8)
    (directory / "0001.pb").write_bytes(different)
    record.update(bytes=len(different), sha256=hashlib.sha256(different).hexdigest())
    original = json.loads((directory / "exports.jsonl").read_text().splitlines()[0])
    (directory / "exports.jsonl").write_text(
        json.dumps(original) + "\n" + json.dumps(record) + "\n"
    )
    with pytest.raises(ValueError, match="SDK result trace"):
        trace_messages(tmp_path, session)


def test_real_chunked_export_is_saved_and_listener_closes(tmp_path):
    collector = SdkTrace(tmp_path / "sdk-trace")
    address = urlsplit(collector.endpoint)
    raw = export()
    client = http.client.HTTPConnection(address.hostname, address.port, timeout=5)
    try:
        client.request(
            "POST",
            address.path,
            iter([raw[:100], raw[100:]]),
            {"Content-Type": "application/x-protobuf"},
            encode_chunked=True,
        )
        assert client.getresponse().status == 200
        collector.wait_for({"response-1"}, timeout=0.1)
        with pytest.raises(ValueError, match="export all completed"):
            collector.wait_for({"missing"}, timeout=0.01)
    finally:
        client.close()
        collector.close()
    assert len(read_sdk_spans(tmp_path / "sdk-trace")) == 1
    with socket.socket() as connection:
        assert connection.connect_ex((address.hostname, address.port)) != 0


@pytest.mark.parametrize("headers", [b"", b"POST / HTTP/1.1\r\nHost:"])
def test_incomplete_headers_cannot_block_listener_cleanup(tmp_path, headers):
    collector = SdkTrace(tmp_path / "sdk-trace")
    address = urlsplit(collector.endpoint)
    client = socket.create_connection((address.hostname, address.port))
    client.sendall(headers)
    time.sleep(0.05)
    started = time.monotonic()
    try:
        if headers:
            with pytest.raises(ValueError, match="SDK export request"):
                collector.close()
        else:
            collector.close()
    finally:
        client.close()
    assert time.monotonic() - started < 7


def test_live_span_limit_applies_across_exports_and_keeps_retry_and_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sdk, "MAX_SPANS", 2)
    collector = SdkTrace(tmp_path / "sdk-trace")
    address = urlsplit(collector.endpoint)

    def post(raw):
        client = http.client.HTTPConnection(address.hostname, address.port, timeout=5)
        try:
            client.request(
                "POST", address.path, raw, {"Content-Type": "application/x-protobuf"}
            )
            response = client.getresponse()
            response.read()
            return response.status
        finally:
            client.close()

    first = export()
    try:
        assert post(first) == 200
        assert post(first) == 200
        assert (
            post(export({"gen_ai.response.id": "response-2"}, span_id=b"\x03" * 8))
            == 200
        )
        assert len(collector.span_identities) == 2
        assert (
            post(export({"gen_ai.response.id": "response-3"}, span_id=b"\x04" * 8))
            == 400
        )
        assert len(collector.span_identities) == 2
        assert collector.received_ids == {"response-1", "response-2"}
    finally:
        with pytest.raises(ValueError, match="span count"):
            collector.close()
    receipt = json.loads((collector.directory / "closed.json").read_text())
    assert receipt["exports"] == 4 and receipt["unique_sdk_spans"] == 2
    assert (collector.directory / "0000.pb").read_bytes() == (
        collector.directory / "0001.pb"
    ).read_bytes()
    assert (collector.directory / "0003.pb").exists()
    with pytest.raises(ValueError, match="close cleanly"):
        read_sdk_spans(collector.directory)
