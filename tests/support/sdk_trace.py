"""Original protobuf SDK exports and isolated provider bindings for evidence tests."""

import hashlib
import json

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)


def export(attributes=None, *, status_code=0, **changes):
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
    span.status.code = status_code
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


def saved_trace(tmp_path, raw, *, api="chat"):
    binding = "openai-completions" if api == "chat" else "openai-responses"
    provider = f"oh-my-vllm-{api}"
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
        json.dumps({"providers": {provider: {"api": binding}}})
    )
    (state / "settings.yml").write_text("{}")
    (tmp_path / "states.json").write_text(
        json.dumps(
            [
                {
                    "sessionId": "fixed-session",
                    "model": {
                        "id": "qwen3.8-27b-fp8",
                        "provider": provider,
                        "api": binding,
                    },
                }
            ]
        )
    )
    return {
        "api": api,
        "session_id": "fixed-session",
        "sdk_trace": {
            "directory": "sdk-trace",
            "source": "omp_sdk_otlp",
            "api_binding": binding,
            "api_binding_source": "isolated_provider_configuration",
            "configuration_sha256": {
                name: hashlib.sha256((state / name).read_bytes()).hexdigest()
                for name in ("models.yml", "settings.yml")
            },
        },
    }
