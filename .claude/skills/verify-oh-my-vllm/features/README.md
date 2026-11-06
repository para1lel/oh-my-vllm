# Verification feature map

## Sub-features

| Topic | User behavior |
|---|---|
| [Chat](chat.md) | JSON replies, SSE chunks, completion marker, usage |
| [Responses](responses.md) | Create, retrieve, continue, delete, typed SSE |
| [Tools and output format](tools.md) | Chat tools, Responses tools, JSON Schema output |
| [Operations](operations.md) | Diagnostics, validation errors, offline run/bench, process cleanup |

The [Rust CLI entry](../../../../crates/zmq-worker/src/main.rs) and [HTTP entry](../../../../crates/zmq-worker/src/serving/mod.rs) supply the app.
The [existing service cases](../../../../tests/test_service.py) define the CPU boundary examples.

## How to get to it (user POV)

Start with [Launch](../SKILL.md#launch).
The app has CLI commands and HTTP routes. It has no browser interface.
Use `GET /v1/models` to get the public model ID before API calls.
Use the selected topic for its route and request controls.

## Driving it with verify.py

Run this command from the repository root after environment setup:

```bash
scripts/with-env.sh python .claude/skills/verify-oh-my-vllm/verify.py run --feature all --binary target/debug/oh-my-vllm-zmq-worker
```

Select one topic with `--feature chat`, `--feature responses`, `--feature tools`, or `--feature operations`.
One invocation starts the app, checks readiness, drives the topic, saves evidence, and stops owned processes.
Read the [evidence contract](../SKILL.md#evidence) before reporting completion.

## Gotchas

The default [worker fixture](../../../../tests/worker_fixture.py) supplies fixed replies and synthetic model identities.
It checks service and transport behavior only.
It does not prove GPU inference, numerical correctness, capacity, or performance.
Use the [checkpoint path](../SKILL.md#checkpoint-service-path) for inference with model weights and native modules.
Keep raw evidence external to the repository.
Update this map and its Chinese translation when routes or CLI options change.
