# Chat replies

## Sub-features

- A full JSON reply from `POST /v1/chat/completions`.
- SSE chunks with `stream=true` and a final `[DONE]` marker.
- Output token usage with `stream_options.include_usage=true`.

See [Chat request parsing](../../../../crates/zmq-worker/src/serving/request.rs) and [output events](../../../../crates/zmq-worker/src/serving/events.rs).

## How to get to it (user POV)

Read the model ID from `GET /v1/models` after [Launch](../SKILL.md#launch).
Send that ID in `model` with a `messages` array and `max_tokens`.
Use `stream=false` for JSON or `stream=true` for SSE.
The helper uses the same prompt and seed for these two requests.

## Driving it with verify.py

```bash
scripts/with-env.sh python .claude/skills/verify-oh-my-vllm/verify.py run --feature chat --binary target/debug/oh-my-vllm-zmq-worker
```

The [Chat drive](../drive.py) sends JSON and SSE requests through the new Rust service.
It compares reply text and output token counts.
It checks fixed fixture text, the completion marker, the finish reason, and output usage greater than zero.
The CPU fixture reply is `test answer`.
Read `http/*-chat-json/response.body`, `http/*-chat-sse/response.body`, and `features/chat.json` in the evidence directory.

## Gotchas

Repeated prompts can reuse the prefix cache.
Prompt cache counters can differ between JSON and SSE even when output token counts agree.
The helper saves each full usage object.
The fixed fixture reply checks parsing and transport. It does not check model generation or sampling.
For inference, use [the checkpoint path](../SKILL.md#checkpoint-service-path) and the selected model's acceptance procedure.
