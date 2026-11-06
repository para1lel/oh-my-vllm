# Responses history and stream

## Sub-features

- Creation with `POST /v1/responses`.
- Retrieval with `GET /v1/responses/{id}`.
- Text continuation with `previous_response_id`.
- Deletion with `DELETE /v1/responses/{id}` and a subsequent 404 response.
- Typed SSE with `store=false`.

See [the Responses service](../../../../crates/zmq-worker/src/serving/mod.rs) and [stream events](../../../../crates/zmq-worker/src/serving/events.rs).

## How to get to it (user POV)

After [Launch](../SKILL.md#launch), send the public model ID and `input` to the creation route.
Save the returned response ID.
Use it for retrieval, deletion, or the next request's `previous_response_id`.
Use `stream=true` for typed SSE events.

## Driving it with verify.py

```bash
scripts/with-env.sh python .claude/skills/verify-oh-my-vllm/verify.py run --feature responses --binary target/debug/oh-my-vllm-zmq-worker
```

The [Responses drive](../drive.py) creates a parent, retrieves it, and creates a child continuation.
It deletes the parent and records its 404 retrieval.
It retrieves the child after parent deletion, then deletes the child.
It also captures a different stream with storage disabled.
The stream must end with `response.completed` and continuous sequence numbers.
Read the `http/` request/response pairs and `features/responses.json` in the evidence directory.

## Gotchas

Storage is enabled by default.
The service applies storage lifetime, count, and byte limits.
Stored IDs belong to that service instance. Use a new parent after a new launch.

The CPU fixture creates synthetic prompt tokens for history.
This does not check tokenizer history or checkpoint generation.
Use [the checkpoint path](../SKILL.md#checkpoint-service-path) for those checks.
