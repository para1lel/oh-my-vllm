# Work state

Work continues on `nightly`. The repository keeps the Qwen service path and removes incomplete model work.

## Current service path

Qwen serves inference on one B200 through the Rust HTTP service, the Rust scheduler, and the Python worker.
The worker uses the ZMQ DEALER and msgpack boundary.
The service supports Chat and Responses APIs, streaming, tools, structured output, media, prefix cache, MTP4, and adaptive CUDA Graph selection.
The explicit `stub` backend is used only by service tests.

The scheduler selects decode work, complete prefill blocks, and residual tokens in that order.
The cache policy owns Qwen FA and GDN storage assignments.
Rust validates output, state, transfers, and worker lifecycle before it publishes a result.

## Next work

- Run the Qwen HTTP and AgentX regression matrices after changes to shared runtime code.
- Keep model-specific rules in `python/oh_my_vllm/models/qwen38/` and `crates/zmq-worker/src/models/qwen38/`.
- Keep raw reports, host settings, and temporary process records in external locations.

Use the [test scope](testing.md), [execution rules](execution.md), and [delivery procedure](../CONTRIBUTING.md).
