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

The DSV4.1 model and checkpoint facts are recorded in the [model contract research](research/dsv41-model-checkpoint-contract.md).
The report keeps CED, tokenizer, pad-token, DSpark, cache, Engram residency, and parallelism decisions open.
The [DEP4 load-balancing research](research/dep4-load-balancing-research.md) defines stable step ownership, multi-signal load estimates, replicated hot experts with a slow control loop, and quiescent map changes.
It is a design contract, not an implementation or a measured result.

## Next work

- Run the Qwen HTTP and AgentX regression matrices after changes to shared runtime code.
- Resolve the open DSV4.1 parallelism, physical cache, state ownership, DSpark, and acceptance decisions before implementation.
- Keep model-specific rules in `python/oh_my_vllm/models/qwen38/` and `crates/zmq-worker/src/models/qwen38/`.
- Keep raw reports, host settings, and temporary process records in external locations.

Use the [test scope](testing.md), [execution rules](execution.md), and [delivery procedure](../CONTRIBUTING.md).
