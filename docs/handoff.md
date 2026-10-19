# Work state

Use `main` for work.
Use the [GPU execution rules](execution.md).
The worker uses traditional operator scheduling with the native kernels from the measured historical implementation.
An Execute that contains prefill uses Eager execution for all fragments.
A decode-only Execute records and replays CUDA Graphs.
The tile scheduler, task plans, and protocol fixtures have been removed.

Rust supplies HTTP, scheduling, logical KV, and worker lifecycle controls.
Python supplies model execution through ZMQ DEALER and msgpack.
Graph replay uses the caller's stream.
Graph closure waits for GPU work before it releases resources.
Failed recording releases its temporary resource leases.

The builder is `development.native_build`.
The [numerical tools](operators/evaluation.md) compare short-input, single-iteration full-model logits with PyTorch FP64 computation.
Keep the frozen reference and its identity checks.
Single-operator tolerance tests and their reference providers have been removed.
Old operator timing, public comparison, and peak calibration tools have also been removed.

Use the 24 [HTTP configurations](performance.md) and nominal-resource accounting for performance work.
Use different artifacts for historical measurements and evidence for this source.
Optimize the native operators only when full HTTP measurements support the change.
Do not change numerical precision, state ownership, or service behavior.
Extend full-model inputs when changes affect MTP, vision, or long contexts.

Keep host paths and artifact locations in ignored local configuration.
