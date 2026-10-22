# Shared worker mechanisms

Keep ZMQ and msgpack transport in the bridge.
Use the placeholder only when the model path is `stub`.
A model load or computation failure must return an error.

Use runtime packages and native CUDA modules. Do not import development tools.
Rust controls logical cache identities. Python stores tensors and private states.
Keep constraint state consistent with accepted tokens after MTP rejection.

Keep model computation and token syntax in the selected model adapter.
Use the [adapter guide](../../../docs/deepseek-v41/adapters.md) for ownership and compatibility paths.

Use the [execution rules](../../../docs/execution.md) for Eager and CUDA Graph selection.
Select execution from the complete Execute message before batch division.

Before you add a test, tell the maintainer why it is necessary.
