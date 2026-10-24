# oh-my-vllm

This inference framework uses Rust for service, scheduling, and logical state.
Python model adapters run native CUDA operators through ZMQ DEALER and msgpack.
Each adapter owns its model configuration, computation, cache layout, and token syntax.

Use the [service guide](docs/serving.md) for installation, commands, and HTTP requests.
The service supplies Chat and Responses APIs, streaming, tools, and structured output.

[中文](README.zh.md) | [Contribution procedure](CONTRIBUTING.md)

## Model adapters

| Adapter | Execution | Status and documents |
|---|---|---|
| Qwen3.8-27B-FP8 | One B200, ordinary decode and MTP4, 262144 total tokens | [Implemented adapter](docs/models/qwen38/README.md), [latest HTTP results](docs/models/qwen38/results.md) |
| DeepSeek-V4.1-Flash | Four B200 GPUs, TP4/TEP4/DEP4, DSpark5, 1048576 total tokens | [Adapter design](docs/models/deepseek_v41/README.md), [implementation plan](docs/models/deepseek_v41/plan.md). Service inference is not enabled |

The adapters have the same architectural role. Their implementation and acceptance status can differ.
Unknown or unimplemented models cause startup errors.

## Development

Use the [adapter contract](docs/adapters.md), [GPU execution rules](docs/execution.md), and [test scope](docs/testing.md).
Model directories contain their mathematics, fixed numerical limits, measurement procedures, and latest complete results.
The [handoff](docs/handoff.md) gives current work and next actions.
