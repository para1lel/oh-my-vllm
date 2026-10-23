# oh-my-vllm

This program supplies a Rust service for Qwen3.8-27B-FP8 on one NVIDIA B200.
The Python worker loads the model and runs text, image, and video inference.
Use ordinary decode or MTP4, with constrained decoding.

Use the [service guide](docs/serving.md) for installation, commands, and HTTP requests.
The service supplies Chat and Responses APIs, streaming, tools, and structured output.
The [host KV cache](docs/kv-offload.md) supports request offload and reusable prefixes with reserved pinned memory.

[中文](README.zh.md) | [Contribution procedure](CONTRIBUTING.md)

## Execution and checks

The worker uses the [GPU execution rules](docs/execution.md) for operator scheduling and resource ownership.
The [full-model numerical checks](docs/operators/evaluation.md) compare fixed endpoints against direct PyTorch FP64 computation.
The [operator checks](docs/operators/microbench.md) use repository fixtures and fixed theoretical-latency ratio limits.

## Measurements and planned extension

The [Qwen results](docs/deepseek-v41/qwen-results.md) contain the last completed HTTP matrix and its measurement identities.
Use the [HTTP procedure](docs/performance.md) and [AgentX procedure](docs/agentx.md) for measurement commands and definitions.

The [DeepSeek architecture](docs/deepseek-v41/design.md) gives the planned four-GPU extension on `nightly`.
DeepSeek inference is not enabled.
