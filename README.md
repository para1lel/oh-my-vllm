# oh-my-vllm

This program supplies a Rust service for Qwen3.8-27B-FP8 on one NVIDIA B200.
The Python worker loads the model and runs text, image, and video inference.
Use ordinary decode or MTP4, with constrained decoding.

Use the [service guide](docs/serving.md) for installation, commands, and HTTP requests.

[中文](README.zh.md) · [Contribution procedure](CONTRIBUTING.md)

## GPU execution

The worker uses [operator scheduling](docs/execution.md).
An Execute that contains prefill uses Eager execution for all its fragments.
A decode-only Execute uses CUDA Graph recording and replay of the same native operators.
The [full-model numerical checks](docs/operators/evaluation.md) compare fixed short, MTP restoration, media, and long-context endpoints against direct PyTorch FP64 computation.
The [operator checks](docs/operators/microbench.md) use repository PyTorch fixtures and fixed theoretical-latency ratio limits.

## End-to-end HTTP performance

Performance measurements use 24 configurations, nominal HGX B200 rates, and the reference calculation in the [HTTP procedure](docs/performance.md).
The complete timer covers submission, service preparation, inference, sampling, transport, and response parsing.
Requests use serving sampling defaults, auto tool constraints, and client ordinal seeds.
Requests have uncached prefixes, a 32768-token budget, and at most 4 running requests.

The benchmark uses `http-reference-v3` with the [documented assumptions](docs/performance.md#memory).
All 24 configurations have measurements for source `2169359dfef1516b`.
Each long-output configuration has two efficiency measurements of more than 40%.
The run uses 1024 FA blocks, 128 GDN slots, and prompt seed `serving-tools-v1`.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.871 | 0.697 | 37.27% |
| Ordinary | 1 | 32768 | 8192 | 63.714 | 32.001 | 50.23% |
| Ordinary | 2 | 32768 | 64 | 3.320 | 1.178 | 35.47% |
| Ordinary | 2 | 32768 | 8192 | 74.758 | 35.359 | 47.30% |
| Ordinary | 4 | 32768 | 64 | 5.875 | 2.131 | 36.27% |
| Ordinary | 4 | 32768 | 8192 | 82.412 | 42.067 | 51.04% |
| MTP4 | 1 | 32768 | 64 | 1.627 | 0.583 | 35.83% |
| MTP4 | 1 | 32768 | 8192 | 26.607 | 13.263 | 49.85% |
| MTP4 | 1 | 65536 | 64 | 3.705 | 1.224 | 33.05% |
| MTP4 | 1 | 65536 | 8192 | 26.746 | 13.346 | 49.90% |
| MTP4 | 1 | 131072 | 64 | 7.786 | 3.082 | 39.58% |
| MTP4 | 1 | 131072 | 8192 | 30.650 | 15.213 | 49.64% |
| MTP4 | 2 | 32768 | 64 | 3.038 | 1.053 | 34.66% |
| MTP4 | 2 | 32768 | 8192 | 29.778 | 13.361 | 44.87% |
| MTP4 | 2 | 65536 | 64 | 6.391 | 2.343 | 36.67% |
| MTP4 | 2 | 65536 | 8192 | 34.489 | 14.971 | 43.41% |
| MTP4 | 2 | 131072 | 64 | 15.313 | 6.087 | 39.75% |
| MTP4 | 2 | 131072 | 8192 | 44.917 | 20.139 | 44.83% |
| MTP4 | 4 | 32768 | 64 | 5.707 | 1.988 | 34.84% |
| MTP4 | 4 | 32768 | 8192 | 38.023 | 15.435 | 40.59% |
| MTP4 | 4 | 65536 | 64 | 12.456 | 4.597 | 36.91% |
| MTP4 | 4 | 65536 | 8192 | 48.766 | 21.306 | 43.69% |
| MTP4 | 4 | 131072 | 64 | 30.069 | 12.049 | 40.07% |
| MTP4 | 4 | 131072 | 8192 | 71.087 | 31.634 | 44.50% |
