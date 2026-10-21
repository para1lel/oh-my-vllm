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

The benchmark uses `http-reference-v4` with the [documented assumptions](docs/performance.md#memory).
All 24 configurations have measurements for source `773da8d74be29baf`.
Each long-output configuration has two efficiency measurements of more than 40%.
Some short-output configurations with measured times of 5 seconds or more have an efficiency of less than 40%.
The run uses 1024 FA blocks, 128 GDN slots, and prompt seed `serving-tools-v1`.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.867 | 0.697 | 37.35% |
| Ordinary | 1 | 32768 | 8192 | 63.907 | 32.001 | 50.07% |
| Ordinary | 2 | 32768 | 64 | 3.240 | 1.178 | 36.34% |
| Ordinary | 2 | 32768 | 8192 | 74.880 | 35.359 | 47.22% |
| Ordinary | 4 | 32768 | 64 | 5.871 | 2.131 | 36.29% |
| Ordinary | 4 | 32768 | 8192 | 82.448 | 42.067 | 51.02% |
| MTP4 | 1 | 32768 | 64 | 1.628 | 0.583 | 35.79% |
| MTP4 | 1 | 32768 | 8192 | 26.704 | 13.263 | 49.67% |
| MTP4 | 1 | 65536 | 64 | 5.766 | 1.224 | 21.23% |
| MTP4 | 1 | 65536 | 8192 | 26.893 | 13.346 | 49.63% |
| MTP4 | 1 | 131072 | 64 | 7.792 | 3.082 | 39.55% |
| MTP4 | 1 | 131072 | 8192 | 30.711 | 15.213 | 49.54% |
| MTP4 | 2 | 32768 | 64 | 3.000 | 1.053 | 35.10% |
| MTP4 | 2 | 32768 | 8192 | 30.444 | 13.361 | 43.89% |
| MTP4 | 2 | 65536 | 64 | 6.395 | 2.343 | 36.64% |
| MTP4 | 2 | 65536 | 8192 | 36.339 | 14.971 | 41.20% |
| MTP4 | 2 | 131072 | 64 | 15.248 | 6.087 | 39.92% |
| MTP4 | 2 | 131072 | 8192 | 43.270 | 20.139 | 46.54% |
| MTP4 | 4 | 32768 | 64 | 5.717 | 1.988 | 34.78% |
| MTP4 | 4 | 32768 | 8192 | 37.385 | 15.435 | 41.29% |
| MTP4 | 4 | 65536 | 64 | 12.445 | 4.597 | 36.94% |
| MTP4 | 4 | 65536 | 8192 | 48.077 | 21.306 | 44.32% |
| MTP4 | 4 | 131072 | 64 | 30.112 | 12.049 | 40.01% |
| MTP4 | 4 | 131072 | 8192 | 69.055 | 31.634 | 45.81% |
