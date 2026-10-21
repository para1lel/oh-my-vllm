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
All 24 configurations have measurements for source `89383e6f97116234`.
The table contains one complete measurement run.
The run uses 1024 FA blocks, 128 GDN slots, and prompt seed `serving-tools-v1`.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.834 | 0.697 | 38.02% |
| Ordinary | 1 | 32768 | 8192 | 64.009 | 32.001 | 49.99% |
| Ordinary | 2 | 32768 | 64 | 3.218 | 1.178 | 36.59% |
| Ordinary | 2 | 32768 | 8192 | 74.757 | 35.359 | 47.30% |
| Ordinary | 4 | 32768 | 64 | 5.980 | 2.131 | 35.63% |
| Ordinary | 4 | 32768 | 8192 | 82.491 | 42.067 | 51.00% |
| MTP4 | 1 | 32768 | 64 | 1.608 | 0.583 | 36.24% |
| MTP4 | 1 | 32768 | 8192 | 26.705 | 13.263 | 49.66% |
| MTP4 | 1 | 65536 | 64 | 3.357 | 1.224 | 36.47% |
| MTP4 | 1 | 65536 | 8192 | 26.884 | 13.346 | 49.64% |
| MTP4 | 1 | 131072 | 64 | 10.642 | 3.082 | 28.96% |
| MTP4 | 1 | 131072 | 8192 | 30.751 | 15.213 | 49.47% |
| MTP4 | 2 | 32768 | 64 | 3.036 | 1.053 | 34.68% |
| MTP4 | 2 | 32768 | 8192 | 30.451 | 13.361 | 43.88% |
| MTP4 | 2 | 65536 | 64 | 6.390 | 2.343 | 36.67% |
| MTP4 | 2 | 65536 | 8192 | 33.165 | 14.971 | 45.14% |
| MTP4 | 2 | 131072 | 64 | 15.269 | 6.087 | 39.86% |
| MTP4 | 2 | 131072 | 8192 | 43.246 | 20.139 | 46.57% |
| MTP4 | 4 | 32768 | 64 | 5.693 | 1.988 | 34.93% |
| MTP4 | 4 | 32768 | 8192 | 41.547 | 15.435 | 37.15% |
| MTP4 | 4 | 65536 | 64 | 12.484 | 4.597 | 36.82% |
| MTP4 | 4 | 65536 | 8192 | 50.050 | 21.306 | 42.57% |
| MTP4 | 4 | 131072 | 64 | 30.013 | 12.049 | 40.14% |
| MTP4 | 4 | 131072 | 8192 | 68.887 | 31.634 | 45.92% |
