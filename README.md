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

## AgentX trace performance

This [AgentX measurement](docs/agentx.md) uses MTP4, four session trees, and 900 seconds of profiling on one B200.
Each point shows one completed request. Color shows its measured prefix cache hit ratio.
Server processing time contains preparation and later scheduling waits. It excludes the preceding admission queue and complete network transport.

![AgentX MTP4 request output rate and prefix cache hit ratio](docs/assets/agentx-mtp4.png)

Use the [request CSV](docs/assets/agentx-mtp4.csv) for the plotted values and client timing.
The [measurement procedure](docs/agentx.md#request-data) defines the fields and exclusions.

## End-to-end HTTP performance

Performance measurements use 24 configurations, nominal HGX B200 rates, and the reference calculation in the [HTTP procedure](docs/performance.md).
The complete timer covers submission, service preparation, inference, sampling, transport, and response parsing.
Requests use serving sampling defaults, auto tool constraints, and client ordinal seeds.
Requests have uncached prefixes, a 32768-token budget, and at most 4 running requests.

The benchmark uses `http-reference-v4` with the [documented assumptions](docs/performance.md#memory).
All 24 configurations have measurements for source `3800d3d7aec04529`.
The table contains one complete measurement run.
The run uses 1024 FA blocks, 128 GDN slots, and workload `serving-tools-v1`.
The prompt seed is `d32f3d5d2c4b4174a1bc2e7c155f0c03`.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.846 | 0.697 | 37.76% |
| Ordinary | 1 | 32768 | 8192 | 63.825 | 32.001 | 50.14% |
| Ordinary | 2 | 32768 | 64 | 3.207 | 1.178 | 36.72% |
| Ordinary | 2 | 32768 | 8192 | 74.546 | 35.359 | 47.43% |
| Ordinary | 4 | 32768 | 64 | 5.823 | 2.131 | 36.59% |
| Ordinary | 4 | 32768 | 8192 | 82.110 | 42.067 | 51.23% |
| MTP4 | 1 | 32768 | 64 | 1.594 | 0.572 | 35.90% |
| MTP4 | 1 | 32768 | 8192 | 20.959 | 10.384 | 49.55% |
| MTP4 | 1 | 65536 | 64 | 4.689 | 1.230 | 26.23% |
| MTP4 | 1 | 65536 | 8192 | 23.032 | 11.378 | 49.40% |
| MTP4 | 1 | 131072 | 64 | 8.119 | 3.107 | 38.27% |
| MTP4 | 1 | 131072 | 8192 | 32.112 | 14.558 | 45.34% |
| MTP4 | 2 | 32768 | 64 | 3.021 | 1.037 | 34.32% |
| MTP4 | 2 | 32768 | 8192 | 28.640 | 12.299 | 42.94% |
| MTP4 | 2 | 65536 | 64 | 13.640 | 2.362 | 17.31% |
| MTP4 | 2 | 65536 | 8192 | 34.524 | 15.550 | 45.04% |
| MTP4 | 2 | 131072 | 64 | 15.167 | 6.076 | 40.06% |
| MTP4 | 2 | 131072 | 8192 | 47.076 | 22.158 | 47.07% |
| MTP4 | 4 | 32768 | 64 | 10.319 | 1.971 | 19.10% |
| MTP4 | 4 | 32768 | 8192 | 39.925 | 16.159 | 40.47% |
| MTP4 | 4 | 65536 | 64 | 12.357 | 4.601 | 37.23% |
| MTP4 | 4 | 65536 | 8192 | 48.689 | 20.106 | 41.29% |
| MTP4 | 4 | 131072 | 64 | 30.308 | 12.075 | 39.84% |
| MTP4 | 4 | 131072 | 8192 | 75.209 | 34.489 | 45.86% |
