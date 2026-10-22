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

This [AgentX measurement](docs/agentx.md) uses MTP4, six session trees, and 900 seconds of profiling on one B200.
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
All 24 configurations have measurements for source `a4caca0e0e2f7728`.
The table contains one complete measurement run.

Runtime sizing selects 1225 FA blocks for ordinary decode and 1016 for MTP4.
The run uses 128 GDN slots and workload `serving-tools-v1`.
FA sizing uses the 85% [service memory budget](docs/serving.md#cache-memory).
The prompt seed is `d32f3d5d2c4b4174a1bc2e7c155f0c03`.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.847 | 0.697 | 37.75% |
| Ordinary | 1 | 32768 | 8192 | 63.771 | 32.001 | 50.18% |
| Ordinary | 2 | 32768 | 64 | 3.321 | 1.178 | 35.46% |
| Ordinary | 2 | 32768 | 8192 | 75.280 | 35.359 | 46.97% |
| Ordinary | 4 | 32768 | 64 | 5.813 | 2.131 | 36.65% |
| Ordinary | 4 | 32768 | 8192 | 82.542 | 42.067 | 50.96% |
| MTP4 | 1 | 32768 | 64 | 1.580 | 0.572 | 36.20% |
| MTP4 | 1 | 32768 | 8192 | 20.866 | 10.384 | 49.77% |
| MTP4 | 1 | 65536 | 64 | 3.311 | 1.230 | 37.15% |
| MTP4 | 1 | 65536 | 8192 | 23.122 | 11.378 | 49.21% |
| MTP4 | 1 | 131072 | 64 | 7.756 | 3.107 | 40.06% |
| MTP4 | 1 | 131072 | 8192 | 29.443 | 14.558 | 49.45% |
| MTP4 | 2 | 32768 | 64 | 2.918 | 1.037 | 35.53% |
| MTP4 | 2 | 32768 | 8192 | 28.455 | 12.299 | 43.22% |
| MTP4 | 2 | 65536 | 64 | 6.383 | 2.362 | 37.00% |
| MTP4 | 2 | 65536 | 8192 | 34.645 | 15.550 | 44.88% |
| MTP4 | 2 | 131072 | 64 | 15.145 | 6.076 | 40.12% |
| MTP4 | 2 | 131072 | 8192 | 46.791 | 22.158 | 47.35% |
| MTP4 | 4 | 32768 | 64 | 7.541 | 1.971 | 26.14% |
| MTP4 | 4 | 32768 | 8192 | 39.418 | 16.159 | 40.99% |
| MTP4 | 4 | 65536 | 64 | 13.087 | 4.601 | 35.15% |
| MTP4 | 4 | 65536 | 8192 | 47.610 | 20.106 | 42.23% |
| MTP4 | 4 | 131072 | 64 | 29.945 | 12.075 | 40.32% |
| MTP4 | 4 | 131072 | 8192 | 74.403 | 34.489 | 46.35% |
