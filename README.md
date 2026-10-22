# oh-my-vllm

This program supplies a Rust service for Qwen3.8-27B-FP8 on one NVIDIA B200.
The Python worker loads the model and runs text, image, and video inference.
Use ordinary decode or MTP4, with constrained decoding.

Use the [service guide](docs/serving.md) for installation, commands, and HTTP requests.
The [host KV cache](docs/kv-offload.md) supports request offload and reusable prefixes with dynamic pinned memory.

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
All 24 configurations have measurements for source `0adf95f4df308dd3`.
The table contains one complete measurement run.

Runtime sizing selects 1225 FA blocks for ordinary decode and 1016 for MTP4.
The run uses 128 GDN slots and workload `serving-tools-v1`.
FA sizing uses the 85% [service memory budget](docs/serving.md#cache-memory).
The prompt seed is `d32f3d5d2c4b4174a1bc2e7c155f0c03`.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.850 | 0.697 | 37.68% |
| Ordinary | 1 | 32768 | 8192 | 63.968 | 32.001 | 50.03% |
| Ordinary | 2 | 32768 | 64 | 3.219 | 1.178 | 36.58% |
| Ordinary | 2 | 32768 | 8192 | 75.111 | 35.359 | 47.08% |
| Ordinary | 4 | 32768 | 64 | 5.826 | 2.131 | 36.57% |
| Ordinary | 4 | 32768 | 8192 | 84.956 | 42.067 | 49.52% |
| MTP4 | 1 | 32768 | 64 | 1.588 | 0.572 | 36.04% |
| MTP4 | 1 | 32768 | 8192 | 21.181 | 10.384 | 49.03% |
| MTP4 | 1 | 65536 | 64 | 3.298 | 1.230 | 37.29% |
| MTP4 | 1 | 65536 | 8192 | 23.191 | 11.378 | 49.06% |
| MTP4 | 1 | 131072 | 64 | 7.774 | 3.107 | 39.97% |
| MTP4 | 1 | 131072 | 8192 | 29.503 | 14.558 | 49.35% |
| MTP4 | 2 | 32768 | 64 | 2.921 | 1.037 | 35.50% |
| MTP4 | 2 | 32768 | 8192 | 28.551 | 12.299 | 43.08% |
| MTP4 | 2 | 65536 | 64 | 6.468 | 2.362 | 36.51% |
| MTP4 | 2 | 65536 | 8192 | 35.897 | 14.086 | 39.24% |
| MTP4 | 2 | 131072 | 64 | 16.697 | 6.076 | 36.39% |
| MTP4 | 2 | 131072 | 8192 | 50.140 | 22.158 | 44.19% |
| MTP4 | 4 | 32768 | 64 | 8.481 | 1.972 | 23.26% |
| MTP4 | 4 | 32768 | 8192 | 41.356 | 15.841 | 38.30% |
| MTP4 | 4 | 65536 | 64 | 13.996 | 4.594 | 32.83% |
| MTP4 | 4 | 65536 | 8192 | 57.194 | 21.393 | 37.40% |
| MTP4 | 4 | 131072 | 64 | 33.439 | 12.071 | 36.10% |
| MTP4 | 4 | 131072 | 8192 | 79.504 | 33.644 | 42.32% |
