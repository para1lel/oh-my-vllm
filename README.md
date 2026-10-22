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
The benchmark drain limit is 120 seconds.
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
All 24 configurations have measurements for source `7f8d6872c91a17c0`.
The table contains one complete measurement run.

The run uses fixed FA capacities: 1824 blocks for ordinary decode and 1608 for MTP4.
The run uses 128 GDN slots and workload `serving-tools-v1`.
Use the [service memory settings](docs/serving.md#cache-memory) to override the defaults.
The prompt seed is `d32f3d5d2c4b4174a1bc2e7c155f0c03`.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.850 | 0.697 | 37.69% |
| Ordinary | 1 | 32768 | 8192 | 64.086 | 32.001 | 49.93% |
| Ordinary | 2 | 32768 | 64 | 3.214 | 1.178 | 36.64% |
| Ordinary | 2 | 32768 | 8192 | 75.359 | 35.359 | 46.92% |
| Ordinary | 4 | 32768 | 64 | 5.820 | 2.131 | 36.61% |
| Ordinary | 4 | 32768 | 8192 | 85.257 | 42.067 | 49.34% |
| MTP4 | 1 | 32768 | 64 | 1.579 | 0.572 | 36.25% |
| MTP4 | 1 | 32768 | 8192 | 21.120 | 10.384 | 49.17% |
| MTP4 | 1 | 65536 | 64 | 3.286 | 1.230 | 37.43% |
| MTP4 | 1 | 65536 | 8192 | 23.193 | 11.378 | 49.06% |
| MTP4 | 1 | 131072 | 64 | 7.728 | 3.107 | 40.21% |
| MTP4 | 1 | 131072 | 8192 | 29.468 | 14.558 | 49.40% |
| MTP4 | 2 | 32768 | 64 | 2.919 | 1.037 | 35.52% |
| MTP4 | 2 | 32768 | 8192 | 29.847 | 12.299 | 41.21% |
| MTP4 | 2 | 65536 | 64 | 6.330 | 2.362 | 37.30% |
| MTP4 | 2 | 65536 | 8192 | 34.662 | 15.550 | 44.86% |
| MTP4 | 2 | 131072 | 64 | 15.370 | 6.062 | 39.44% |
| MTP4 | 2 | 131072 | 8192 | 46.579 | 21.853 | 46.92% |
| MTP4 | 4 | 32768 | 64 | 8.759 | 1.972 | 22.52% |
| MTP4 | 4 | 32768 | 8192 | 36.662 | 13.782 | 37.59% |
| MTP4 | 4 | 65536 | 64 | 14.419 | 4.583 | 31.78% |
| MTP4 | 4 | 65536 | 8192 | 63.690 | 24.615 | 38.65% |
| MTP4 | 4 | 131072 | 64 | 32.236 | 12.075 | 37.46% |
| MTP4 | 4 | 131072 | 8192 | 83.305 | 34.128 | 40.97% |
