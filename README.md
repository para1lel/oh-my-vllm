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
All 24 configurations have measurements for source `c764aecd711d2fff`.
The table contains one complete measurement run.
The run uses 1024 FA blocks, 128 GDN slots, and prompt seed `serving-tools-v1`.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.816 | 0.697 | 38.39% |
| Ordinary | 1 | 32768 | 8192 | 63.603 | 32.001 | 50.31% |
| Ordinary | 2 | 32768 | 64 | 3.185 | 1.178 | 36.97% |
| Ordinary | 2 | 32768 | 8192 | 74.569 | 35.359 | 47.42% |
| Ordinary | 4 | 32768 | 64 | 5.797 | 2.131 | 36.75% |
| Ordinary | 4 | 32768 | 8192 | 82.359 | 42.067 | 51.08% |
| MTP4 | 1 | 32768 | 64 | 1.601 | 0.583 | 36.39% |
| MTP4 | 1 | 32768 | 8192 | 26.670 | 13.263 | 49.73% |
| MTP4 | 1 | 65536 | 64 | 3.291 | 1.224 | 37.21% |
| MTP4 | 1 | 65536 | 8192 | 26.849 | 13.346 | 49.71% |
| MTP4 | 1 | 131072 | 64 | 7.710 | 3.082 | 39.97% |
| MTP4 | 1 | 131072 | 8192 | 32.186 | 15.213 | 47.27% |
| MTP4 | 2 | 32768 | 64 | 2.962 | 1.053 | 35.55% |
| MTP4 | 2 | 32768 | 8192 | 31.249 | 13.361 | 42.76% |
| MTP4 | 2 | 65536 | 64 | 6.548 | 2.343 | 35.79% |
| MTP4 | 2 | 65536 | 8192 | 32.912 | 14.971 | 45.49% |
| MTP4 | 2 | 131072 | 64 | 17.088 | 6.087 | 35.62% |
| MTP4 | 2 | 131072 | 8192 | 43.192 | 20.139 | 46.63% |
| MTP4 | 4 | 32768 | 64 | 5.628 | 1.988 | 35.33% |
| MTP4 | 4 | 32768 | 8192 | 40.712 | 15.435 | 37.91% |
| MTP4 | 4 | 65536 | 64 | 12.273 | 4.597 | 37.45% |
| MTP4 | 4 | 65536 | 8192 | 50.119 | 21.306 | 42.51% |
| MTP4 | 4 | 131072 | 64 | 30.370 | 12.049 | 39.67% |
| MTP4 | 4 | 131072 | 8192 | 67.852 | 31.634 | 46.62% |
