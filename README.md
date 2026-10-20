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

The benchmark uses `http-reference-v2` with the [documented assumptions](docs/performance.md#memory).
All 24 configurations have measurements for source `6b374aee04f3b5d3`.
The run uses 1024 FA blocks, 128 GDN slots, and prompt seed `serving-tools-v1`.
`full` is the same complete run of all 24 configurations.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency | Run |
|---|---:|---:|---:|---:|---:|---:|---|
| Ordinary | 1 | 32768 | 64 | 1.966 | 0.454 | 23.11% | `full` |
| Ordinary | 1 | 32768 | 8192 | 66.299 | 31.549 | 47.59% | `full` |
| Ordinary | 2 | 32768 | 64 | 3.507 | 0.908 | 25.90% | `full` |
| Ordinary | 2 | 32768 | 8192 | 79.870 | 34.458 | 43.14% | `full` |
| Ordinary | 4 | 32768 | 64 | 6.409 | 1.817 | 28.35% | `full` |
| Ordinary | 4 | 32768 | 8192 | 89.594 | 40.264 | 44.94% | `full` |
| MTP4 | 1 | 32768 | 64 | 1.671 | 0.457 | 27.32% | `full` |
| MTP4 | 1 | 32768 | 8192 | 20.274 | 9.365 | 46.19% | `full` |
| MTP4 | 1 | 65536 | 64 | 3.560 | 1.099 | 30.88% | `full` |
| MTP4 | 1 | 65536 | 8192 | 30.700 | 13.794 | 44.93% | `full` |
| MTP4 | 1 | 131072 | 64 | 9.206 | 2.948 | 32.03% | `full` |
| MTP4 | 1 | 131072 | 8192 | 30.848 | 11.616 | 37.65% | `full` |
| MTP4 | 2 | 32768 | 64 | 3.536 | 0.913 | 25.81% | `full` |
| MTP4 | 2 | 32768 | 8192 | 29.223 | 11.585 | 39.64% | `full` |
| MTP4 | 2 | 65536 | 64 | 8.021 | 2.200 | 27.42% | `full` |
| MTP4 | 2 | 65536 | 8192 | 37.455 | 13.442 | 35.89% | `full` |
| MTP4 | 2 | 131072 | 64 | 17.747 | 5.897 | 33.23% | `full` |
| MTP4 | 2 | 131072 | 8192 | 48.699 | 15.556 | 31.94% | `full` |
| MTP4 | 4 | 32768 | 64 | 6.188 | 1.826 | 29.51% | `full` |
| MTP4 | 4 | 32768 | 8192 | 40.991 | 14.748 | 35.98% | `full` |
| MTP4 | 4 | 65536 | 64 | 16.239 | 4.398 | 27.08% | `full` |
| MTP4 | 4 | 65536 | 8192 | 49.195 | 15.656 | 31.82% | `full` |
| MTP4 | 4 | 131072 | 64 | 34.343 | 11.795 | 34.34% | `full` |
| MTP4 | 4 | 131072 | 8192 | 73.775 | 20.316 | 27.54% | `full` |
