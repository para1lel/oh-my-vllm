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
All 24 configurations have measurements for source `50db7eec3f23dd46`.
The run uses 1024 FA blocks, 128 GDN slots, and prompt seed `serving-tools-v1`.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.886 | 0.454 | 24.09% |
| Ordinary | 1 | 32768 | 8192 | 64.023 | 31.549 | 49.28% |
| Ordinary | 2 | 32768 | 64 | 3.250 | 0.908 | 27.96% |
| Ordinary | 2 | 32768 | 8192 | 75.668 | 34.458 | 45.54% |
| Ordinary | 4 | 32768 | 64 | 5.896 | 1.817 | 30.82% |
| Ordinary | 4 | 32768 | 8192 | 83.316 | 40.264 | 48.33% |
| MTP4 | 1 | 32768 | 64 | 1.617 | 0.457 | 28.23% |
| MTP4 | 1 | 32768 | 8192 | 24.182 | 11.463 | 47.40% |
| MTP4 | 1 | 65536 | 64 | 3.333 | 1.100 | 33.00% |
| MTP4 | 1 | 65536 | 8192 | 23.091 | 10.155 | 43.98% |
| MTP4 | 1 | 131072 | 64 | 7.774 | 2.948 | 37.93% |
| MTP4 | 1 | 131072 | 8192 | 28.870 | 11.205 | 38.81% |
| MTP4 | 2 | 32768 | 64 | 2.991 | 0.913 | 30.51% |
| MTP4 | 2 | 32768 | 8192 | 31.799 | 13.010 | 40.91% |
| MTP4 | 2 | 65536 | 64 | 10.260 | 2.199 | 21.43% |
| MTP4 | 2 | 65536 | 8192 | 32.512 | 12.144 | 37.35% |
| MTP4 | 2 | 131072 | 64 | 20.047 | 5.898 | 29.42% |
| MTP4 | 2 | 131072 | 8192 | 44.688 | 14.648 | 32.78% |
| MTP4 | 4 | 32768 | 64 | 5.717 | 1.826 | 31.95% |
| MTP4 | 4 | 32768 | 8192 | 46.203 | 16.539 | 35.80% |
| MTP4 | 4 | 65536 | 64 | 12.415 | 4.398 | 35.42% |
| MTP4 | 4 | 65536 | 8192 | 50.826 | 17.344 | 34.12% |
| MTP4 | 4 | 131072 | 64 | 30.211 | 11.795 | 39.04% |
| MTP4 | 4 | 131072 | 8192 | 74.168 | 22.758 | 30.68% |
