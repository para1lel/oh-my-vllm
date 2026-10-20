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
All 24 configurations have measurements for source `aed771ff2dd58536`.
The run uses 1024 FA blocks, 128 GDN slots, and prompt seed `serving-tools-v1`.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.865 | 0.454 | 24.36% |
| Ordinary | 1 | 32768 | 8192 | 63.434 | 31.549 | 49.74% |
| Ordinary | 2 | 32768 | 64 | 3.242 | 0.908 | 28.02% |
| Ordinary | 2 | 32768 | 8192 | 75.349 | 34.458 | 45.73% |
| Ordinary | 4 | 32768 | 64 | 5.885 | 1.817 | 30.87% |
| Ordinary | 4 | 32768 | 8192 | 82.899 | 40.264 | 48.57% |
| MTP4 | 1 | 32768 | 64 | 2.241 | 0.457 | 20.37% |
| MTP4 | 1 | 32768 | 8192 | 24.318 | 11.463 | 47.14% |
| MTP4 | 1 | 65536 | 64 | 4.085 | 1.100 | 26.92% |
| MTP4 | 1 | 65536 | 8192 | 24.685 | 10.155 | 41.14% |
| MTP4 | 1 | 131072 | 64 | 7.875 | 2.948 | 37.44% |
| MTP4 | 1 | 131072 | 8192 | 29.106 | 11.205 | 38.50% |
| MTP4 | 2 | 32768 | 64 | 3.006 | 0.913 | 30.36% |
| MTP4 | 2 | 32768 | 8192 | 32.381 | 13.010 | 40.18% |
| MTP4 | 2 | 65536 | 64 | 7.078 | 2.199 | 31.07% |
| MTP4 | 2 | 65536 | 8192 | 32.897 | 12.144 | 36.91% |
| MTP4 | 2 | 131072 | 64 | 15.288 | 5.898 | 38.58% |
| MTP4 | 2 | 131072 | 8192 | 44.852 | 14.648 | 32.66% |
| MTP4 | 4 | 32768 | 64 | 5.860 | 1.826 | 31.17% |
| MTP4 | 4 | 32768 | 8192 | 46.009 | 16.539 | 35.95% |
| MTP4 | 4 | 65536 | 64 | 12.440 | 4.398 | 35.35% |
| MTP4 | 4 | 65536 | 8192 | 50.618 | 17.344 | 34.26% |
| MTP4 | 4 | 131072 | 64 | 30.506 | 11.795 | 38.66% |
| MTP4 | 4 | 131072 | 8192 | 74.304 | 22.758 | 30.63% |
