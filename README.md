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
Numerical checks use a short-input, single-iteration [full-model logits comparison](docs/operators/evaluation.md) against direct PyTorch FP64 computation.

## End-to-end HTTP performance

Performance measurements use 24 configurations, nominal HGX B200 rates, and the reference calculation in the [HTTP procedure](docs/performance.md).
The complete timer covers submission, service preparation, inference, sampling, transport, and response parsing.
Requests use serving sampling defaults, auto tool constraints, and client ordinal seeds.
Requests have uncached prefixes, a 32768-token budget, and at most 4 running requests.

The benchmark uses `http-reference-v1` with the [documented assumptions](docs/performance.md#memory).
All 24 configurations have measurements for source `90fdd2baa2d783b9`.
The run uses 1024 FA blocks, 128 GDN slots, and prompt seed `serving-tools-v1`.
The table uses a faster measurement from a separate run for MTP4 with batch 2 and 8192 output tokens.
The input length is 131072 tokens.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.531 | 0.454 | 29.68% |
| Ordinary | 1 | 32768 | 8192 | 65.587 | 31.388 | 47.86% |
| Ordinary | 2 | 32768 | 64 | 2.553 | 0.908 | 35.59% |
| Ordinary | 2 | 32768 | 8192 | 73.275 | 34.136 | 46.59% |
| Ordinary | 4 | 32768 | 64 | 6.379 | 1.817 | 28.48% |
| Ordinary | 4 | 32768 | 8192 | 76.711 | 39.620 | 51.65% |
| MTP4 | 1 | 32768 | 64 | 1.246 | 0.457 | 36.64% |
| MTP4 | 1 | 32768 | 8192 | 20.487 | 10.008 | 48.85% |
| MTP4 | 1 | 65536 | 64 | 2.530 | 1.100 | 43.47% |
| MTP4 | 1 | 65536 | 8192 | 20.985 | 10.000 | 47.65% |
| MTP4 | 1 | 131072 | 64 | 5.750 | 2.949 | 51.29% |
| MTP4 | 1 | 131072 | 8192 | 29.246 | 13.465 | 46.04% |
| MTP4 | 2 | 32768 | 64 | 2.260 | 0.913 | 40.38% |
| MTP4 | 2 | 32768 | 8192 | 24.177 | 10.335 | 42.74% |
| MTP4 | 2 | 65536 | 64 | 4.870 | 2.199 | 45.17% |
| MTP4 | 2 | 65536 | 8192 | 31.567 | 13.690 | 43.37% |
| MTP4 | 2 | 131072 | 64 | 11.112 | 5.898 | 53.07% |
| MTP4 | 2 | 131072 | 8192 | 36.919 | 14.851 | 40.23% |
| MTP4 | 4 | 32768 | 64 | 4.354 | 1.827 | 41.96% |
| MTP4 | 4 | 32768 | 8192 | 34.968 | 14.317 | 40.94% |
| MTP4 | 4 | 65536 | 64 | 9.246 | 4.399 | 47.57% |
| MTP4 | 4 | 65536 | 8192 | 42.521 | 17.559 | 41.29% |
| MTP4 | 4 | 131072 | 64 | 24.673 | 11.794 | 47.80% |
| MTP4 | 4 | 131072 | 8192 | 60.733 | 24.388 | 40.16% |
