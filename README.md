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
Requests use greedy text generation, uncached prefixes, a 32768-token budget, and at most 4 running requests.

The benchmark uses `http-reference-v1` with the [documented assumptions](docs/performance.md#memory).
All 24 configurations have measurements for source `94e344e1fe67a394`.
The run uses 1024 FA blocks, 128 GDN slots, and prompt seed `nightly-dual-v1`.
For MTP4 with batch 4, 65536 input tokens, and 8192 output tokens, the table uses the faster measurement from a subsequent run.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary decode | 1 | 32768 | 64 | 1.564 | 0.454 | 29.05% |
| Ordinary decode | 1 | 32768 | 8192 | 64.242 | 31.388 | 48.86% |
| Ordinary decode | 2 | 32768 | 64 | 2.698 | 0.908 | 33.67% |
| Ordinary decode | 2 | 32768 | 8192 | 76.406 | 34.136 | 44.68% |
| Ordinary decode | 4 | 32768 | 64 | 4.873 | 1.817 | 37.29% |
| Ordinary decode | 4 | 32768 | 8192 | 80.557 | 39.620 | 49.18% |
| MTP4 | 1 | 32768 | 64 | 1.290 | 0.457 | 35.40% |
| MTP4 | 1 | 32768 | 8192 | 18.857 | 9.272 | 49.17% |
| MTP4 | 1 | 65536 | 64 | 2.657 | 1.100 | 41.39% |
| MTP4 | 1 | 65536 | 8192 | 22.745 | 10.755 | 47.29% |
| MTP4 | 1 | 131072 | 64 | 6.119 | 2.949 | 48.19% |
| MTP4 | 1 | 131072 | 8192 | 29.335 | 12.586 | 42.90% |
| MTP4 | 2 | 32768 | 64 | 2.475 | 0.914 | 36.92% |
| MTP4 | 2 | 32768 | 8192 | 23.980 | 10.093 | 42.09% |
| MTP4 | 2 | 65536 | 64 | 5.040 | 2.199 | 43.64% |
| MTP4 | 2 | 65536 | 8192 | 28.249 | 11.094 | 39.27% |
| MTP4 | 2 | 131072 | 64 | 11.909 | 5.897 | 49.52% |
| MTP4 | 2 | 131072 | 8192 | 41.174 | 14.632 | 35.54% |
| MTP4 | 4 | 32768 | 64 | 4.538 | 1.827 | 40.26% |
| MTP4 | 4 | 32768 | 8192 | 31.615 | 12.427 | 39.31% |
| MTP4 | 4 | 65536 | 64 | 9.766 | 4.398 | 45.04% |
| MTP4 | 4 | 65536 | 8192 | 40.408 | 14.611 | 36.16% |
| MTP4 | 4 | 131072 | 64 | 23.364 | 11.795 | 50.48% |
| MTP4 | 4 | 131072 | 8192 | 60.662 | 19.184 | 31.62% |
