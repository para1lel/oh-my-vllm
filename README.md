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
All 24 configurations have measurements for source `ba033cc3fafa0eb1`.
The two runs use 1024 FA blocks, 128 GDN slots, and prompt seed `serving-tools-v1`.
`full` is the run of all 24 configurations.
`repeat` is a different run of seven selected MTP4 configurations.
These seven rows use the measurements from that second run.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency | Run |
|---|---:|---:|---:|---:|---:|---:|---|
| Ordinary | 1 | 32768 | 64 | 1.528 | 0.454 | 29.73% | `full` |
| Ordinary | 1 | 32768 | 8192 | 65.295 | 31.388 | 48.07% | `full` |
| Ordinary | 2 | 32768 | 64 | 2.552 | 0.908 | 35.60% | `full` |
| Ordinary | 2 | 32768 | 8192 | 73.066 | 34.136 | 46.72% | `full` |
| Ordinary | 4 | 32768 | 64 | 4.483 | 1.817 | 40.53% | `full` |
| Ordinary | 4 | 32768 | 8192 | 76.542 | 39.620 | 51.76% | `full` |
| MTP4 | 1 | 32768 | 64 | 1.256 | 0.457 | 36.34% | `full` |
| MTP4 | 1 | 32768 | 8192 | 20.432 | 10.008 | 48.98% | `full` |
| MTP4 | 1 | 65536 | 64 | 2.533 | 1.100 | 43.42% | `repeat` |
| MTP4 | 1 | 65536 | 8192 | 21.066 | 10.000 | 47.47% | `repeat` |
| MTP4 | 1 | 131072 | 64 | 5.807 | 2.949 | 50.79% | `repeat` |
| MTP4 | 1 | 131072 | 8192 | 29.304 | 13.465 | 45.95% | `repeat` |
| MTP4 | 2 | 32768 | 64 | 2.275 | 0.913 | 40.12% | `repeat` |
| MTP4 | 2 | 32768 | 8192 | 24.426 | 10.335 | 42.31% | `full` |
| MTP4 | 2 | 65536 | 64 | 4.848 | 2.199 | 45.37% | `full` |
| MTP4 | 2 | 65536 | 8192 | 31.866 | 13.690 | 42.96% | `full` |
| MTP4 | 2 | 131072 | 64 | 11.164 | 5.898 | 52.83% | `full` |
| MTP4 | 2 | 131072 | 8192 | 36.640 | 14.851 | 40.53% | `full` |
| MTP4 | 4 | 32768 | 64 | 4.377 | 1.827 | 41.74% | `repeat` |
| MTP4 | 4 | 32768 | 8192 | 33.199 | 13.356 | 40.23% | `full` |
| MTP4 | 4 | 65536 | 64 | 9.347 | 4.399 | 47.06% | `full` |
| MTP4 | 4 | 65536 | 8192 | 43.046 | 16.928 | 39.33% | `full` |
| MTP4 | 4 | 131072 | 64 | 21.768 | 11.794 | 54.18% | `repeat` |
| MTP4 | 4 | 131072 | 8192 | 59.426 | 22.200 | 37.36% | `full` |
