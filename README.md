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
All 24 configurations have measurements for source `339fa40679328caf`.
The run uses 1024 FA blocks, 128 GDN slots, and prompt seed `nightly-dual-v1`.
The table uses faster measurements from separate runs for two MTP4 configurations with batch 4 and 8192 output tokens.
Their input lengths are 32768 and 131072 tokens.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.534 | 0.454 | 29.61% |
| Ordinary | 1 | 32768 | 8192 | 64.489 | 31.388 | 48.67% |
| Ordinary | 2 | 32768 | 64 | 4.075 | 0.908 | 22.30% |
| Ordinary | 2 | 32768 | 8192 | 71.604 | 34.136 | 47.67% |
| Ordinary | 4 | 32768 | 64 | 4.496 | 1.817 | 40.42% |
| Ordinary | 4 | 32768 | 8192 | 75.132 | 39.620 | 52.73% |
| MTP4 | 1 | 32768 | 64 | 1.309 | 0.457 | 34.90% |
| MTP4 | 1 | 32768 | 8192 | 18.036 | 9.364 | 51.92% |
| MTP4 | 1 | 65536 | 64 | 2.534 | 1.100 | 43.40% |
| MTP4 | 1 | 65536 | 8192 | 21.842 | 11.082 | 50.74% |
| MTP4 | 1 | 131072 | 64 | 5.677 | 2.948 | 51.92% |
| MTP4 | 1 | 131072 | 8192 | 25.241 | 11.785 | 46.69% |
| MTP4 | 2 | 32768 | 64 | 2.293 | 0.913 | 39.82% |
| MTP4 | 2 | 32768 | 8192 | 21.080 | 9.909 | 47.00% |
| MTP4 | 2 | 65536 | 64 | 4.852 | 2.200 | 45.34% |
| MTP4 | 2 | 65536 | 8192 | 25.236 | 11.531 | 45.69% |
| MTP4 | 2 | 131072 | 64 | 11.507 | 5.898 | 51.26% |
| MTP4 | 2 | 131072 | 8192 | 34.752 | 14.635 | 42.11% |
| MTP4 | 4 | 32768 | 64 | 5.476 | 1.826 | 33.35% |
| MTP4 | 4 | 32768 | 8192 | 26.623 | 12.063 | 45.31% |
| MTP4 | 4 | 65536 | 64 | 9.198 | 4.399 | 47.83% |
| MTP4 | 4 | 65536 | 8192 | 34.700 | 15.482 | 44.62% |
| MTP4 | 4 | 131072 | 64 | 21.770 | 11.795 | 54.18% |
| MTP4 | 4 | 131072 | 8192 | 49.632 | 19.997 | 40.29% |
