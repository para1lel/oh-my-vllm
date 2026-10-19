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

Final acceptance uses 24 configurations, nominal HGX B200 rates, and the reference calculation in the [GPU execution rules](docs/execution.md#http-performance).
Each configuration must have efficiency of 50% or more at final acceptance. Intermediate changes can report failed or unmeasured configurations.
The complete timer covers submission, service preparation, inference, sampling, transport, and response parsing.
Requests use greedy text generation, uncached prefixes, a 32768-token budget, and at most 4 running requests.

The benchmark uses `http-reference-v1` with the [documented assumptions](docs/accounting.md).
All 24 configurations have measurements for source `2fc07b4a1ccdf72d`.
0 configurations have efficiency of 50% or more. 24 configurations have lower efficiency.
The run uses 1024 FA blocks, 128 GDN slots, and prompt seed `nightly-dual-v1`.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency | 50% or more |
|---|---:|---:|---:|---:|---:|---:|---|
| Ordinary decode | 1 | 32768 | 64 | 2.011 | 0.454 | 22.59% | No |
| Ordinary decode | 1 | 32768 | 8192 | 64.264 | 31.388 | 48.84% | No |
| Ordinary decode | 2 | 32768 | 64 | 2.654 | 0.908 | 34.23% | No |
| Ordinary decode | 2 | 32768 | 8192 | 76.182 | 34.136 | 44.81% | No |
| Ordinary decode | 4 | 32768 | 64 | 4.869 | 1.817 | 37.32% | No |
| Ordinary decode | 4 | 32768 | 8192 | 80.414 | 39.620 | 49.27% | No |
| MTP4 | 1 | 32768 | 64 | 1.330 | 0.457 | 34.33% | No |
| MTP4 | 1 | 32768 | 8192 | 18.872 | 9.272 | 49.13% | No |
| MTP4 | 1 | 65536 | 64 | 2.671 | 1.100 | 41.18% | No |
| MTP4 | 1 | 65536 | 8192 | 22.809 | 10.755 | 47.16% | No |
| MTP4 | 1 | 131072 | 64 | 6.142 | 2.949 | 48.01% | No |
| MTP4 | 1 | 131072 | 8192 | 29.285 | 12.586 | 42.98% | No |
| MTP4 | 2 | 32768 | 64 | 2.487 | 0.914 | 36.74% | No |
| MTP4 | 2 | 32768 | 8192 | 23.975 | 10.093 | 42.10% | No |
| MTP4 | 2 | 65536 | 64 | 5.059 | 2.199 | 43.48% | No |
| MTP4 | 2 | 65536 | 8192 | 28.228 | 11.094 | 39.30% | No |
| MTP4 | 2 | 131072 | 64 | 11.908 | 5.897 | 49.53% | No |
| MTP4 | 2 | 131072 | 8192 | 45.623 | 14.632 | 32.07% | No |
| MTP4 | 4 | 32768 | 64 | 5.704 | 1.827 | 32.04% | No |
| MTP4 | 4 | 32768 | 8192 | 32.033 | 11.935 | 37.26% | No |
| MTP4 | 4 | 65536 | 64 | 9.809 | 4.398 | 44.84% | No |
| MTP4 | 4 | 65536 | 8192 | 40.616 | 14.770 | 36.37% | No |
| MTP4 | 4 | 131072 | 64 | 23.712 | 11.794 | 49.74% | No |
| MTP4 | 4 | 131072 | 8192 | 61.357 | 19.184 | 31.27% | No |
