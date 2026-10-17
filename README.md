# oh-my-vllm

This program supplies a Rust service for Qwen3.8-27B-FP8 on one NVIDIA B200.
The Python worker loads the model and runs text, image, and video inference.
Use ordinary decode or MTP4, with constrained decoding.

Use the [service guide](docs/serving.md) for installation, commands, and HTTP requests.

[中文](README.zh.md) · [Contribution procedure](CONTRIBUTING.md)

## Performance measurements

Measurements on 2026-10-17 used one NVIDIA B200 and Qwen3.8-27B-FP8.
The runtime sources are in revision [`d516bf6`](https://github.com/para1lel/oh-my-vllm/commit/d516bf6ec62c451158e4f160ed6bc0f587165e02).

### Operators

The table gives expected roofline efficiency for the [modeled workload](docs/operators/evaluation.md).
Each shape uses cold weights, 2 warmup runs, and 5 measured runs.

| Serving operator | Eager | CUDA Graph |
|---|---:|---:|
| FFN | 48.9% | 54.4% |
| FA | 34.4% | 38.1% |
| GDN | 25.1% | 28.5% |
| Vision | 15.2% | 16.1% |
| Sampling | 65.2% | 68.7% |
| MTP context | 23.6% | 27.8% |
| MTP draft | 63.9% | 69.3% |

All 7 operators passed the specified small numerical checks and have full timing coverage.
Only sampling and MTP draft meet the 60% performance limit in the two execution modes.
The full operator suite does not pass.

### End-to-end HTTP

The test uses uncached prefixes, a token budget of 32768, and at most 4 running requests.
Cache capacities are 1024 FA blocks and 128 GDN slots.
Each request uses text input and greedy sampling, with 8192 output tokens.

Each mode uses 1 warmup batch at each batch size: 1, 2, and 4 requests.
Each warmup request has 64 input tokens and 16 output tokens.
Each configuration has 1 measured run.
Times are in seconds. HTTP time covers concurrent submission until the client receives all outputs.

Roofline time is the sum of serving operator lower bounds from the trace, with measured MTP acceptance.
Efficiency is `roofline_seconds / HTTP_seconds`.

| Mode | Batch | Input tokens per request | HTTP time | Roofline time | Roofline efficiency |
|---|---:|---:|---:|---:|---:|
| Ordinary decode | 1 | 32768 | 65.38 | 31.87 | 48.75% |
| Ordinary decode | 2 | 32768 | 81.26 | 35.11 | 43.20% |
| Ordinary decode | 4 | 32768 | 91.65 | 41.57 | 45.36% |
| MTP4 | 1 | 32768 | 22.37 | 9.86 | 44.07% |
| MTP4 | 1 | 65536 | 30.35 | 11.15 | 36.74% |
| MTP4 | 1 | 131072 | 45.74 | 15.19 | 33.20% |
| MTP4 | 2 | 32768 | 30.67 | 11.24 | 36.63% |
| MTP4 | 2 | 65536 | 47.67 | 14.06 | 29.50% |
| MTP4 | 2 | 131072 | 76.32 | 20.46 | 26.80% |
| MTP4 | 4 | 32768 | 47.89 | 14.17 | 29.58% |
| MTP4 | 4 | 65536 | 70.58 | 20.07 | 28.43% |
| MTP4 | 4 | 131072 | 133.70 | 32.16 | 24.06% |

All 28 measured responses have 8192 output tokens and zero cached prefix tokens.
All 12 configurations are below the [60% end-to-end performance limit](docs/performance.md).
