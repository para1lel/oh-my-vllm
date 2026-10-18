# oh-my-vllm

This program supplies a Rust service for Qwen3.8-27B-FP8 on one NVIDIA B200.
The Python worker loads the model and runs text, image, and video inference.
Use ordinary decode or MTP4, with constrained decoding.

Use the [service guide](docs/serving.md) for installation, commands, and HTTP requests.

[中文](README.zh.md) · [Contribution procedure](CONTRIBUTING.md)

## Performance measurements

Measurements on 2026-10-18 used one NVIDIA B200 and Qwen3.8-27B-FP8.
The runtime source revision is [`6720f42`](https://github.com/para1lel/oh-my-vllm/commit/6720f42048ce3e1370c4e38cba54d366eb383fd3). Numbers are rounded after calculation.

### Operators

The table gives expected roofline efficiency for the [modeled workload](docs/operators/evaluation.md).
Case efficiencies use their modeled invocation counts as weights in the arithmetic mean.
Each shape uses cold weights, 2 warmup runs, and the median of 5 measured runs.
The [operator evaluator](development/kernels/__main__.py) prepares each shape with an eager run before graph capture.
The measured time does not include this run.

| Serving operator | Eager | CUDA Graph |
|---|---:|---:|
| FFN | 49.63% | 55.30% |
| FA | 35.77% | 40.31% |
| GDN | 27.15% | 31.72% |
| Vision | 11.09% | 11.39% |
| Sampling | 65.22% | 68.65% |
| MTP context | 21.70% | 33.54% |
| MTP draft | 44.97% | 51.26% |

The specified small numerical checks were satisfactory for all 7 operators.
Measurements include all 75 modeled cases in each execution mode.
The efficiency is 50% or more for 4 of the 14 operator/mode results.
The full operator audit is not accepted because SM clock values recorded before numerical checks were 1965 or 1155 MHz.

### End-to-end HTTP

Requests use text input, greedy sampling, and uncached prefixes.
The token budget per scheduler step is 32768.
The maximum number of running requests is 4.
Cache capacities are 1024 FA blocks and 128 GDN slots.

Each mode has 1 warmup batch at each batch size: 1, 2, and 4.
Each warmup request has 64 input tokens and 16 output tokens.
Each configuration has 1 measured run.
Prompt text is prepared before timing.

HTTP time includes request thread creation, server request preparation, queue time, full inference, transport, JSON parsing, and thread shutdown.
Roofline time is the sum of serving operator lower bounds from the execution trace. It uses measured MTP acceptance counts.
Efficiency is `roofline_seconds / HTTP_seconds`.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Roofline time (s) | Roofline efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary decode | 1 | 32768 | 64 | 2.79 | 0.72 | 25.89% |
| Ordinary decode | 1 | 32768 | 8192 | 62.66 | 31.87 | 50.87% |
| Ordinary decode | 2 | 32768 | 64 | 4.62 | 1.23 | 26.52% |
| Ordinary decode | 2 | 32768 | 8192 | 74.90 | 35.10 | 46.86% |
| Ordinary decode | 4 | 32768 | 64 | 14.89 | 2.23 | 14.98% |
| Ordinary decode | 4 | 32768 | 8192 | 83.71 | 41.55 | 49.63% |
| MTP4 | 1 | 32768 | 64 | 3.31 | 0.59 | 17.93% |
| MTP4 | 1 | 32768 | 8192 | 21.27 | 10.28 | 48.33% |
| MTP4 | 1 | 65536 | 64 | 5.01 | 1.28 | 25.48% |
| MTP4 | 1 | 65536 | 8192 | 25.34 | 11.70 | 46.16% |
| MTP4 | 1 | 131072 | 64 | 14.45 | 3.20 | 22.16% |
| MTP4 | 1 | 131072 | 8192 | 35.23 | 15.62 | 44.34% |
| MTP4 | 2 | 32768 | 64 | 4.21 | 1.07 | 25.41% |
| MTP4 | 2 | 32768 | 8192 | 35.31 | 12.44 | 35.24% |
| MTP4 | 2 | 65536 | 64 | 8.90 | 2.46 | 27.62% |
| MTP4 | 2 | 65536 | 8192 | 38.43 | 14.95 | 38.91% |
| MTP4 | 2 | 131072 | 64 | 21.16 | 6.29 | 29.73% |
| MTP4 | 2 | 131072 | 8192 | 55.37 | 20.08 | 36.26% |
| MTP4 | 4 | 32768 | 64 | 8.15 | 2.09 | 25.63% |
| MTP4 | 4 | 32768 | 8192 | 34.81 | 13.92 | 39.99% |
| MTP4 | 4 | 65536 | 64 | 17.28 | 4.80 | 27.77% |
| MTP4 | 4 | 65536 | 8192 | 55.90 | 21.48 | 38.42% |
| MTP4 | 4 | 131072 | 64 | 41.72 | 12.47 | 29.90% |
| MTP4 | 4 | 131072 | 8192 | 82.22 | 32.46 | 39.48% |

All 56 measured responses have the configured input and output lengths, with zero cached prefix tokens.
The efficiency is [50%](docs/performance.md) or more for 1 of the 24 configurations.
The full HTTP comparison is not accepted.
