# oh-my-vllm

This program supplies a Rust service for Qwen3.8-27B-FP8 on one NVIDIA B200.
The Python worker loads the model and runs text, image, and video inference.
Use ordinary decode or MTP4, with constrained decoding.

Use the [service guide](docs/serving.md) for installation, commands, and HTTP requests.

[中文](README.zh.md) · [Contribution procedure](CONTRIBUTING.md)

## Performance measurements

Measurements on 2026-10-19 used NVIDIA B200 GPUs and Qwen3.8-27B-FP8.
The source fingerprints are HTTP `dfa05a0cd304f8495c8ee31df8ea20f8ad8d6a690e4125d315be276f398f6a37` and operator suite `b9c18e05c43fe7dbda38b3830f4041847c7d13bad716a5175a2e58419ad4faef`.
Numbers are rounded after calculation.
Operator measurements use GPU `GPU-ade4492d-f119-ca40-f613-13058a658f63`. HTTP measurements use GPU `GPU-b0da5a7e-8dcf-e7dd-9836-a29a45d01b45`.
Each measurement uses the calibration for its GPU.

### Measured GPU peaks

The [calibration procedure](docs/calibration.md) measures each GPU UUID separately.
HBM uses vector copies. Dense GEMMs measure Tensor Core throughput. Native FMA loops measure scalar throughput.

Each resource uses the best median across configurations, with seven samples of ten invocations after warmup.
These measured rates replace nominal peaks in the latency lower bound and roofline efficiency calculations.
The lower bound is a model estimate based on calibration. It is not a physical limit.

| GPU UUID | HBM (TB/s) | FP8 (PFLOP/s) | BF16 (PFLOP/s) | TF32 (PFLOP/s) | FP32 (TFLOP/s) | FP64 (TFLOP/s) |
|---|---:|---:|---:|---:|---:|---:|
| `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` | 5.846 | 2.544 | 1.289 | 0.661 | 42.485 | 21.571 |
| `GPU-1b174534-ebba-826f-b452-8e7f3c05c301` | 6.146 | 4.247 | 2.168 | 1.101 | 72.626 | 36.909 |
| `GPU-ade4492d-f119-ca40-f613-13058a658f63` | 6.049 | 4.237 | 2.151 | 1.099 | 72.564 | 36.842 |
| `GPU-b0da5a7e-8dcf-e7dd-9836-a29a45d01b45` | 6.131 | 4.242 | 2.166 | 1.099 | 72.624 | 36.890 |
| `GPU-427d2866-8615-6524-b2db-2333b40b0c5e` | 6.102 | 4.241 | 2.167 | 1.101 | 72.632 | 36.923 |
| `GPU-81c2e3a2-685d-59a9-e801-7e068fffc377` | 6.141 | 4.247 | 2.139 | 1.101 | 72.637 | 36.921 |
| `GPU-a832d9c1-260f-9e37-0c99-95e62ab16ca5` | 6.074 | 4.247 | 2.161 | 1.101 | 72.625 | 36.916 |
| `GPU-80cebbaf-a106-2605-b902-f5f9a08645ea` | 6.110 | 4.244 | 2.170 | 1.100 | 72.616 | 36.884 |

All GPUs use driver 590.48.01, CUDA 13.1, a 1000 W power limit, and a 3996 MHz memory clock.
Calibration snapshots show a 1155 MHz SM clock on GPU0 and 1965 MHz during GPU1–GPU7 computation.

### Operators

Operator tests have no minimum efficiency requirement. Each delivery must include full measurements and a table update.
The table uses the [modeled workload](docs/operators/evaluation.md) and cold weights.
Each shape has two warmup runs and five measured runs. Timing uses the median.
The [evaluator](development/kernels/__main__.py) prepares graph capture with one untimed eager run.

Time columns use invocation-weighted means of per-shape lower bounds and median durations.
Efficiencies use the invocation-weighted arithmetic mean of each shape's efficiency.
They are not ratios of the time columns.

| Serving operator | Lower bound (µs/call) | Eager (µs/call) | Eager efficiency | CUDA Graph (µs/call) | CUDA Graph efficiency |
|---|---:|---:|---:|---:|---:|
| FFN | 88.66 | 140.59 | 66.30% | 136.17 | 70.93% |
| FA | 237.40 | 365.33 | 47.53% | 356.40 | 53.29% |
| GDN | 41.26 | 122.87 | 35.96% | 112.39 | 43.73% |
| Vision | 117.86 | 512.94 | 18.56% | 501.93 | 19.31% |
| Sampling | 420.47 | 514.52 | 83.05% | 488.62 | 87.36% |
| MTP context | 58.67 | 139.08 | 37.92% | 133.34 | 42.42% |
| MTP draft | 557.61 | 679.49 | 82.04% | 628.54 | 88.68% |

The fixed small numerical checks passed for all seven operators.
Measurements cover all 75 modeled cases in each execution mode. Large cases supply timing data only.
The complete operator audit passed.

### End-to-end HTTP

Requests use text input, greedy sampling, and uncached prefixes.
The scheduler token budget is 32768. At most four requests run at the same time.
Cache capacities are 1024 FA blocks and 128 GDN slots.

Each mode has one warmup batch at each batch size, with 64 input tokens and 16 output tokens per request.
Each configuration has one measured run. Prompt text is prepared before timing.

HTTP time includes request thread creation, server request preparation, queue time, inference, transport, JSON parsing, and thread shutdown.
Roofline time adds serving operator lower bounds from the recorded trace, with measured MTP acceptance counts.
Efficiency is `roofline_seconds / HTTP_seconds`.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Roofline time (s) | Roofline efficiency | Meets 50% |
|---|---:|---:|---:|---:|---:|---:|---|
| Ordinary decode | 1 | 32768 | 64 | 1.52 | 0.78 | 51.12% | Yes |
| Ordinary decode | 1 | 32768 | 8192 | 60.37 | 39.90 | 66.10% | Yes |
| Ordinary decode | 2 | 32768 | 64 | 2.61 | 1.28 | 49.16% | No |
| Ordinary decode | 2 | 32768 | 8192 | 70.87 | 43.83 | 61.84% | Yes |
| Ordinary decode | 4 | 32768 | 64 | 4.63 | 2.29 | 49.39% | No |
| Ordinary decode | 4 | 32768 | 8192 | 77.11 | 51.67 | 67.01% | Yes |
| MTP4 | 1 | 32768 | 64 | 1.28 | 0.62 | 48.28% | No |
| MTP4 | 1 | 32768 | 8192 | 18.91 | 12.33 | 65.20% | Yes |
| MTP4 | 1 | 65536 | 64 | 4.70 | 1.29 | 27.51% | No |
| MTP4 | 1 | 65536 | 8192 | 21.81 | 14.11 | 64.72% | Yes |
| MTP4 | 1 | 131072 | 64 | 6.12 | 3.26 | 53.32% | Yes |
| MTP4 | 1 | 131072 | 8192 | 27.59 | 17.88 | 64.79% | Yes |
| MTP4 | 2 | 32768 | 64 | 2.63 | 1.12 | 42.56% | No |
| MTP4 | 2 | 32768 | 8192 | 24.21 | 13.92 | 57.48% | Yes |
| MTP4 | 2 | 65536 | 64 | 5.03 | 2.47 | 49.17% | No |
| MTP4 | 2 | 65536 | 8192 | 30.35 | 17.83 | 58.76% | Yes |
| MTP4 | 2 | 131072 | 64 | 11.91 | 6.41 | 53.84% | Yes |
| MTP4 | 2 | 131072 | 8192 | 41.03 | 23.67 | 57.69% | Yes |
| MTP4 | 4 | 32768 | 64 | 5.66 | 2.10 | 37.04% | No |
| MTP4 | 4 | 32768 | 8192 | 32.57 | 18.05 | 55.43% | Yes |
| MTP4 | 4 | 65536 | 64 | 10.50 | 4.82 | 45.86% | No |
| MTP4 | 4 | 65536 | 8192 | 39.82 | 23.04 | 57.86% | Yes |
| MTP4 | 4 | 131072 | 64 | 27.11 | 12.65 | 46.64% | No |
| MTP4 | 4 | 131072 | 8192 | 62.66 | 37.03 | 59.10% | Yes |

All 56 measured responses have the configured input and output lengths, with zero cached prefix tokens.
15 of the 24 configurations meet the [HTTP efficiency requirement of 50%](docs/performance.md).
The full HTTP comparison did not pass.
