# oh-my-vllm

This program supplies a Rust service for Qwen3.8-27B-FP8 on one NVIDIA B200.
The Python worker loads the model and runs text, image, and video inference.
Use ordinary decode or MTP4, with constrained decoding.

Use the [service guide](docs/serving.md) for installation, commands, and HTTP requests.

[中文](README.zh.md) · [Contribution procedure](CONTRIBUTING.md)

## Performance measurements

Measurements on 2026-10-18 used NVIDIA B200 GPUs and Qwen3.8-27B-FP8.
The runtime source revision is [`52ca39c`](https://github.com/para1lel/oh-my-vllm/commit/52ca39cca7d48ebdeb5d1864926ec462eb5a98de). Numbers are rounded after calculation.
Operator and HTTP measurements use only GPU `GPU-1b174534-ebba-826f-b452-8e7f3c05c301` and its calibration.

### Measured GPU peaks

The [calibration procedure](docs/calibration.md) measures each GPU UUID separately.
HBM uses vector copies. Dense GEMMs measure Tensor Core throughput. Native FMA loops measure scalar throughput.

Each resource uses the best median across configurations, with seven samples of ten invocations after warmup.
These measured rates replace nominal peaks in the latency lower bound and roofline efficiency calculations.
The lower bound is a model estimate based on calibration. It is not a physical limit.

| GPU UUID | HBM (TB/s) | FP8 (PFLOP/s) | BF16 (PFLOP/s) | TF32 (PFLOP/s) | FP32 (TFLOP/s) | FP64 (TFLOP/s) |
|---|---:|---:|---:|---:|---:|---:|
| `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` | 5.850 | 2.544 | 1.289 | 0.661 | 42.487 | 21.572 |
| `GPU-1b174534-ebba-826f-b452-8e7f3c05c301` | 6.135 | 4.247 | 2.130 | 1.099 | 72.627 | 36.906 |
| `GPU-ade4492d-f119-ca40-f613-13058a658f63` | 6.041 | 4.238 | 2.151 | 1.099 | 72.591 | 36.854 |
| `GPU-b0da5a7e-8dcf-e7dd-9836-a29a45d01b45` | 6.147 | 4.241 | 2.166 | 1.099 | 72.632 | 36.891 |
| `GPU-427d2866-8615-6524-b2db-2333b40b0c5e` | 6.124 | 4.240 | 2.172 | 1.101 | 72.621 | 36.919 |
| `GPU-81c2e3a2-685d-59a9-e801-7e068fffc377` | 6.129 | 4.247 | 2.171 | 1.101 | 72.630 | 36.916 |
| `GPU-a832d9c1-260f-9e37-0c99-95e62ab16ca5` | 6.119 | 4.248 | 2.171 | 1.101 | 72.631 | 36.905 |
| `GPU-80cebbaf-a106-2605-b902-f5f9a08645ea` | 6.154 | 4.244 | 2.156 | 1.100 | 72.606 | 36.891 |

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
| FFN | 87.94 | 166.68 | 63.25% | 162.67 | 69.11% |
| FA | 238.02 | 589.29 | 44.83% | 576.99 | 50.56% |
| GDN | 47.50 | 269.00 | 33.83% | 259.63 | 39.79% |
| Vision | 119.01 | 514.32 | 18.66% | 502.02 | 19.49% |
| Sampling | 414.57 | 514.83 | 81.84% | 490.16 | 85.83% |
| MTP context | 58.76 | 144.66 | 37.31% | 138.88 | 41.77% |
| MTP draft | 549.78 | 682.01 | 80.59% | 628.82 | 87.39% |

The fixed small numerical checks passed for all seven operators.
Measurements cover all 75 modeled cases in each execution mode. Large cases supply timing data only.
The complete operator audit passed.

### End-to-end HTTP

Requests use text input, greedy sampling, and uncached prefixes.
The scheduler token budget is 32768. At most four requests run concurrently.
Cache capacities are 1024 FA blocks and 128 GDN slots.

Each mode has one warmup batch at each batch size, with 64 input tokens and 16 output tokens per request.
Each configuration has one measured run. Prompt text is prepared before timing.

HTTP time includes request thread creation, server request preparation, queue time, inference, transport, JSON parsing, and thread shutdown.
Roofline time adds serving operator lower bounds from the recorded trace, with measured MTP acceptance counts.
Efficiency is `roofline_seconds / HTTP_seconds`.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Roofline time (s) | Roofline efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary decode | 1 | 32768 | 64 | 2.62 | 0.81 | 30.91% |
| Ordinary decode | 1 | 32768 | 8192 | 62.63 | 39.91 | 63.71% |
| Ordinary decode | 2 | 32768 | 64 | 4.42 | 1.34 | 30.33% |
| Ordinary decode | 2 | 32768 | 8192 | 74.65 | 43.86 | 58.75% |
| Ordinary decode | 4 | 32768 | 64 | 9.28 | 2.41 | 25.93% |
| Ordinary decode | 4 | 32768 | 8192 | 84.44 | 51.76 | 61.30% |
| MTP4 | 1 | 32768 | 64 | 2.38 | 0.63 | 26.53% |
| MTP4 | 1 | 32768 | 8192 | 21.41 | 12.82 | 59.89% |
| MTP4 | 1 | 65536 | 64 | 4.97 | 1.36 | 27.45% |
| MTP4 | 1 | 65536 | 8192 | 27.67 | 16.04 | 57.96% |
| MTP4 | 1 | 131072 | 64 | 11.52 | 3.39 | 29.40% |
| MTP4 | 1 | 131072 | 8192 | 35.15 | 18.39 | 52.32% |
| MTP4 | 2 | 32768 | 64 | 4.22 | 1.17 | 27.61% |
| MTP4 | 2 | 32768 | 8192 | 30.54 | 14.72 | 48.18% |
| MTP4 | 2 | 65536 | 64 | 9.50 | 2.62 | 27.55% |
| MTP4 | 2 | 65536 | 8192 | 36.73 | 17.30 | 47.09% |
| MTP4 | 2 | 131072 | 64 | 21.30 | 6.66 | 31.27% |
| MTP4 | 2 | 131072 | 8192 | 55.14 | 24.13 | 43.75% |
| MTP4 | 4 | 32768 | 64 | 8.21 | 2.22 | 27.08% |
| MTP4 | 4 | 32768 | 8192 | 36.65 | 18.34 | 50.04% |
| MTP4 | 4 | 65536 | 64 | 22.06 | 5.10 | 23.10% |
| MTP4 | 4 | 65536 | 8192 | 48.13 | 23.06 | 47.91% |
| MTP4 | 4 | 131072 | 64 | 41.79 | 13.21 | 31.61% |
| MTP4 | 4 | 131072 | 8192 | 81.03 | 37.61 | 46.41% |

All 56 measured responses have the configured input and output lengths, with zero cached prefix tokens.
7 of the 24 configurations meet the [HTTP efficiency requirement of 50%](docs/performance.md).
The full HTTP comparison did not pass.
