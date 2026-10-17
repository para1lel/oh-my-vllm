# oh-my-vllm

This program supplies a Rust service for Qwen3.8-27B-FP8 on one NVIDIA B200.
The Python worker loads the model and runs text, image, and video inference.
Use ordinary decode or MTP4, with constrained decoding.

Use the [service guide](docs/serving.md) for installation, commands, and HTTP requests.

[中文](README.zh.md) · [Contribution procedure](CONTRIBUTING.md)

## Performance measurements

Measurements on 2026-10-17 used one NVIDIA B200 and Qwen3.8-27B-FP8.
The runtime sources are in revision [`4356493`](https://github.com/para1lel/oh-my-vllm/commit/43564930ca5d940e8e4ff2643c501c9de1784195).

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
Each request generates 32 tokens.
The table gives full batch latency in seconds, from concurrent HTTP submission through receipt of all outputs.

| Mode | Input tokens per request | Batch 1 | Batch 2 | Batch 4 |
|---|---:|---:|---:|---:|
| Ordinary decode | 32768 | 3.80 | 7.38 | 14.49 |
| MTP4 | 32768 | 3.69 | 12.33 | 14.64 |
| MTP4 | 65536 | 8.53 | 17.02 | 37.79 |
| MTP4 | 131072 | 22.07 | 47.47 | 86.36 |

All 12 configurations passed the short functional check, with 28 responses.
Their roofline efficiencies are 8.4% to 15.8%, below the 60% limit.
These short runs do not complete the [8192-token end-to-end performance gate](docs/performance.md).
