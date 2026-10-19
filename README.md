# oh-my-vllm

This program supplies a Rust service for Qwen3.8-27B-FP8 on one NVIDIA B200.
The Python worker loads the model and runs text, image, and video inference.
Use ordinary decode or MTP4, with constrained decoding.

Use the [service guide](docs/serving.md) for installation, commands, and HTTP requests.

[中文](README.zh.md) · [Contribution procedure](CONTRIBUTING.md)

## Tile execution migration

The [design](docs/tile-execution.md) specifies final GPU coverage and numerical behavior.
The [plan](docs/tile-migration.md) gives the migration sequence.
The runtime still uses the existing native operators. Implementation migration has not started.

## End-to-end HTTP performance

Final acceptance uses 24 configurations, nominal HGX B200 rates, and the reference calculation in the design.
Each configuration must reach 50%. Migration stages have no performance threshold.
The complete timer covers submission, service preparation, inference, sampling, transport, and response parsing.
Requests use greedy text generation, uncached prefixes, a 32768-token budget, and at most 4 running requests.

The current benchmark uses the old accounting method. It cannot supply acceptance evidence for the new method.
The new accounting version and measurements are pending. Old calibrated results are not used in this table.
A dash means no measurement with the new accounting method. It does not mean zero time or a passing result.

| Mode | Batch | Input tokens per request | Output tokens per request | HTTP time (s) | Reference time (s) | Efficiency | Meets 50% |
|---|---:|---:|---:|---:|---:|---:|---|
| Ordinary decode | 1 | 32768 | 64 | — | — | — | Unmeasured |
| Ordinary decode | 1 | 32768 | 8192 | — | — | — | Unmeasured |
| Ordinary decode | 2 | 32768 | 64 | — | — | — | Unmeasured |
| Ordinary decode | 2 | 32768 | 8192 | — | — | — | Unmeasured |
| Ordinary decode | 4 | 32768 | 64 | — | — | — | Unmeasured |
| Ordinary decode | 4 | 32768 | 8192 | — | — | — | Unmeasured |
| MTP4 | 1 | 32768 | 64 | — | — | — | Unmeasured |
| MTP4 | 1 | 32768 | 8192 | — | — | — | Unmeasured |
| MTP4 | 1 | 65536 | 64 | — | — | — | Unmeasured |
| MTP4 | 1 | 65536 | 8192 | — | — | — | Unmeasured |
| MTP4 | 1 | 131072 | 64 | — | — | — | Unmeasured |
| MTP4 | 1 | 131072 | 8192 | — | — | — | Unmeasured |
| MTP4 | 2 | 32768 | 64 | — | — | — | Unmeasured |
| MTP4 | 2 | 32768 | 8192 | — | — | — | Unmeasured |
| MTP4 | 2 | 65536 | 64 | — | — | — | Unmeasured |
| MTP4 | 2 | 65536 | 8192 | — | — | — | Unmeasured |
| MTP4 | 2 | 131072 | 64 | — | — | — | Unmeasured |
| MTP4 | 2 | 131072 | 8192 | — | — | — | Unmeasured |
| MTP4 | 4 | 32768 | 64 | — | — | — | Unmeasured |
| MTP4 | 4 | 32768 | 8192 | — | — | — | Unmeasured |
| MTP4 | 4 | 65536 | 64 | — | — | — | Unmeasured |
| MTP4 | 4 | 65536 | 8192 | — | — | — | Unmeasured |
| MTP4 | 4 | 131072 | 64 | — | — | — | Unmeasured |
| MTP4 | 4 | 131072 | 8192 | — | — | — | Unmeasured |
