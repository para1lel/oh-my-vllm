# Qwen HTTP results

## Scope and method

This report contains the last completed Qwen HTTP matrix.
The source identity below identifies the measured implementation.
Qwen keeps its single-worker execution path, coordinator, wire fields, and seven native modules.

These measurements do not execute the typed four-rank model path.
They do not measure DeepSeek, four-GPU execution, DSpark, or million-token capacity.
The date is 2026-10-24 in Asia/Shanghai.
The [Qwen procedure](performance.md) gives the 24 HTTP configurations and the `serving-tools-v1` workload.

Each row uses one B200, concurrent HTTP requests, and the specified uncached input/output lengths for each request.
The service has a limit of four active requests, a 32768-token step budget, and a 262144-token context limit.
Sampling uses temperature 1.0, top-k 20, top-p 0.95, presence penalty 0.5, the repetition guard, and client-ordinal seeds.
Reasoning is off. Tool choice is auto.
Required-tool checks and short warmup requests occur before the measured cases.

HTTP seconds include request-thread creation, submission, preparation, queueing, model execution, and full JSON response reads.
The timer does not include initialization or trace analysis.

Reference seconds use the recorded `http-reference-v4` [lower-bound calculation](performance.md#lower-bound-calculation).
This is a conservative theoretical resource estimate, not a proof of minimum physical latency.
Efficiency is reference seconds divided by full HTTP seconds, expressed as a percentage.
Each row has one measured request per batch member.
Accepted/proposed counts come from measured requests and agree between service logs and execution traces.
Ordinary decode has no draft acceptance ratio.

## HTTP measurements

| Mode | Batch | Input | Output | HTTP s | Reference s | Efficiency | Accepted/proposed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.840 | 0.697 | 37.90% | - |
| Ordinary | 1 | 32768 | 8192 | 64.060 | 32.001 | 49.95% | - |
| Ordinary | 2 | 32768 | 64 | 3.376 | 1.178 | 34.88% | - |
| Ordinary | 2 | 32768 | 8192 | 76.638 | 35.359 | 46.14% | - |
| Ordinary | 4 | 32768 | 64 | 6.048 | 2.131 | 35.23% | - |
| Ordinary | 4 | 32768 | 8192 | 83.054 | 42.067 | 50.65% | - |
| MTP4 | 1 | 32768 | 64 | 1.599 | 0.577 | 36.11% | 41/85 |
| MTP4 | 1 | 32768 | 8192 | 23.508 | 11.665 | 49.62% | 6153/8150 |
| MTP4 | 1 | 65536 | 64 | 3.273 | 1.214 | 37.08% | 44/74 |
| MTP4 | 1 | 65536 | 8192 | 35.139 | 17.475 | 49.73% | 5391/11200 |
| MTP4 | 1 | 131072 | 64 | 7.828 | 3.113 | 39.77% | 39/93 |
| MTP4 | 1 | 131072 | 8192 | 30.420 | 14.958 | 49.17% | 6357/7336 |
| MTP4 | 2 | 32768 | 64 | 3.074 | 1.049 | 34.11% | 85/156 |
| MTP4 | 2 | 32768 | 8192 | 31.342 | 13.518 | 43.13% | 12272/16435 |
| MTP4 | 2 | 65536 | 64 | 6.460 | 2.331 | 36.08% | 92/132 |
| MTP4 | 2 | 65536 | 8192 | 31.268 | 13.775 | 44.05% | 12960/13684 |
| MTP4 | 2 | 131072 | 64 | 15.444 | 6.081 | 39.38% | 90/140 |
| MTP4 | 2 | 131072 | 8192 | 50.556 | 23.608 | 46.70% | 12348/16134 |
| MTP4 | 4 | 32768 | 64 | 5.892 | 1.974 | 33.51% | 176/291 |
| MTP4 | 4 | 32768 | 8192 | 40.019 | 16.258 | 40.63% | 25279/29931 |
| MTP4 | 4 | 65536 | 64 | 12.670 | 4.607 | 36.36% | 173/307 |
| MTP4 | 4 | 65536 | 8192 | 51.195 | 21.918 | 42.81% | 24867/31577 |
| MTP4 | 4 | 131072 | 64 | 32.169 | 12.062 | 37.49% | 178/287 |
| MTP4 | 4 | 131072 | 8192 | 75.526 | 34.797 | 46.07% | 25171/30358 |

## Failures and memory

All 24 configurations completed, with 56 measured requests and no measured request failures.
The benchmark checked full output lengths, zero cached input tokens, tool frames, and runtime identities.

The table gives startup accounting from worker trace identities, in bytes.
Initialized usage is a device-wide observation after worker setup, not a measured per-case peak.
The benchmark did not collect peak GPU memory, pinned-host peaks, or an inter-token latency distribution.
Per-request queue and processing times are kept in the external validation record.

| Mode | FA blocks | GDN blocks | FA bytes | GDN bytes | Planned bytes | Initialized used bytes |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1824 | 128 | 93717528576 | 30482952192 | 166469238272 | 166328991744 |
| MTP4 | 1608 | 128 | 87783112704 | 30482952192 | 168011158016 | 163583885312 |

Reserved pinned host cache at startup: ordinary decode 124200480768 bytes, MTP4 118266064896 bytes.
Device allocation and reservation changed during the run.
Pinned host-cache capacity stayed fixed in service snapshots.

## Provenance

Portable artifact name: `deepseek-v41-nightly/qwen-runtime-regression-v10/http`.
Prompt seed: `nightly-model-adapters-v1`.
The topology is TP1 on one B200.
The external identity file records the selected device UUID.

Driver: `590.48.01`. Memory clock: 3996 MHz. Power limit: 1000 W.
The run did not change device clocks or power settings.

Request timeout is 7200 seconds. Worker-step timeout is 600 seconds. Shutdown grace is 35 seconds.

The source identity binds paths and bytes in the benchmark source closure.
The release executable and all seven native modules are bound independently.
The identity file includes checkpoint shard, tokenizer asset, native source, and native binary identities.
Nested identity digests use UTF-8 JSON with sorted keys and separators `(",", ":")`.
Raw artifacts and service logs are external to the repository.

| Identity | SHA256 |
|---|---|
| Source | `ab35cfd393b8e6dd882f110909fed0c658a62c9a57c0be9962b25a6ef37c7065` |
| Executable | `03255e5476c1026c07e6cd5f1c917388d3a037ba2e1b41b44b969d206dfc174d` |
| Native set | `d07239111105d99ea7de179631a477fadd725e540b83950e401614ccd93001db` |
| Checkpoint | `d26f1d65231264de91c0966777e2c6fa2658f72c278c50f40aead734ce092f4b` |
| Tokenizer | `f78b803474f10eb7a86741417823af5be59cf86f2b57db1869740e18c380bfdd` |
| Identity file | `5966d3a34af2969ac1d73f21976897d7bd548583f1ed0e446947a39e987dff98` |
| Validation record | `57995a6e687938b9e11ed63bd617790a9f321c49aac22b3b1ad413b515b3f611` |
