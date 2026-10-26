# Qwen HTTP results

## Scope and method

This report contains the last completed Qwen HTTP matrix.
The source identity below identifies the measured implementation.
Qwen keeps its single-worker execution path, coordinator, wire fields, and seven native modules.

These measurements do not execute the typed four-rank model path.
They do not measure multi-GPU execution, speculative methods other than MTP, or million-token capacity.
The date is 2026-10-25 in Asia/Shanghai.
The [Qwen procedure](performance.md) gives the 24 HTTP configurations and the `serving-tools-v1` workload.

Each row uses one B200, concurrent HTTP requests, and the specified uncached input/output lengths for each request.
The service has a limit of four active requests, a 32768-token step budget, and a 262144-token context limit.
Sampling uses temperature 1.0, top-k 20, top-p 0.95, presence penalty 0.5, the repetition guard, and client-ordinal seeds.
Reasoning is off. Tool choice is auto.
Required-tool checks and short warmup requests occur before the measured cases.

HTTP seconds include request-thread creation, submission, preparation, queueing, model execution, and full JSON response reads.
The timer does not include initialization or trace analysis.
Measured target prefill fragments used Eager. Target decode and verification fragments used CUDA Graphs.
This matrix does not measure prefill Graph promotion.

Reference seconds use the recorded `http-reference-v4` [lower-bound calculation](performance.md#lower-bound-calculation).
This is a conservative theoretical resource estimate, not a proof of minimum physical latency.
Efficiency is reference seconds divided by full HTTP seconds, expressed as a percentage.
Each row has one measured request per batch member.
Accepted/proposed counts come from measured requests and agree between service logs and execution traces.
Ordinary decode has no draft acceptance ratio.

## HTTP measurements

| Mode | Batch | Input | Output | HTTP s | Reference s | Efficiency | Accepted/proposed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.825 | 0.697 | 38.21% | - |
| Ordinary | 1 | 32768 | 8192 | 63.901 | 32.001 | 50.08% | - |
| Ordinary | 2 | 32768 | 64 | 3.414 | 1.178 | 34.49% | - |
| Ordinary | 2 | 32768 | 8192 | 75.766 | 35.359 | 46.67% | - |
| Ordinary | 4 | 32768 | 64 | 6.158 | 2.131 | 34.60% | - |
| Ordinary | 4 | 32768 | 8192 | 82.775 | 42.067 | 50.82% | - |
| MTP4 | 1 | 32768 | 64 | 1.590 | 0.577 | 36.30% | 41/85 |
| MTP4 | 1 | 32768 | 8192 | 23.672 | 11.665 | 49.28% | 6153/8150 |
| MTP4 | 1 | 65536 | 64 | 3.255 | 1.214 | 37.28% | 44/74 |
| MTP4 | 1 | 65536 | 8192 | 35.150 | 17.475 | 49.72% | 5391/11200 |
| MTP4 | 1 | 131072 | 64 | 7.796 | 3.113 | 39.94% | 39/93 |
| MTP4 | 1 | 131072 | 8192 | 30.402 | 14.958 | 49.20% | 6357/7336 |
| MTP4 | 2 | 32768 | 64 | 3.101 | 1.049 | 33.83% | 85/156 |
| MTP4 | 2 | 32768 | 8192 | 31.448 | 13.518 | 42.99% | 12272/16435 |
| MTP4 | 2 | 65536 | 64 | 6.429 | 2.331 | 36.26% | 92/132 |
| MTP4 | 2 | 65536 | 8192 | 31.320 | 13.775 | 43.98% | 12960/13684 |
| MTP4 | 2 | 131072 | 64 | 15.358 | 6.081 | 39.60% | 90/140 |
| MTP4 | 2 | 131072 | 8192 | 50.637 | 23.608 | 46.62% | 12348/16134 |
| MTP4 | 4 | 32768 | 64 | 5.870 | 1.974 | 33.64% | 176/291 |
| MTP4 | 4 | 32768 | 8192 | 39.898 | 15.922 | 39.91% | 24868/31573 |
| MTP4 | 4 | 65536 | 64 | 12.641 | 4.585 | 36.27% | 176/300 |
| MTP4 | 4 | 65536 | 8192 | 51.108 | 21.918 | 42.89% | 24867/31577 |
| MTP4 | 4 | 131072 | 64 | 30.856 | 12.062 | 39.09% | 178/287 |
| MTP4 | 4 | 131072 | 8192 | 77.974 | 34.797 | 44.63% | 25171/30358 |

## Failures and memory

All 24 configurations completed, with 56 measured requests and no measured request failures.
The benchmark checked full output lengths, zero cached input tokens, tool frames, and runtime identities.

The table gives startup accounting from worker trace identities, in bytes.
Initialized usage is a device-wide observation after worker setup, not a measured per-case peak.
The benchmark did not collect peak GPU memory, pinned-host peaks, or an inter-token latency distribution.
Per-request queue and processing times are kept in the external service logs.

| Mode | FA blocks | GDN blocks | FA bytes | GDN bytes | Planned bytes | Initialized used bytes |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1824 | 128 | 93717528576 | 30482952192 | 166469303808 | 166329057280 |
| MTP4 | 1608 | 128 | 87783112704 | 30482952192 | 168011158016 | 163583885312 |

Reserved pinned host cache at startup: ordinary decode 124200480768 bytes, MTP4 118266064896 bytes.
Device allocation and reservation changed during the run.
Pinned host-cache capacity stayed fixed in service snapshots.

## Provenance

Portable artifact name: `oh-my-vllm-nightly/qwen-runtime-regression-v11/http`.
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
| Source | `d2ae273c57559f3a5bb8482c86cc9b3722b0afcc2dd3dfff6a442ae16a0760da` |
| Executable | `e28775b8c0272c7548704ac00c733860e0c04fc88f23ac50c11c8cb5c7fe10aa` |
| Native set | `d07239111105d99ea7de179631a477fadd725e540b83950e401614ccd93001db` |
| Checkpoint | `d26f1d65231264de91c0966777e2c6fa2658f72c278c50f40aead734ce092f4b` |
| Tokenizer | `f78b803474f10eb7a86741417823af5be59cf86f2b57db1869740e18c380bfdd` |
| Identity file | `55ede05b44f8188016d1b326d7abd633d443734cbc57e6c580f191838a281ae0` |
| Validation record | `d5d0ba23adbe5e83c2a1bcf9ff8291f39567b72702bd3b24a6a38c8eda74a53c` |
