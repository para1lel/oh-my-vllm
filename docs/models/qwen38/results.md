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
| Ordinary | 1 | 32768 | 64 | 1.834 | 0.697 | 38.01% | - |
| Ordinary | 1 | 32768 | 8192 | 63.756 | 32.001 | 50.19% | - |
| Ordinary | 2 | 32768 | 64 | 3.238 | 1.178 | 36.36% | - |
| Ordinary | 2 | 32768 | 8192 | 75.163 | 35.359 | 47.04% | - |
| Ordinary | 4 | 32768 | 64 | 5.889 | 2.131 | 36.18% | - |
| Ordinary | 4 | 32768 | 8192 | 82.645 | 42.067 | 50.90% | - |
| MTP4 | 1 | 32768 | 64 | 1.589 | 0.577 | 36.33% | 41/85 |
| MTP4 | 1 | 32768 | 8192 | 23.628 | 11.665 | 49.37% | 6153/8150 |
| MTP4 | 1 | 65536 | 64 | 3.298 | 1.231 | 37.32% | 41/85 |
| MTP4 | 1 | 65536 | 8192 | 23.609 | 11.615 | 49.20% | 6393/7192 |
| MTP4 | 1 | 131072 | 64 | 7.743 | 3.114 | 40.21% | 39/94 |
| MTP4 | 1 | 131072 | 8192 | 28.848 | 14.200 | 49.22% | 6473/6870 |
| MTP4 | 2 | 32768 | 64 | 2.932 | 1.033 | 35.24% | 89/143 |
| MTP4 | 2 | 32768 | 8192 | 33.381 | 14.566 | 43.64% | 11998/17532 |
| MTP4 | 2 | 65536 | 64 | 6.357 | 2.352 | 37.00% | 89/143 |
| MTP4 | 2 | 65536 | 8192 | 36.079 | 16.294 | 45.16% | 12517/15452 |
| MTP4 | 2 | 131072 | 64 | 15.473 | 6.099 | 39.42% | 87/150 |
| MTP4 | 2 | 131072 | 8192 | 47.733 | 22.334 | 46.79% | 12497/15539 |
| MTP4 | 4 | 32768 | 64 | 5.821 | 1.992 | 34.22% | 170/311 |
| MTP4 | 4 | 32768 | 8192 | 39.229 | 15.835 | 40.37% | 25062/30800 |
| MTP4 | 4 | 65536 | 64 | 12.341 | 4.581 | 37.12% | 174/302 |
| MTP4 | 4 | 65536 | 8192 | 51.383 | 22.217 | 43.24% | 24539/32885 |
| MTP4 | 4 | 131072 | 64 | 30.005 | 12.036 | 40.11% | 182/272 |
| MTP4 | 4 | 131072 | 8192 | 72.561 | 31.156 | 42.94% | 25862/27591 |

## Failures and memory

All 24 configurations completed, with 56 measured requests and no measured request failures.
The benchmark checked full output lengths, zero cached input tokens, tool frames, and runtime identities.

The table gives startup accounting from worker trace identities, in bytes.
Initialized usage is a device-wide observation after worker setup, not a measured per-case peak.
The benchmark did not collect peak GPU memory, pinned-host peaks, or an inter-token latency distribution.
Per-request queue and processing times are kept in the external validation record.

| Mode | FA blocks | GDN blocks | FA bytes | GDN bytes | Planned bytes | Initialized used bytes |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1824 | 128 | 93717528576 | 30482952192 | 166601424384 | 166461177856 |
| MTP4 | 1608 | 128 | 87783112704 | 30482952192 | 168143278592 | 163716005888 |

Reserved pinned host cache at startup: ordinary decode 124200480768 bytes, MTP4 118266064896 bytes.
Device allocation and reservation changed during the run.
Pinned host-cache capacity stayed fixed in service snapshots.

## Provenance

Portable artifact name: `deepseek-v41-nightly/qwen-runtime-regression-v9/http`.
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
| Source | `6be486b5f23357bdf40f112f4f06828b84a910e36c641095ad396bfaa7a165e1` |
| Executable | `6393eedc530377593af5cff6e2b21a67fc549a656fa10f257b2fe68827683b4e` |
| Native set | `d07239111105d99ea7de179631a477fadd725e540b83950e401614ccd93001db` |
| Checkpoint | `d26f1d65231264de91c0966777e2c6fa2658f72c278c50f40aead734ce092f4b` |
| Tokenizer | `f78b803474f10eb7a86741417823af5be59cf86f2b57db1869740e18c380bfdd` |
| Identity file | `04da9de415cadde42326d03378c226a958519e52e1f12bdb395818be6f5271ad` |
| Validation record | `cf5ddfc1936bc444face67ba9138e5478315295adca3f77a7215cce6de00f9e0` |
