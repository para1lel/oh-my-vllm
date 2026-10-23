# Qwen HTTP results

## Scope and method

This report contains the last completed Qwen HTTP matrix.
The source identity below identifies the measured implementation.
Qwen keeps its single-worker execution path, coordinator, wire fields, and seven native modules.

These measurements do not execute the typed four-rank model path.
They do not measure DeepSeek, four-GPU execution, DSpark, or million-token capacity.
The date is 2026-10-23 in Asia/Shanghai.
The [Qwen procedure](../performance.md) gives the 24 HTTP configurations and the `serving-tools-v1` workload.

Each row uses one B200, concurrent HTTP requests, and the specified uncached input/output lengths for each request.
The service has a limit of four active requests, a 32768-token step budget, and a 262144-token context limit.
Sampling uses temperature 1.0, top-k 20, top-p 0.95, presence penalty 0.5, the repetition guard, and client-ordinal seeds.
Reasoning is off. Tool choice is auto.
Required-tool checks and short warmup requests occur before the measured cases.

HTTP seconds include request-thread creation, submission, preparation, queueing, model execution, and full JSON response reads.
The timer does not include initialization or trace analysis.
Input/s and output/s are equal to the batch token totals divided by the same full HTTP seconds.
TTFT is the mean server first-token time in milliseconds, not client streaming latency.
Accepted/proposed counts come from measured requests and agree between service logs and execution traces.
Ordinary decode has no draft acceptance ratio.

## HTTP measurements

| Mode | Batch | Input | Output | HTTP s | Input/s | Output/s | TTFT ms | Accepted/proposed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.898 | 17264.7 | 33.7 | 1388.0 | - |
| Ordinary | 1 | 32768 | 8192 | 64.512 | 507.9 | 127.0 | 1386.0 | - |
| Ordinary | 2 | 32768 | 64 | 3.229 | 20296.1 | 39.6 | 2656.0 | - |
| Ordinary | 2 | 32768 | 8192 | 75.100 | 872.7 | 218.2 | 2661.0 | - |
| Ordinary | 4 | 32768 | 64 | 5.923 | 22130.2 | 43.2 | 5272.8 | - |
| Ordinary | 4 | 32768 | 8192 | 83.039 | 1578.4 | 394.6 | 5303.0 | - |
| MTP4 | 1 | 32768 | 64 | 1.602 | 20451.6 | 39.9 | 1364.0 | 41/85 |
| MTP4 | 1 | 32768 | 8192 | 23.820 | 1375.7 | 343.9 | 1366.0 | 6153/8150 |
| MTP4 | 1 | 65536 | 64 | 3.286 | 19942.5 | 19.5 | 3036.0 | 41/85 |
| MTP4 | 1 | 65536 | 8192 | 23.718 | 2763.1 | 345.4 | 3045.0 | 6393/7192 |
| MTP4 | 1 | 131072 | 64 | 7.743 | 16927.2 | 8.3 | 7444.0 | 39/94 |
| MTP4 | 1 | 131072 | 8192 | 28.940 | 4529.1 | 283.1 | 7438.0 | 6473/6870 |
| MTP4 | 2 | 32768 | 64 | 2.931 | 22362.6 | 43.7 | 2669.0 | 89/143 |
| MTP4 | 2 | 32768 | 8192 | 33.914 | 1932.4 | 483.1 | 2713.5 | 11998/17532 |
| MTP4 | 2 | 65536 | 64 | 6.537 | 20052.1 | 19.6 | 6130.0 | 89/143 |
| MTP4 | 2 | 65536 | 8192 | 38.623 | 3393.6 | 424.2 | 8310.5 | 12517/15452 |
| MTP4 | 2 | 131072 | 64 | 15.608 | 16795.3 | 8.2 | 15097.0 | 87/150 |
| MTP4 | 2 | 131072 | 8192 | 47.987 | 5462.8 | 341.4 | 15035.0 | 12497/15539 |
| MTP4 | 4 | 32768 | 64 | 6.432 | 20379.6 | 39.8 | 5683.8 | 169/321 |
| MTP4 | 4 | 32768 | 8192 | 37.319 | 3512.2 | 878.1 | 5463.5 | 25594/28667 |
| MTP4 | 4 | 65536 | 64 | 12.518 | 20941.1 | 20.5 | 12024.5 | 176/292 |
| MTP4 | 4 | 65536 | 8192 | 51.749 | 5065.6 | 633.2 | 12010.8 | 24539/32885 |
| MTP4 | 4 | 131072 | 64 | 30.064 | 17438.8 | 8.5 | 29627.8 | 180/279 |
| MTP4 | 4 | 131072 | 8192 | 67.683 | 7746.2 | 484.1 | 29641.0 | 25862/27591 |

## Prefill

The 32768-token budget applies to the full step, not each request.
Twice as many requests do not double the projection row count per step.
The scheduler and worker permit at most one partial prefill request per batch, with 784-token blocks.

FA uses the ring kernel when a batch has at least 512 query tokens in one request.
Kernel selection changed in 45 of 109 measured shapes. All 45 cases had lower `median_ms` values.
The table uses full-FA `median_ms` values for 784 query tokens and 262144 context tokens per request.
Each case uses 30 samples, with at least 20 ms of GPU work per sample.
The baseline is the original FA implementation at `d24fef8`.

| Batch | Before, ms | After, ms |
|---|---:|---:|
| 1 | 4.9425 | 4.9870 |
| 2 | 12.2000 | 9.9796 |
| 4 | 24.5150 | 19.9376 |

Artifacts `fa-extended-baseline.json` and `fa-extended-candidate.json` bind the native identities and all measured shapes.
They are in the external `qwen-prefill-scaling-20261023` artifact set.
Numerical and operator latency limits did not change.

The host KV pool reserves pinned capacity before Ready.
This moves pinned host-cache allocation from request processing to startup.
Prefix storage and required request offload stay enabled.
Transfers can still use cache capacity and memory bandwidth.
The [host cache guide](../kv-offload.md) gives the startup memory and failure conditions.

## Failures and memory

All 24 configurations completed, with 56 measured requests and no measured request failures.
The benchmark checked full output lengths, zero cached input tokens, tool frames, and runtime identities.

The table gives startup accounting from worker trace identities, in bytes.
Initialized usage is a device-wide observation after worker setup, not a measured per-case peak.
The benchmark did not collect peak GPU memory, pinned-host peaks, or an inter-token latency distribution.
Per-request queue and processing times are kept in the external validation record.

| Mode | FA blocks | GDN blocks | FA bytes | GDN bytes | Planned bytes | Initialized used bytes |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1824 | 128 | 93717528576 | 30482952192 | 166469303808 | 166329057280 |
| MTP4 | 1608 | 128 | 87783112704 | 30482952192 | 168011158016 | 163583885312 |

Reserved pinned host cache at startup: ordinary decode 124200480768 bytes, MTP4 118266064896 bytes.
Device allocation and reservation changed during the run.
Pinned host-cache capacity stayed fixed in service snapshots.

## Provenance

Portable artifact name: `deepseek-v41-nightly/qwen-runtime-regression-v2/http`.
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
| Source | `d5ac69fedb0e014ecf1b2414125eb501ea8dcbe56627833bc37d113c5e40aeae` |
| Executable | `6393eedc530377593af5cff6e2b21a67fc549a656fa10f257b2fe68827683b4e` |
| Native set | `d07239111105d99ea7de179631a477fadd725e540b83950e401614ccd93001db` |
| Checkpoint | `d26f1d65231264de91c0966777e2c6fa2658f72c278c50f40aead734ce092f4b` |
| Tokenizer | `f78b803474f10eb7a86741417823af5be59cf86f2b57db1869740e18c380bfdd` |
| Identity file | `588c8aa4e9afe6ebdb54600727380c388ebd115c8a58679ad933ca378b81b8a9` |
| Validation record | `90a25323f4bd730bd067caa8440b9f5b0b4b358689aca3ee34d51745c0f7d560` |
