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
| Ordinary | 1 | 32768 | 64 | 1.913 | 17132.4 | 33.5 | 1396.0 | - |
| Ordinary | 1 | 32768 | 8192 | 64.979 | 504.3 | 126.1 | 1398.0 | - |
| Ordinary | 2 | 32768 | 64 | 3.253 | 20146.7 | 39.3 | 2661.5 | - |
| Ordinary | 2 | 32768 | 8192 | 75.500 | 868.0 | 217.0 | 2655.5 | - |
| Ordinary | 4 | 32768 | 64 | 6.018 | 21780.8 | 42.5 | 5331.5 | - |
| Ordinary | 4 | 32768 | 8192 | 84.633 | 1548.7 | 387.2 | 5338.0 | - |
| MTP4 | 1 | 32768 | 64 | 1.653 | 19821.3 | 38.7 | 1395.0 | 41/85 |
| MTP4 | 1 | 32768 | 8192 | 24.769 | 1323.0 | 330.7 | 1380.0 | 6153/8150 |
| MTP4 | 1 | 65536 | 64 | 3.366 | 19470.6 | 19.0 | 3106.0 | 41/85 |
| MTP4 | 1 | 65536 | 8192 | 24.418 | 2684.0 | 335.5 | 3095.0 | 6393/7192 |
| MTP4 | 1 | 131072 | 64 | 7.775 | 16857.3 | 8.2 | 7478.0 | 39/94 |
| MTP4 | 1 | 131072 | 8192 | 29.113 | 4502.1 | 281.4 | 7492.0 | 6473/6870 |
| MTP4 | 2 | 32768 | 64 | 2.961 | 22133.7 | 43.2 | 2684.0 | 89/143 |
| MTP4 | 2 | 32768 | 8192 | 33.815 | 1938.0 | 484.5 | 2762.0 | 11998/17532 |
| MTP4 | 2 | 65536 | 64 | 6.483 | 20218.8 | 19.7 | 6084.0 | 89/143 |
| MTP4 | 2 | 65536 | 8192 | 36.467 | 3594.3 | 449.3 | 6066.0 | 12517/15452 |
| MTP4 | 2 | 131072 | 64 | 15.622 | 16780.6 | 8.2 | 15096.0 | 87/150 |
| MTP4 | 2 | 131072 | 8192 | 48.328 | 5424.3 | 339.0 | 15061.5 | 12497/15539 |
| MTP4 | 4 | 32768 | 64 | 6.089 | 21527.6 | 42.0 | 5587.5 | 170/311 |
| MTP4 | 4 | 32768 | 8192 | 37.586 | 3487.3 | 871.8 | 5440.2 | 25594/28667 |
| MTP4 | 4 | 65536 | 64 | 16.174 | 16208.2 | 15.8 | 15744.2 | 178/287 |
| MTP4 | 4 | 65536 | 8192 | 49.476 | 5298.5 | 662.3 | 12011.5 | 25020/30968 |
| MTP4 | 4 | 131072 | 64 | 30.106 | 17414.9 | 8.5 | 29639.5 | 180/280 |
| MTP4 | 4 | 131072 | 8192 | 67.518 | 7765.2 | 485.3 | 29825.8 | 25925/27344 |

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

Portable artifact name: `deepseek-v41-nightly/qwen-runtime-regression-v3/http`.
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
| Source | `be7a08d9e376b9789084cc53171f72bde8e8c9a2960df00d82911434b4d66f52` |
| Executable | `6393eedc530377593af5cff6e2b21a67fc549a656fa10f257b2fe68827683b4e` |
| Native set | `d07239111105d99ea7de179631a477fadd725e540b83950e401614ccd93001db` |
| Checkpoint | `d26f1d65231264de91c0966777e2c6fa2658f72c278c50f40aead734ce092f4b` |
| Tokenizer | `f78b803474f10eb7a86741417823af5be59cf86f2b57db1869740e18c380bfdd` |
| Identity file | `098c83b7dbc01a7181f5d07acaec0586bb9f56f39df196827dbee45e366423a3` |
| Validation record | `1ff6460bec092ad52140225085d6b9f31eaeedd9b41e93f8817ecfeef9f511e8` |
