# Qwen HTTP results

## Scope and method

This report contains the last completed Qwen HTTP matrix.
The source identity below identifies the measured implementation.
Qwen keeps its single-worker execution path, coordinator, wire fields, and seven native modules.

These measurements do not execute the typed four-rank model path.
They do not measure DeepSeek, four-GPU execution, DSpark, or million-token capacity.
The date is 2026-10-24 in Asia/Shanghai.
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
| Ordinary | 1 | 32768 | 64 | 1.936 | 16927.8 | 33.1 | 1392.0 | - |
| Ordinary | 1 | 32768 | 8192 | 64.721 | 506.3 | 126.6 | 1389.0 | - |
| Ordinary | 2 | 32768 | 64 | 3.250 | 20167.5 | 39.4 | 2671.5 | - |
| Ordinary | 2 | 32768 | 8192 | 75.196 | 871.5 | 217.9 | 2662.5 | - |
| Ordinary | 4 | 32768 | 64 | 5.954 | 22013.1 | 43.0 | 5308.0 | - |
| Ordinary | 4 | 32768 | 8192 | 83.503 | 1569.7 | 392.4 | 5297.2 | - |
| MTP4 | 1 | 32768 | 64 | 1.703 | 19239.0 | 37.6 | 1360.0 | 41/85 |
| MTP4 | 1 | 32768 | 8192 | 23.659 | 1385.0 | 346.2 | 1365.0 | 6153/8150 |
| MTP4 | 1 | 65536 | 64 | 3.294 | 19893.6 | 19.4 | 3046.0 | 41/85 |
| MTP4 | 1 | 65536 | 8192 | 23.674 | 2768.3 | 346.0 | 3070.0 | 6393/7192 |
| MTP4 | 1 | 131072 | 64 | 7.736 | 16942.5 | 8.3 | 7440.0 | 39/94 |
| MTP4 | 1 | 131072 | 8192 | 29.058 | 4510.7 | 281.9 | 7485.0 | 6473/6870 |
| MTP4 | 2 | 32768 | 64 | 3.010 | 21770.4 | 42.5 | 2743.0 | 89/143 |
| MTP4 | 2 | 32768 | 8192 | 33.538 | 1954.1 | 488.5 | 2698.0 | 11998/17532 |
| MTP4 | 2 | 65536 | 64 | 8.046 | 16290.2 | 15.9 | 7722.0 | 89/143 |
| MTP4 | 2 | 65536 | 8192 | 36.687 | 3572.8 | 446.6 | 6077.0 | 12517/15452 |
| MTP4 | 2 | 131072 | 64 | 17.941 | 14611.6 | 7.1 | 16221.0 | 87/150 |
| MTP4 | 2 | 131072 | 8192 | 49.301 | 5317.2 | 332.3 | 15084.0 | 12497/15539 |
| MTP4 | 4 | 32768 | 64 | 9.199 | 14248.5 | 27.8 | 8420.0 | 170/311 |
| MTP4 | 4 | 32768 | 8192 | 37.550 | 3490.6 | 872.7 | 5546.5 | 25594/28667 |
| MTP4 | 4 | 65536 | 64 | 12.407 | 21129.5 | 20.6 | 11972.0 | 172/311 |
| MTP4 | 4 | 65536 | 8192 | 55.345 | 4736.6 | 592.1 | 15622.5 | 24702/32239 |
| MTP4 | 4 | 131072 | 64 | 32.628 | 16068.9 | 7.8 | 32190.8 | 180/279 |
| MTP4 | 4 | 131072 | 8192 | 71.093 | 7374.6 | 460.9 | 33550.0 | 25925/27344 |

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
| Ordinary | 1824 | 128 | 93717528576 | 30482952192 | 166469238272 | 166328991744 |
| MTP4 | 1608 | 128 | 87783112704 | 30482952192 | 168011158016 | 163583885312 |

Reserved pinned host cache at startup: ordinary decode 124200480768 bytes, MTP4 118266064896 bytes.
Device allocation and reservation changed during the run.
Pinned host-cache capacity stayed fixed in service snapshots.

## Provenance

Portable artifact name: `deepseek-v41-nightly/qwen-runtime-regression-v4/http`.
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
| Source | `38addec9737816c916878dcef64dfecf2367faedd940d4eac8b9c39578402618` |
| Executable | `6393eedc530377593af5cff6e2b21a67fc549a656fa10f257b2fe68827683b4e` |
| Native set | `d07239111105d99ea7de179631a477fadd725e540b83950e401614ccd93001db` |
| Checkpoint | `d26f1d65231264de91c0966777e2c6fa2658f72c278c50f40aead734ce092f4b` |
| Tokenizer | `f78b803474f10eb7a86741417823af5be59cf86f2b57db1869740e18c380bfdd` |
| Identity file | `191c6790bac30ba9763973206c4943e1c85d414036b3ab256df8bd76d0436b7b` |
| Validation record | `39be55399ce447b8f5483d941b13eeaa5f97157003d052260d19c747488f28af` |
