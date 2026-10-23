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
| Ordinary | 1 | 32768 | 64 | 1.826 | 17945.1 | 35.0 | 1338.0 | - |
| Ordinary | 1 | 32768 | 8192 | 63.833 | 513.3 | 128.3 | 1338.0 | - |
| Ordinary | 2 | 32768 | 64 | 3.229 | 20298.1 | 39.6 | 2656.5 | - |
| Ordinary | 2 | 32768 | 8192 | 74.459 | 880.2 | 220.0 | 2656.5 | - |
| Ordinary | 4 | 32768 | 64 | 5.871 | 22326.0 | 43.6 | 5255.0 | - |
| Ordinary | 4 | 32768 | 8192 | 82.289 | 1592.8 | 398.2 | 5280.8 | - |
| MTP4 | 1 | 32768 | 64 | 1.604 | 20432.0 | 39.9 | 1358.0 | 41/85 |
| MTP4 | 1 | 32768 | 8192 | 23.520 | 1393.2 | 348.3 | 1354.0 | 6153/8150 |
| MTP4 | 1 | 65536 | 64 | 3.285 | 19947.4 | 19.5 | 3037.0 | 41/85 |
| MTP4 | 1 | 65536 | 8192 | 23.700 | 2765.2 | 345.6 | 3030.0 | 6393/7192 |
| MTP4 | 1 | 131072 | 64 | 7.738 | 16939.8 | 8.3 | 7438.0 | 39/94 |
| MTP4 | 1 | 131072 | 8192 | 28.901 | 4535.1 | 283.4 | 7425.0 | 6473/6870 |
| MTP4 | 2 | 32768 | 64 | 2.930 | 22369.8 | 43.7 | 2663.0 | 89/143 |
| MTP4 | 2 | 32768 | 8192 | 33.625 | 1949.0 | 487.3 | 2686.0 | 11998/17532 |
| MTP4 | 2 | 65536 | 64 | 6.372 | 20569.0 | 20.1 | 6057.0 | 89/143 |
| MTP4 | 2 | 65536 | 8192 | 36.188 | 3622.0 | 452.7 | 6055.5 | 12517/15452 |
| MTP4 | 2 | 131072 | 64 | 15.440 | 16978.4 | 8.3 | 14927.0 | 87/150 |
| MTP4 | 2 | 131072 | 8192 | 48.298 | 5427.6 | 339.2 | 15467.0 | 12497/15539 |
| MTP4 | 4 | 32768 | 64 | 5.914 | 22163.0 | 43.3 | 5360.2 | 169/309 |
| MTP4 | 4 | 32768 | 8192 | 37.406 | 3504.1 | 876.0 | 5470.5 | 25594/28667 |
| MTP4 | 4 | 65536 | 64 | 12.364 | 21202.2 | 20.7 | 11942.0 | 176/292 |
| MTP4 | 4 | 65536 | 8192 | 52.109 | 5030.7 | 628.8 | 12026.5 | 24539/32885 |
| MTP4 | 4 | 131072 | 64 | 32.070 | 16348.2 | 8.0 | 31631.5 | 180/279 |
| MTP4 | 4 | 131072 | 8192 | 67.864 | 7725.6 | 482.8 | 30275.5 | 25925/27344 |

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

Portable artifact name: `deepseek-v41-nightly/qwen-runtime-regression-v8/http`.
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
| Source | `bcb7dec2792c868dfcf92b038b8ca00a27f7fba28db8199db6ba8ae197853533` |
| Executable | `6393eedc530377593af5cff6e2b21a67fc549a656fa10f257b2fe68827683b4e` |
| Native set | `d07239111105d99ea7de179631a477fadd725e540b83950e401614ccd93001db` |
| Checkpoint | `d26f1d65231264de91c0966777e2c6fa2658f72c278c50f40aead734ce092f4b` |
| Tokenizer | `f78b803474f10eb7a86741417823af5be59cf86f2b57db1869740e18c380bfdd` |
| Identity file | `9ddd60f14a4bba4519466d55de7bd7596a6f63f2a0a2333b80b9fa8958c0b634` |
| Validation record | `b4d17ce46c4ea384439e9345a4e2ff7a7ef42ef98757040214990d798ea58cdd` |
