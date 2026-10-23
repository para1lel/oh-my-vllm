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
| Ordinary | 1 | 32768 | 64 | 1.938 | 16908.6 | 33.0 | 1390.0 | - |
| Ordinary | 1 | 32768 | 8192 | 64.546 | 507.7 | 126.9 | 1396.0 | - |
| Ordinary | 2 | 32768 | 64 | 3.231 | 20283.5 | 39.6 | 2652.5 | - |
| Ordinary | 2 | 32768 | 8192 | 74.740 | 876.9 | 219.2 | 2655.5 | - |
| Ordinary | 4 | 32768 | 64 | 5.904 | 22201.6 | 43.4 | 5269.0 | - |
| Ordinary | 4 | 32768 | 8192 | 82.626 | 1586.3 | 396.6 | 5280.0 | - |
| MTP4 | 1 | 32768 | 64 | 1.609 | 20363.3 | 39.8 | 1370.0 | 41/85 |
| MTP4 | 1 | 32768 | 8192 | 23.810 | 1376.2 | 344.1 | 1360.0 | 6153/8150 |
| MTP4 | 1 | 65536 | 64 | 3.295 | 19891.3 | 19.4 | 3043.0 | 41/85 |
| MTP4 | 1 | 65536 | 8192 | 23.756 | 2758.7 | 344.8 | 3031.0 | 6393/7192 |
| MTP4 | 1 | 131072 | 64 | 7.741 | 16932.7 | 8.3 | 7442.0 | 39/94 |
| MTP4 | 1 | 131072 | 8192 | 28.892 | 4536.6 | 283.5 | 7438.0 | 6473/6870 |
| MTP4 | 2 | 32768 | 64 | 2.939 | 22299.2 | 43.6 | 2672.0 | 89/143 |
| MTP4 | 2 | 32768 | 8192 | 34.059 | 1924.2 | 481.0 | 2869.0 | 11998/17532 |
| MTP4 | 2 | 65536 | 64 | 6.407 | 20457.2 | 20.0 | 6093.0 | 89/143 |
| MTP4 | 2 | 65536 | 8192 | 36.594 | 3581.7 | 447.7 | 6080.5 | 12517/15452 |
| MTP4 | 2 | 131072 | 64 | 15.500 | 16912.4 | 8.3 | 14987.5 | 87/150 |
| MTP4 | 2 | 131072 | 8192 | 48.246 | 5433.5 | 339.6 | 15151.0 | 12497/15539 |
| MTP4 | 4 | 32768 | 64 | 5.928 | 22110.1 | 43.2 | 5403.5 | 170/311 |
| MTP4 | 4 | 32768 | 8192 | 37.976 | 3451.4 | 862.9 | 5557.8 | 25594/28667 |
| MTP4 | 4 | 65536 | 64 | 13.376 | 19598.4 | 19.1 | 12945.0 | 178/287 |
| MTP4 | 4 | 65536 | 8192 | 52.069 | 5034.5 | 629.3 | 12393.5 | 24702/32239 |
| MTP4 | 4 | 131072 | 64 | 29.966 | 17496.2 | 8.5 | 29551.2 | 181/280 |
| MTP4 | 4 | 131072 | 8192 | 69.802 | 7511.0 | 469.4 | 29520.8 | 25862/27591 |

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

Portable artifact name: `deepseek-v41-nightly/qwen-runtime-regression-v6/http`.
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
| Source | `e93babec746b232d2d09f377b27add98b027eb008abfb2b7336f502420cedfaa` |
| Executable | `6393eedc530377593af5cff6e2b21a67fc549a656fa10f257b2fe68827683b4e` |
| Native set | `d07239111105d99ea7de179631a477fadd725e540b83950e401614ccd93001db` |
| Checkpoint | `d26f1d65231264de91c0966777e2c6fa2658f72c278c50f40aead734ce092f4b` |
| Tokenizer | `f78b803474f10eb7a86741417823af5be59cf86f2b57db1869740e18c380bfdd` |
| Identity file | `6949d0b4bfbb9d338ab57a8ad45582d4ae3f650c0d4b017ca433dc039e297efc` |
| Validation record | `58a00ded4f34f41980cd05a8e5decb1edb14eaa7e78df54fce00227b548f96aa` |
