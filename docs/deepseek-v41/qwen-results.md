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
| Ordinary | 1 | 32768 | 64 | 1.916 | 17101.4 | 33.4 | 1411.0 | - |
| Ordinary | 1 | 32768 | 8192 | 65.235 | 502.3 | 125.6 | 1379.0 | - |
| Ordinary | 2 | 32768 | 64 | 3.221 | 20349.6 | 39.7 | 2650.5 | - |
| Ordinary | 2 | 32768 | 8192 | 74.722 | 877.1 | 219.3 | 2651.5 | - |
| Ordinary | 4 | 32768 | 64 | 5.908 | 22184.5 | 43.3 | 5273.0 | - |
| Ordinary | 4 | 32768 | 8192 | 83.008 | 1579.0 | 394.8 | 5259.2 | - |
| MTP4 | 1 | 32768 | 64 | 1.602 | 20460.8 | 40.0 | 1354.0 | 41/85 |
| MTP4 | 1 | 32768 | 8192 | 23.773 | 1378.4 | 344.6 | 1357.0 | 6153/8150 |
| MTP4 | 1 | 65536 | 64 | 3.279 | 19984.3 | 19.5 | 3032.0 | 41/85 |
| MTP4 | 1 | 65536 | 8192 | 23.645 | 2771.7 | 346.5 | 3031.0 | 6393/7192 |
| MTP4 | 1 | 131072 | 64 | 7.750 | 16913.6 | 8.3 | 7452.0 | 39/94 |
| MTP4 | 1 | 131072 | 8192 | 28.767 | 4556.3 | 284.8 | 7439.0 | 6473/6870 |
| MTP4 | 2 | 32768 | 64 | 2.937 | 22313.9 | 43.6 | 2670.5 | 89/143 |
| MTP4 | 2 | 32768 | 8192 | 33.642 | 1948.0 | 487.0 | 2686.0 | 11998/17532 |
| MTP4 | 2 | 65536 | 64 | 6.378 | 20550.3 | 20.1 | 6059.5 | 89/143 |
| MTP4 | 2 | 65536 | 8192 | 38.120 | 3438.4 | 429.8 | 6051.5 | 12200/16722 |
| MTP4 | 2 | 131072 | 64 | 15.470 | 16945.1 | 8.3 | 15058.0 | 90/139 |
| MTP4 | 2 | 131072 | 8192 | 49.835 | 5260.2 | 328.8 | 16945.0 | 12497/15539 |
| MTP4 | 4 | 32768 | 64 | 5.849 | 22409.3 | 43.8 | 5345.2 | 166/327 |
| MTP4 | 4 | 32768 | 8192 | 41.327 | 3171.6 | 792.9 | 7672.5 | 25062/30800 |
| MTP4 | 4 | 65536 | 64 | 12.346 | 21232.4 | 20.7 | 11958.8 | 173/308 |
| MTP4 | 4 | 65536 | 8192 | 52.567 | 4986.9 | 623.4 | 12983.5 | 24702/32239 |
| MTP4 | 4 | 131072 | 64 | 30.000 | 17476.0 | 8.5 | 29563.2 | 180/279 |
| MTP4 | 4 | 131072 | 8192 | 69.008 | 7597.5 | 474.8 | 29529.8 | 25862/27591 |

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
Allocated capacity in service snapshots stayed constant across all cases.

## Provenance

Portable artifact name: `qwen-prefill-scaling-20261023/http-v2`.
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
| Source | `35dc4b243bbd2448c1db86f12147b8a7499c3c22a8d08963e7eb5287e09b28e9` |
| Executable | `6393eedc530377593af5cff6e2b21a67fc549a656fa10f257b2fe68827683b4e` |
| Native set | `d07239111105d99ea7de179631a477fadd725e540b83950e401614ccd93001db` |
| Checkpoint | `d26f1d65231264de91c0966777e2c6fa2658f72c278c50f40aead734ce092f4b` |
| Tokenizer | `f78b803474f10eb7a86741417823af5be59cf86f2b57db1869740e18c380bfdd` |
| Identity file | `7ffb37c639a2d8545ac18b762905fd22e30fa069a22ddc3786c3211747f1a690` |
| Validation record | `584179628d57e084a9a10e9dfb28f99a5a0d392ec8f16028bd0a80b68a963ee6` |
