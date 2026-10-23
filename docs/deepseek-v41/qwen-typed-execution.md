# Nightly Qwen typed-execution regression

## Scope and method

These measurements check Qwen after addition of the typed four-rank execution interface.
Qwen keeps its single-worker execution path, coordinator, wire fields, and seven native modules.

These measurements do not execute the typed four-rank model path.
They do not measure DeepSeek, four-GPU execution, DSpark, or million-token capacity.
The date is 2026-10-23 in Asia/Shanghai.
The [Qwen procedure](../performance.md) defines the 24 HTTP configurations and the `serving-tools-v1` workload.

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
| Ordinary | 1 | 32768 | 64 | 1.842 | 17785.4 | 34.7 | 1356.0 | - |
| Ordinary | 1 | 32768 | 8192 | 64.102 | 511.2 | 127.8 | 1336.0 | - |
| Ordinary | 2 | 32768 | 64 | 3.246 | 20186.7 | 39.4 | 2657.5 | - |
| Ordinary | 2 | 32768 | 8192 | 75.449 | 868.6 | 217.2 | 2692.0 | - |
| Ordinary | 4 | 32768 | 64 | 5.828 | 22489.6 | 43.9 | 5227.2 | - |
| Ordinary | 4 | 32768 | 8192 | 85.480 | 1533.4 | 383.3 | 5217.0 | - |
| MTP4 | 1 | 32768 | 64 | 1.595 | 20547.9 | 40.1 | 1356.0 | 41/85 |
| MTP4 | 1 | 32768 | 8192 | 23.684 | 1383.5 | 345.9 | 1340.0 | 6153/8150 |
| MTP4 | 1 | 65536 | 64 | 3.295 | 19888.8 | 19.4 | 3046.0 | 41/85 |
| MTP4 | 1 | 65536 | 8192 | 23.659 | 2770.0 | 346.3 | 3047.0 | 6393/7192 |
| MTP4 | 1 | 131072 | 64 | 7.780 | 16848.2 | 8.2 | 7446.0 | 36/106 |
| MTP4 | 1 | 131072 | 8192 | 28.260 | 4638.0 | 289.9 | 7470.0 | 6522/6673 |
| MTP4 | 2 | 32768 | 64 | 2.923 | 22423.8 | 43.8 | 2658.5 | 88/147 |
| MTP4 | 2 | 32768 | 8192 | 31.925 | 2052.8 | 513.2 | 2675.5 | 12257/16495 |
| MTP4 | 2 | 65536 | 64 | 6.327 | 20717.0 | 20.2 | 6025.5 | 88/148 |
| MTP4 | 2 | 65536 | 8192 | 36.811 | 3560.6 | 445.1 | 6016.5 | 12167/16851 |
| MTP4 | 2 | 131072 | 64 | 15.356 | 17071.2 | 8.3 | 14982.0 | 87/146 |
| MTP4 | 2 | 131072 | 8192 | 45.565 | 5753.2 | 359.6 | 17926.5 | 12985/13588 |
| MTP4 | 4 | 32768 | 64 | 8.463 | 15488.1 | 30.3 | 8122.5 | 183/262 |
| MTP4 | 4 | 32768 | 8192 | 43.243 | 3031.1 | 757.8 | 6940.0 | 25213/30190 |
| MTP4 | 4 | 65536 | 64 | 13.993 | 18733.9 | 18.3 | 13561.2 | 184/260 |
| MTP4 | 4 | 65536 | 8192 | 58.043 | 4516.4 | 564.5 | 14759.0 | 24931/31323 |
| MTP4 | 4 | 131072 | 64 | 32.273 | 16245.6 | 7.9 | 31861.2 | 180/276 |
| MTP4 | 4 | 131072 | 8192 | 81.755 | 6413.0 | 400.8 | 36686.8 | 25110/30606 |

## Failures and memory

All 24 configurations completed, with 56 measured requests and no measured request failures.
The benchmark checked full output lengths, zero cached input tokens, tool frames, and runtime identities.
The owned CPU reference job did not execute during measurement.
It continued after the benchmark.

The table gives startup accounting from worker trace identities, in bytes.
Initialized usage is a device-wide observation after worker setup, not a measured per-case peak.
The benchmark did not collect peak GPU memory, pinned-host peaks, or an inter-token latency distribution.
Per-request queue and processing times are kept in the external validation record.

| Mode | FA blocks | GDN blocks | FA bytes | GDN bytes | Planned bytes | Initialized used bytes |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1824 | 128 | 93717528576 | 30482952192 | 166469303808 | 166329057280 |
| MTP4 | 1608 | 128 | 87783112704 | 30482952192 | 168011158016 | 163583885312 |

## Provenance

Portable artifact name: `qwen-http-typed-execution-v1`.
Prompt seed: `nightly-model-adapters-v1`.
The topology is TP1 on one B200. Physical GPU 0 was not used.
The external identity file records the selected device UUID.

Driver: `590.48.01`. Memory clock: 3996 MHz. Power limit: 1000 W.
The run did not change device clocks or power settings.

Request timeout is 7200 seconds. Worker-step timeout is 600 seconds. Shutdown grace is 35 seconds.

The source identity binds paths and bytes in the benchmark source closure, with the uncommitted changes in this branch.
The release executable and all seven native modules are bound independently.
The identity file includes checkpoint shard, tokenizer asset, native source, and native binary identities.
Nested identity digests use UTF-8 JSON with sorted keys and separators `(",", ":")`.
Raw artifacts and service logs are external to the repository.

| Identity | SHA256 |
|---|---|
| Source | `5c4d6c0b1679d6724a098ca42a270cdeb1a07390ab788953bb0a4cb8d6843faa` |
| Executable | `88235fd97aefda79a9bb7b096058f673ec08ce2f30310c0ee7247d399ea81a5c` |
| Native set | `0d1085decad2a033591a985f10392bcdc7689fbc70353a533aafe04dbcb19ae6` |
| Checkpoint | `d26f1d65231264de91c0966777e2c6fa2658f72c278c50f40aead734ce092f4b` |
| Tokenizer | `f78b803474f10eb7a86741417823af5be59cf86f2b57db1869740e18c380bfdd` |
| Identity file | `36ba6ffa4941eb3f294456a41b408e05b19b786a94f1d0c30bb5c38a0e507980` |
| Validation record | `c560b58ed664896437bfd5fa213d811d753dcbd9cf243d378803412d489d9ed9` |
