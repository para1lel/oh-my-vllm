# Nightly Qwen cache policy regression

## Scope and method

These measurements check Qwen after extraction of its model cache policy.
Shared scheduling uses model-selected state assignments and transfer owners.
Qwen keeps its coordinator, wire fields, and seven native modules.

The new typed arena and rank supervisor are not connected to the Qwen scheduler.
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
| Ordinary | 1 | 32768 | 64 | 1.845 | 17762.6 | 34.7 | 1357.0 | - |
| Ordinary | 1 | 32768 | 8192 | 63.614 | 515.1 | 128.8 | 1333.0 | - |
| Ordinary | 2 | 32768 | 64 | 3.198 | 20491.2 | 40.0 | 2639.5 | - |
| Ordinary | 2 | 32768 | 8192 | 75.190 | 871.6 | 217.9 | 2635.5 | - |
| Ordinary | 4 | 32768 | 64 | 5.859 | 22372.4 | 43.7 | 5251.8 | - |
| Ordinary | 4 | 32768 | 8192 | 85.610 | 1531.0 | 382.8 | 5228.2 | - |
| MTP4 | 1 | 32768 | 64 | 1.604 | 20424.6 | 39.9 | 1365.0 | 41/85 |
| MTP4 | 1 | 32768 | 8192 | 23.760 | 1379.1 | 344.8 | 1347.0 | 6153/8150 |
| MTP4 | 1 | 65536 | 64 | 3.279 | 19984.7 | 19.5 | 3032.0 | 41/85 |
| MTP4 | 1 | 65536 | 8192 | 23.642 | 2772.0 | 346.5 | 3040.0 | 6393/7192 |
| MTP4 | 1 | 131072 | 64 | 7.788 | 16830.5 | 8.2 | 7454.0 | 36/106 |
| MTP4 | 1 | 131072 | 8192 | 28.273 | 4636.0 | 289.7 | 7462.0 | 6522/6673 |
| MTP4 | 2 | 32768 | 64 | 2.914 | 22491.7 | 43.9 | 2648.0 | 88/147 |
| MTP4 | 2 | 32768 | 8192 | 32.107 | 2041.1 | 510.3 | 2675.5 | 12257/16495 |
| MTP4 | 2 | 65536 | 64 | 6.359 | 20610.8 | 20.1 | 6052.5 | 88/148 |
| MTP4 | 2 | 65536 | 8192 | 36.818 | 3560.0 | 445.0 | 6036.0 | 12167/16851 |
| MTP4 | 2 | 131072 | 64 | 15.356 | 17071.3 | 8.3 | 15008.0 | 84/162 |
| MTP4 | 2 | 131072 | 8192 | 45.131 | 5808.5 | 363.0 | 17634.0 | 12985/13588 |
| MTP4 | 4 | 32768 | 64 | 8.374 | 15652.6 | 30.6 | 8014.5 | 180/273 |
| MTP4 | 4 | 32768 | 8192 | 42.845 | 3059.2 | 764.8 | 6806.8 | 25213/30190 |
| MTP4 | 4 | 65536 | 64 | 15.063 | 17402.9 | 17.0 | 14722.8 | 184/260 |
| MTP4 | 4 | 65536 | 8192 | 58.428 | 4486.6 | 560.8 | 14490.5 | 24931/31323 |
| MTP4 | 4 | 131072 | 64 | 31.644 | 16568.3 | 8.1 | 31238.0 | 180/276 |
| MTP4 | 4 | 131072 | 8192 | 86.169 | 6084.4 | 380.3 | 42101.5 | 25608/28618 |

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
| Ordinary | 1824 | 128 | 93717528576 | 30482952192 | 166469238272 | 166328991744 |
| MTP4 | 1608 | 128 | 87783112704 | 30482952192 | 168011158016 | 163583885312 |

## Provenance

Portable artifact name: `qwen-http-cache-policy-v1`.
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
| Source | `5565d1c6afa5869b7b69e15e61b41d0815b13b5d4ebc76a16c9537d88d76808b` |
| Executable | `27ea0f015c0cc3e37ee87b673c387eddbada53f93a2e23f6bde06188b83232a9` |
| Native set | `0d1085decad2a033591a985f10392bcdc7689fbc70353a533aafe04dbcb19ae6` |
| Checkpoint | `d26f1d65231264de91c0966777e2c6fa2658f72c278c50f40aead734ce092f4b` |
| Tokenizer | `f78b803474f10eb7a86741417823af5be59cf86f2b57db1869740e18c380bfdd` |
| Identity file | `1b74541d55d396c96707ed86b3e40e1be51fd43a286990cfe89d53c17d875873` |
| Validation record | `6732ba38e5d100d71e1a05f971391f7a3653b7d46632d2b2fc3c048035c28921` |
