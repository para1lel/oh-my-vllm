# Nightly Qwen adapter regression

## Scope and method

These measurements use the Qwen model adapter after Python family extraction, Rust output selection, and Rust policy extraction.
They do not measure DeepSeek, four-GPU execution, DSpark, or million-token capacity.
The date is 2026-10-23 in Asia/Shanghai.
The [Qwen procedure](../performance.md) defines the 24 HTTP configurations and the `serving-tools-v1` workload.

Each row uses one B200, concurrent HTTP requests, and exact uncached input/output lengths per request.
The service has a limit of four active requests, a 32768-token step budget, and a 262144-token context limit.
Sampling uses temperature 1.0, top-k 20, top-p 0.95, presence penalty 0.5, the repetition guard, and client-ordinal seeds.
Reasoning is off. Tool choice is auto.
Required-tool checks and short warmup requests occur before the measured cases.

HTTP seconds include request-thread creation, submission, preparation, queueing, model execution, and full JSON response reads.
The timer does not include initialization or trace analysis.
Input/s and output/s equal the batch token totals divided by the same full HTTP seconds.
TTFT is the mean server first-token time in milliseconds, not client streaming latency.
Accepted/proposed counts come from measured requests and agree between service logs and execution traces.
Ordinary decode has no draft acceptance ratio.

## HTTP measurements

| Mode | Batch | Input | Output | HTTP s | Input/s | Output/s | TTFT ms | Accepted/proposed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordinary | 1 | 32768 | 64 | 1.993 | 16438.4 | 32.1 | 1511.0 | - |
| Ordinary | 1 | 32768 | 8192 | 63.417 | 516.7 | 129.2 | 1342.0 | - |
| Ordinary | 2 | 32768 | 64 | 3.259 | 20108.8 | 39.3 | 2631.5 | - |
| Ordinary | 2 | 32768 | 8192 | 75.372 | 869.5 | 217.4 | 2628.0 | - |
| Ordinary | 4 | 32768 | 64 | 5.800 | 22600.2 | 44.1 | 5199.2 | - |
| Ordinary | 4 | 32768 | 8192 | 85.915 | 1525.6 | 381.4 | 5195.5 | - |
| MTP4 | 1 | 32768 | 64 | 1.578 | 20761.6 | 40.5 | 1347.0 | 41/85 |
| MTP4 | 1 | 32768 | 8192 | 23.574 | 1390.0 | 347.5 | 1335.0 | 6153/8150 |
| MTP4 | 1 | 65536 | 64 | 3.270 | 20043.9 | 19.6 | 3022.0 | 41/85 |
| MTP4 | 1 | 65536 | 8192 | 23.623 | 2774.2 | 346.8 | 3015.0 | 6393/7192 |
| MTP4 | 1 | 131072 | 64 | 7.753 | 16905.8 | 8.3 | 7419.0 | 36/106 |
| MTP4 | 1 | 131072 | 8192 | 28.256 | 4638.7 | 289.9 | 7402.0 | 6522/6673 |
| MTP4 | 2 | 32768 | 64 | 2.905 | 22559.0 | 44.1 | 2637.5 | 88/147 |
| MTP4 | 2 | 32768 | 8192 | 31.852 | 2057.5 | 514.4 | 2655.5 | 12257/16495 |
| MTP4 | 2 | 65536 | 64 | 6.329 | 20710.9 | 20.2 | 6002.0 | 88/148 |
| MTP4 | 2 | 65536 | 8192 | 36.712 | 3570.2 | 446.3 | 5999.5 | 12167/16851 |
| MTP4 | 2 | 131072 | 64 | 15.238 | 17203.7 | 8.4 | 14870.0 | 87/146 |
| MTP4 | 2 | 131072 | 8192 | 45.050 | 5818.9 | 363.7 | 17521.0 | 12985/13588 |
| MTP4 | 4 | 32768 | 64 | 8.787 | 14916.6 | 29.1 | 8444.5 | 183/262 |
| MTP4 | 4 | 32768 | 8192 | 42.440 | 3088.4 | 772.1 | 7383.0 | 25213/30190 |
| MTP4 | 4 | 65536 | 64 | 13.731 | 19091.3 | 18.6 | 13380.8 | 184/260 |
| MTP4 | 4 | 65536 | 8192 | 54.783 | 4785.1 | 598.1 | 14595.2 | 25323/29753 |
| MTP4 | 4 | 131072 | 64 | 32.556 | 16104.4 | 7.9 | 32131.5 | 180/276 |
| MTP4 | 4 | 131072 | 8192 | 90.221 | 5811.2 | 363.2 | 35719.5 | 25019/30973 |

## Failures and memory

All 24 configurations completed, with 56 measured requests and no measured request failures.
The benchmark checked full output lengths, zero cached input tokens, tool frames, and runtime identities.
An earlier attempt stopped before measurement because the device did not have sufficient free memory at startup.
That attempt supplies no performance samples.
The completed run used a different idle device with the same configuration.

The table gives startup accounting from worker trace identities, in bytes.
Initialized usage is a device-wide observation after worker setup, not a measured per-case peak.
The benchmark did not collect peak GPU memory, pinned-host peaks, or an inter-token latency distribution.
Per-request queue and processing times are kept in the external validation record.

| Mode | FA blocks | GDN blocks | FA bytes | GDN bytes | Planned bytes | Initialized used bytes |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 1824 | 128 | 93717528576 | 30482952192 | 166469303808 | 166329057280 |
| MTP4 | 1608 | 128 | 87783112704 | 30482952192 | 168085148160 | 163657875456 |

## Provenance

Portable artifact name: `qwen-http-adapters-v2`.
Prompt seed: `nightly-model-adapters-v1`.
The topology is TP1 on one B200. Physical GPU 0 was not used.
The external identity file records the selected device UUID.

Driver: `590.48.01`. Memory clock: 3996 MHz. Power limit: 1000 W.
The run did not change device clocks or power settings.

Request timeout is 7200 seconds. Worker-step timeout is 600 seconds. Shutdown grace is 35 seconds.

The source identity binds paths and bytes in the benchmark source closure, with uncommitted policy files.
The release executable and all seven native modules are bound separately.
The identity file includes checkpoint shard, tokenizer asset, native source, and native binary identities.
Nested identity digests use UTF-8 JSON with sorted keys and separators `(",", ":")`.
Raw artifacts and service logs are external to the repository.

| Identity | SHA256 |
|---|---|
| Source | `b65d1eb846b924f6bc5b334f518144eb95df584db50a72835f3c5dd584490945` |
| Executable | `0728f6eeee7bebb5a014830827b2dd78313fb37b6d46ea76feb28a0204f2b87f` |
| Native set | `0d1085decad2a033591a985f10392bcdc7689fbc70353a533aafe04dbcb19ae6` |
| Checkpoint | `d26f1d65231264de91c0966777e2c6fa2658f72c278c50f40aead734ce092f4b` |
| Tokenizer | `f78b803474f10eb7a86741417823af5be59cf86f2b57db1869740e18c380bfdd` |
| Identity file | `f918d1ed9215ee65b6cd5b9a11c531a24ee24fcb7000cc4aa7391fa7e8ce712c` |
| Validation record | `18c6880e160ec17a605dce62bc25783cc939a49c94ad052ad1b3c0674333243e` |
