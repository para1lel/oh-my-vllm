# Nightly Qwen lifecycle regression

## Scope and method

These measurements use the Qwen client with the shared process lifecycle module.
The new typed arena and rank supervisor are not connected to the Qwen scheduler.
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
| Ordinary | 1 | 32768 | 64 | 1.821 | 17996.6 | 35.1 | 1355.0 | - |
| Ordinary | 1 | 32768 | 8192 | 62.763 | 522.1 | 130.5 | 1332.0 | - |
| Ordinary | 2 | 32768 | 64 | 3.210 | 20415.8 | 39.9 | 2646.5 | - |
| Ordinary | 2 | 32768 | 8192 | 74.954 | 874.3 | 218.6 | 2631.0 | - |
| Ordinary | 4 | 32768 | 64 | 5.811 | 22555.9 | 44.1 | 5214.5 | - |
| Ordinary | 4 | 32768 | 8192 | 85.148 | 1539.3 | 384.8 | 5199.8 | - |
| MTP4 | 1 | 32768 | 64 | 1.587 | 20643.9 | 40.3 | 1350.0 | 41/85 |
| MTP4 | 1 | 32768 | 8192 | 23.588 | 1389.2 | 347.3 | 1339.0 | 6153/8150 |
| MTP4 | 1 | 65536 | 64 | 3.299 | 19868.0 | 19.4 | 3051.0 | 41/85 |
| MTP4 | 1 | 65536 | 8192 | 23.602 | 2776.7 | 347.1 | 3037.0 | 6393/7192 |
| MTP4 | 1 | 131072 | 64 | 7.796 | 16812.4 | 8.2 | 7464.0 | 36/106 |
| MTP4 | 1 | 131072 | 8192 | 28.251 | 4639.6 | 290.0 | 7449.0 | 6522/6673 |
| MTP4 | 2 | 32768 | 64 | 2.916 | 22473.7 | 43.9 | 2653.5 | 88/147 |
| MTP4 | 2 | 32768 | 8192 | 31.565 | 2076.2 | 519.1 | 2662.0 | 12257/16495 |
| MTP4 | 2 | 65536 | 64 | 7.527 | 17413.6 | 17.0 | 7229.5 | 88/148 |
| MTP4 | 2 | 65536 | 8192 | 36.589 | 3582.3 | 447.8 | 6013.0 | 12167/16851 |
| MTP4 | 2 | 131072 | 64 | 15.912 | 16475.0 | 8.0 | 15514.0 | 87/146 |
| MTP4 | 2 | 131072 | 8192 | 46.259 | 5666.9 | 354.2 | 18844.0 | 12985/13588 |
| MTP4 | 4 | 32768 | 64 | 8.376 | 15647.8 | 30.6 | 8043.5 | 183/262 |
| MTP4 | 4 | 32768 | 8192 | 46.528 | 2817.1 | 704.3 | 11129.2 | 25213/30190 |
| MTP4 | 4 | 65536 | 64 | 14.776 | 17740.9 | 17.3 | 14300.2 | 184/260 |
| MTP4 | 4 | 65536 | 8192 | 57.931 | 4525.1 | 565.6 | 14883.5 | 24931/31323 |
| MTP4 | 4 | 131072 | 64 | 31.417 | 16687.9 | 8.1 | 30994.5 | 180/276 |
| MTP4 | 4 | 131072 | 8192 | 84.511 | 6203.8 | 387.7 | 35037.2 | 25019/30973 |

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

Portable artifact name: `qwen-http-foundations-v1`.
Prompt seed: `nightly-model-adapters-v1`.
The topology is TP1 on one B200. Physical GPU 0 was not used.
The external identity file records the selected device UUID.

Driver: `590.48.01`. Memory clock: 3996 MHz. Power limit: 1000 W.
The run did not change device clocks or power settings.

Request timeout is 7200 seconds. Worker-step timeout is 600 seconds. Shutdown grace is 35 seconds.

The source identity binds paths and bytes in the benchmark source closure, with the uncommitted changes in this branch.
The release executable and all seven native modules are bound separately.
The identity file includes checkpoint shard, tokenizer asset, native source, and native binary identities.
Nested identity digests use UTF-8 JSON with sorted keys and separators `(",", ":")`.
Raw artifacts and service logs are external to the repository.

| Identity | SHA256 |
|---|---|
| Source | `5eca7ef9eb84510ede83a3bd6e0dd2c20f517ec6106445efd2dba1245d7349ae` |
| Executable | `fe495a7ff976f39ce1d910f0997abbc6697a2899d04503770261a4b7ebe95797` |
| Native set | `0d1085decad2a033591a985f10392bcdc7689fbc70353a533aafe04dbcb19ae6` |
| Checkpoint | `d26f1d65231264de91c0966777e2c6fa2658f72c278c50f40aead734ce092f4b` |
| Tokenizer | `f78b803474f10eb7a86741417823af5be59cf86f2b57db1869740e18c380bfdd` |
| Identity file | `f7f6cf30e1296fce538e0ad48acb8f548ec98d55ec0ba61506d42c80a4316199` |
| Validation record | `4b88fca0d981651f61ae3aace1aaaa19d9680c0c45f34e10cc5db020f4bd6389` |
