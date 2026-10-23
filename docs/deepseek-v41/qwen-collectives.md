# Nightly Qwen runtime regression

## Scope and method

These measurements check Qwen on the branch that adds optional NCCL transport and four-rank control.
Qwen uses one worker and the same seven native modules.
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
| Ordinary | 1 | 32768 | 64 | 1.841 | 17795.8 | 34.8 | 1357.0 | - |
| Ordinary | 1 | 32768 | 8192 | 63.067 | 519.6 | 129.9 | 1343.0 | - |
| Ordinary | 2 | 32768 | 64 | 3.183 | 20586.3 | 40.2 | 2641.0 | - |
| Ordinary | 2 | 32768 | 8192 | 75.481 | 868.3 | 217.1 | 2633.0 | - |
| Ordinary | 4 | 32768 | 64 | 5.843 | 22434.0 | 43.8 | 5245.5 | - |
| Ordinary | 4 | 32768 | 8192 | 85.800 | 1527.6 | 381.9 | 5334.2 | - |
| MTP4 | 1 | 32768 | 64 | 1.596 | 20535.4 | 40.1 | 1356.0 | 41/85 |
| MTP4 | 1 | 32768 | 8192 | 23.732 | 1380.8 | 345.2 | 1346.0 | 6153/8150 |
| MTP4 | 1 | 65536 | 64 | 3.301 | 19851.6 | 19.4 | 3051.0 | 41/85 |
| MTP4 | 1 | 65536 | 8192 | 23.600 | 2776.9 | 347.1 | 3041.0 | 6393/7192 |
| MTP4 | 1 | 131072 | 64 | 7.797 | 16810.7 | 8.2 | 7463.0 | 36/106 |
| MTP4 | 1 | 131072 | 8192 | 28.247 | 4640.2 | 290.0 | 7438.0 | 6522/6673 |
| MTP4 | 2 | 32768 | 64 | 2.933 | 22344.0 | 43.6 | 2657.5 | 88/147 |
| MTP4 | 2 | 32768 | 8192 | 31.856 | 2057.2 | 514.3 | 2694.0 | 12257/16495 |
| MTP4 | 2 | 65536 | 64 | 6.325 | 20723.4 | 20.2 | 6024.5 | 88/148 |
| MTP4 | 2 | 65536 | 8192 | 36.726 | 3569.0 | 446.1 | 6038.5 | 12167/16851 |
| MTP4 | 2 | 131072 | 64 | 15.363 | 17063.3 | 8.3 | 14991.5 | 87/146 |
| MTP4 | 2 | 131072 | 8192 | 47.804 | 5483.7 | 342.7 | 20338.0 | 12985/13588 |
| MTP4 | 4 | 32768 | 64 | 8.377 | 15646.6 | 30.6 | 8032.5 | 183/262 |
| MTP4 | 4 | 32768 | 8192 | 41.615 | 3149.6 | 787.4 | 6619.5 | 25213/30190 |
| MTP4 | 4 | 65536 | 64 | 13.861 | 18912.3 | 18.5 | 13510.0 | 184/260 |
| MTP4 | 4 | 65536 | 8192 | 61.936 | 4232.5 | 529.1 | 17752.5 | 24931/31323 |
| MTP4 | 4 | 131072 | 64 | 32.301 | 16231.4 | 7.9 | 31821.5 | 180/276 |
| MTP4 | 4 | 131072 | 8192 | 90.782 | 5775.2 | 361.0 | 39110.0 | 25019/30973 |

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

Portable artifact name: `qwen-http-collectives-v1`.
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
| Source | `d3c5b15c60ba8c7f68404d34241a084b65684ded5c3ec04cb623c0f5a235f06b` |
| Executable | `e695ae88aa5f92bde890e192ebc988f03fdacfbd6e0f63f49f08ae75ab2920ce` |
| Native set | `0d1085decad2a033591a985f10392bcdc7689fbc70353a533aafe04dbcb19ae6` |
| Checkpoint | `d26f1d65231264de91c0966777e2c6fa2658f72c278c50f40aead734ce092f4b` |
| Tokenizer | `f78b803474f10eb7a86741417823af5be59cf86f2b57db1869740e18c380bfdd` |
| Identity file | `3067226c8c70663fd8095c8f0fb125da50deed2da51ea1b420987cf7823d853e` |
| Validation record | `d7b3b6d6503d067145f892896f1e0f3f2c56681b6711479686572104a3b99167` |
