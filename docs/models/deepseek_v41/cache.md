# Physical cache reference

The [cache implementation](../../../python/oh_my_vllm/models/deepseek_v41/cache.py) maps Rust logical blocks to model storage.
The [packed storage](../../../python/oh_my_vllm/models/deepseek_v41/packed_cache.py) keeps encoded history and its scales in separate planes.
The [quantization contract](quantization.md) defines their values and rounding.

## Current storage

| State | Format | Logical bytes per row |
|---|---|---:|
| Target and MTP windows | 512 BF16 values | 1024 |
| Compressed KV | 512 E2M1 values, one E4M3 scale per 16 values | 288 |
| Index keys | 128 E2M1 values, one UE8M0 scale per 32 values | 68 |

A row is one cache vector at one physical source.
It is not one original token across all layers.
E2M1 codes store two values per byte.
E4M3 and UE8M0 scales have different numerical meanings.
Compressed KV includes all 512 dimensions, with no BF16 RoPE tail.

The [source maps](../../../python/oh_my_vllm/models/deepseek_v41/adapter.py) select KV sources at layers 2, 8, 14, and 20.
Their compression ratios are 2, 2, 2, and 1.
Eight index-source layers share four physical key sources.
Consumers do not allocate another full history.

Active request slots use page-table views of compressed KV, index keys, and Engram history.
CUDA storage maps physical pages on demand while tensor addresses stay fixed.
CPU execution uses dense storage.
Bound storage rejects tensor device or dtype conversion.
Graph leases prevent page-pool reset or release while captured operations can use the pool.

Index reads decode the active context into one shared BF16 workspace.
The [index scorer](../../../python/oh_my_vllm/models/deepseek_v41/model.py) limits query score chunks to 256 MiB per request.
This budget excludes decoded keys, inputs, selected outputs, and native library workspaces.
Candidate filters keep one boolean per block.
Query chunking does not bound the full decoded key workspace.

## Accepted state and transfers

Token pages hold compressed KV, index keys, and Engram history.
Accepted-state pages hold target windows, MTP windows, and incomplete compressor groups at an exact endpoint.
Window storage grows with accepted-state capacity, independently of token-page capacity.

Verification saves private endpoints before it overwrites recurrent storage.
Commit publishes only the accepted endpoint and scheduled checkpoints.
Restoration checks the endpoint before it clears the active slot.
A prefix copy includes only token-page fields.

Host snapshots keep encoded values and scale bytes.
DEP4 snapshots keep their owner rank after request release.
All ranks complete execution before owner release.
The shared [cache policy](../../cache-policy.md) controls logical ownership and transfer completion.

## Official DeepInfer formats

The upstream reference is DeepInfer revision `a679bb86ce36e883a98a97733492b686e5b9a08e` on NVIDIA SM100.
Its [cache definitions](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L109-L137) use these formats:

| State | Numerical format | Logical bytes per row |
|---|---|---:|
| Global KV, default | E2M1 with E4M3 G16 scales | 288 |
| Index keys | E2M1 with UE8M0 G32 scales | 68 |
| Sliding windows | E4M3 with UE8M0 G32 scales | 528 |

The [upstream allocator](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L406-L484) shares physical sources and aligns pages.
Its [compression state](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L205-L218) uses separate FP32 rings.
There is no persistent BF16 copy of full history.

[Main attention](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/config.h#L251-L278) expands cache values to BF16 operands and uses FP32 accumulation.
[Indexer scoring](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh#L371-L419) uses MXFP4 QK instructions with FP32 dot accumulation.
Its head weights and score reduction use BF16.
These upstream instructions do not define this adapter's numerical acceptance.

## Payload calculation

For 64 sequences of 800000 tokens, the four sources have 2000000 entries per sequence.
Compressed KV uses 36864000000 bytes, and index keys use 8704000000 bytes.
Together they use 45568000000 bytes, or 42.438507 GiB, with `1 GiB = 2^30 bytes`.

This is logical payload, not allocated memory or measured capacity.
Weights, windows, rollback state, alignment, page tables, Engram history, and temporary workspaces use additional storage.

Upstream [prefill expansion](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/attn.py#L723-L750) can allocate BF16 history for one source at a time.
A full ratio-1 expansion of those 64 histories alone uses `64 * 800000 * 512 * 2` bytes, or 48.828125 GiB.
This conditional calculation is not a measured peak.
The public source does not prove production replication or per-GPU sequence capacity.
