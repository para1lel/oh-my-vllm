# 物理缓存参考

[缓存实现](../../../python/oh_my_vllm/models/deepseek_v41/cache.py)将 Rust 逻辑 block 映射到模型存储.
[Packed 存储](../../../python/oh_my_vllm/models/deepseek_v41/packed_cache.py)将编码历史及其 scale 保存在独立平面.
[量化契约](quantization.zh.md)规定其数值和舍入.

## 当前存储

| 状态 | 格式 | 每行逻辑字节数 |
|---|---|---:|
| Target 和 MTP window | 512 个 BF16 值 | 1024 |
| 压缩 KV | 512 个 E2M1 值, 每 16 个值一个 E4M3 scale | 288 |
| Index key | 128 个 E2M1 值, 每 32 个值一个 UE8M0 scale | 68 |

一行指一个物理 source 的一个缓存向量.
它不指一个原始 token 在所有层的总存储.
E2M1 编码每字节存储两个值.
E4M3 和 UE8M0 scale 的数值含义不同.
压缩 KV 包含全部 512 个维度, 没有 BF16 RoPE 尾部.

[Source map](../../../python/oh_my_vllm/models/deepseek_v41/adapter.py) 选择第 2, 8, 14 和 20 层作为 KV source.
压缩比分别为 2, 2, 2 和 1.
八个 index-source 层共享四个物理 key source.
使用方不再分配另一份完整历史.

活跃请求 slot 使用 compressed KV, index key 和 Engram history 的 page-table view.
CUDA 存储按需映射物理页, tensor 地址保持不变.
CPU 执行使用稠密存储.
已绑定的存储拒绝 tensor device 或 dtype 转换.
捕获的操作仍可使用 page pool 时, Graph lease 阻止 pool 重置或释放.

Index 读取将活跃上下文解码到一个共享 BF16 workspace.
[Index scorer](../../../python/oh_my_vllm/models/deepseek_v41/model.py) 将每个请求的 query score chunk 限制在 256 MiB.
该预算不包含已解码 key, 输入, 已选输出和原生库 workspace.
Candidate filter 为每个 block 保留一个 boolean.
Query chunking 不限制完整的已解码 key workspace.

## 已接受状态和传输

Token page 保存 compressed KV, index key 和 Engram history.
Accepted-state page 在精确端点保存 target window, MTP window 和未完成 compressor group.
Window 存储随 accepted-state 容量增长, 与 token-page 容量独立.

Verification 在覆盖 recurrent storage 前保存私有端点.
Commit 只发布已接受端点和已调度 checkpoint.
Restoration 在清空活跃 slot 前检查端点.
Prefix copy 只包含 token-page 字段.

Host snapshot 保留编码值和 scale 字节.
DEP4 snapshot 在请求释放后保留 owner rank.
所有 rank 完成执行后才释放 owner.
共享[缓存策略](../../cache-policy.zh.md)控制逻辑所有权和传输完成.

## Proposal Graph 所有权

[Proposal Graph 缓存](../../../python/oh_my_vllm/models/deepseek_v41/target_execution.py#L285)<!-- source: graph = self.proposal_graphs.get(slot) -->中每个请求 slot 最多有一个 proposal Graph.
它先关闭绑定已失效的图, 再创建另一个图.

[完成的主机 store](../../../python/oh_my_vllm/models/deepseek_v41/cache_transfer.py#L145)<!-- source: self._release_request_slot(operation.get("request_id")) --> 通过 worker 回调释放请求 slot.
Worker 的 [`_free_slot`](../../../python/oh_my_vllm/models/deepseek_v41/inference.py#L46)<!-- source: def _free_slot(self, request): --> 先关闭 proposal Graph, 再清空和释放 slot.
[`Worker.close`](../../../python/oh_my_vllm/models/deepseek_v41/inference.py#L243)<!-- source: def close(self): --> 先关闭 proposal Graph, 再关闭缓存存储和权重.
[图释放流程](../../../python/oh_my_vllm/models/deepseek_v41/proposal_graph.py#L179)<!-- source: def close(self): -->先同步设备并重置图, 再释放其 owner 和 PageLease.
[执行指南](README.zh.md#mtp-proposal-graph)给出条件和默认设置.

## 官方 DeepInfer 格式

上游参考为 NVIDIA SM100 上的 DeepInfer revision `a679bb86ce36e883a98a97733492b686e5b9a08e`.
其[缓存定义](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L109-L137)使用以下格式:

| 状态 | 数值格式 | 每行逻辑字节数 |
|---|---|---:|
| Global KV, 默认 | E2M1 和 E4M3 G16 scale | 288 |
| Index key | E2M1 和 UE8M0 G32 scale | 68 |
| 滑动 window | E4M3 和 UE8M0 G32 scale | 528 |

[上游分配器](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L406-L484)共享物理 source 并对齐 page.
其[压缩状态](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L205-L218)使用独立 FP32 ring.
它不持久保存完整历史的 BF16 副本.

[主 attention](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/config.h#L251-L278) 将缓存值展开为 BF16 操作数, 并使用 FP32 累加.
[Indexer 评分](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh#L371-L419)使用 MXFP4 QK 指令和 FP32 dot 累加.
其 head weight 和 score reduction 使用 BF16.
这些上游指令不定义此适配器的数值验收.

## Payload 计算

对于 64 条各含 800000 tokens 的序列, 四个 source 合计每条序列 2000000 个 entry.
Compressed KV 使用 36864000000 字节, index key 使用 8704000000 字节.
合计 45568000000 字节, 即 42.438507 GiB, 其中 `1 GiB = 2^30 bytes`.

这是逻辑 payload, 不是已分配内存或实测容量.
权重, window, rollback state, alignment, page table, Engram history 和临时 workspace 使用额外存储.

上游 [prefill 展开](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/attn.py#L723-L750)可每次为一个 source 分配 BF16 历史.
仅完整展开这些 64 条历史的 ratio-1 source, 就使用 `64 * 800000 * 512 * 2` 字节, 即 48.828125 GiB.
该有条件计算不是实测峰值.
公开源码不证明生产部署的复制方式或每张 GPU 的序列容量.
