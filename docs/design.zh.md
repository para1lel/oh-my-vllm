# 设计：oh-my-vllm

## 概览

oh-my-vllm 按清晰边界分工：Rust 拥有调度策略和 KV 缓存记账，Python 拥有 GPU。两端不共享内存，每个推理步骤通过 ZMQ DEALER socket 交换 msgpack 编码消息。

## ZMQ 协议

两端使用 `zmq.DEALER` / `DealerSocket`。Rust bind `ipc://<socket_path>`，Python connect。拓扑始终为一个 Rust 客户端对应一个 Python worker，因此 DEALER↔DEALER 等价于同步请求/响应通道。

全部消息均为带 `"type"` key 的 msgpack dict。

**Rust → Python：**

```
{"type": "init",
 "model_path": str,
 "num_gpu_blocks": int,
 "mamba_blocks": int | null,   # 独立 GDN 池容量（ADR-007）
 "block_size": int,            # Qwen3.8 为 784
 "tensor_parallel_size": int,
 "max_model_len": int,
 "num_speculative_tokens": int}  # 0 表示关闭 MTP

{"type": "register",
 "request_id": int,
 "prompt_token_ids": list[int]}

{"type": "execute",
 "rpc_id": int,
 "step_id": int,
 "scheduled": [
   {"request_id": int,
    "token_ids": list[int],
    "num_computed_tokens": int,
    "prefill_token_ids": [int],  // 仅接纳/恢复时：完整已接受历史
    "fa_block_table": list[int],
    "mamba_block_table": list[int]}
 ],
 "finished_request_ids": list[int],
 "preempted_request_ids": list[int],
 "num_batched_tokens": int}

{"type": "prepare", "rpc_id": int, "request_id": int, "request": {...}}
{"type": "abort",  "request_id": int}   # 无回复；取消/清理
{"type": "shutdown"}
```

`register` 和 `abort` 没有回复。register 异常会记录日志；缺失的注册会以请求局部 execute 错误报告。prepare 和 execute 回复回显 `rpc_id`；Rust 会丢弃取消 prepare 后迟到的回复。

只在已完成执行步骤之间取消请求。`Scheduler::abort` 释放 Rust 所有权，并将 `finished_request_ids` 通知排入下次 execute；即使已无调度请求，也需发送。driver 必须刷出最后通知。Python 清理注册、采样、已接受状态和 MTP 进度。旧独立 `abort` 消息使用相同的 finished-only 路径，不回复，也不改变 Rust 状态。

**Python → Rust：**

```
{"type": "ready", "logical_num_blocks": int, "mamba_blocks": int}

{"type": "prepared", "rpc_id": int, "prompt_token_ids": list[int]}

{"type": "execute_result",
 "rpc_id": int,
 "outputs": [
   {"request_id": int,
    "error": str | null,          # 仅服务模式，可选：请求局部故障
    "token_ids": list[int],  # prefill 为空、普通为单 token、MTP 为被接受输出
    "num_accepted_draft_tokens": int,  # MTP 接受的 draft 数量
    "new_draft_token_ids": list[int],  # 项目 MTP proposer 生成的实际下一批 draft
    "text": str,                  # 仅服务模式，可选：增量解码文本
    "finish_reason": str | null,  # 仅服务模式，可选：stop/length
    "reasoning_tokens": int}      # 仅服务模式，可选：累计值
 ]}

{"type": "error", "rpc_id": int | absent, "kind": "validation" | "internal" | absent,
 "message": str}
```

ready 容量从已分配的 FA 和 GDN 设备张量读取。
prepare 错误包含 `rpc_id` 和 `kind`；init 错误均不带这两个字段，execute 错误
包含 `rpc_id`，`kind` 缺失时默认为 internal。

## KV 缓存数据结构

### 块布局

Qwen3.8-27B 有两种注意力组，需要独立块表：

- **组 0（全注意力）：** 16 层，标准块布局，每块 `block_size=784` token，启用前缀缓存。
- **组 1（GatedDeltaNet / Mamba）：** 48 层，`mamba_cache_mode="align"`。包含运行/checkpoint 状态、临时保护的前一状态，以及 MTP 模式的 K 个推测槽。null 占位保留块位置。缓存 checkpoint 可在请求结束后继续存活，直到共享池淘汰。

默认两组从一个共享 `BlockPool` 取块，因此 FA 块和 Mamba checkpoint 竞争同一逻辑池。使用 `--mamba-blocks N`（当前所有验收运行均使用）时，GDN 组拥有独立的 N 槽池，ID 在各组内局部有效（ADR-007）。Python 按各自容量分配独立逐层 tensor，块 ID 直接寻址槽。保留历史 CLI 容量单位：ready 暴露 floor(num_gpu_blocks/3)。不存在物理 stride 重映射；ADR-002 描述的是已被替代的实现。

### BlockPool

`BlockPool` 用按块 ID 索引的平坦 `Vec<Block>` 持有全部块元数据，并使用手写双向 LRU 空闲队列 `FreeKVCacheBlockQueue`。以 `enable_caching=true` 构造时，带非空 hash 的已释放块作为缓存保留，直到新分配淘汰它们。这与 vLLM 的 `KVCacheBlock` 设计一致。

### 前缀缓存

链式 hash：每块 hash 为 `SHA-256(parent_hash || token_ids || extra_keys)[0..8]`，截断到 64 位。每个 hash 依赖全部前序 token 内容，但不保证前序块仍驻留。FA 查找要求连续驻留前缀，再与可用 Mamba checkpoint 协调。

coordinator 的 `find_longest_cache_hit` 返回两组一致的最长前缀。对 Qwen3.8 的 `is_simple_hybrid=True`，先计算 FA 命中长度，再将 Mamba 命中限制在该范围，最终长度为 `min(fa_len, mb_len)`。

### 两阶段块分配（vLLM issue #33775）

`allocate_slots` 遵循 vLLM 的顺序，防止从前缀缓存移入空闲队列的块被重复计数：

1. `remove_skipped_blocks`：释放当前步骤不再需要的 Mamba 块，即 align 模式下前两步的块。
2. `get_num_blocks_to_allocate`：只读池，计算所需容量。
3. 检查包含 watermark 的可用容量；不足则返回 `None`，作为抢占信号。
4. `add_local_computed_blocks`：任何组分配新块前，先触达并从队列移除全部复用缓存块。
5. 为每组执行 `allocate_new_blocks`。
6. `cache_blocks`：将新填满的块注册到前缀缓存。

## 调度器

调度器位于 `crates/scheduler/src/lib.rs`，每步运行两阶段循环：

1. **运行队列：** 遍历已有请求，为后续 token 分配槽。`allocate_slots` 返回 `None` 时，通过重计算抢占：释放块，重置 `num_computed_tokens=0`，移至等待队列头。
2. **等待队列：** 在 `max_num_seqs` 和 `max_num_batched_tokens` 限制内接纳新请求。每步预算来自 `max_num_batched_tokens`，每请求取 `min(remaining, budget)` token，然后显式按 Mamba 对齐拆分 prefill，使 checkpoint 在可复用边界处生成。

仅支持重计算抢占，不做 CPU swap。当前代码抢占的是分配失败的请求，而非最后接纳的请求；多个被抢占请求以相反优先级顺序重新入队（审计 SCH-06）。`update()` 信任 worker 返回的接受数和 token 列表（审计 SCH-01）。

## MTP 推测解码

num_speculative_tokens=4 时，独立 Worker 对照 target sample 验证已调度 draft，返回保留 token 和明确的下一批 draft ID。中间 prefill 返回空输出；不会把多项已接受列表压缩为单项。Rust 预留递归状态槽，在分块后迁移已提交源，只回滚已调度且被拒绝的 draft。前缀接纳和重计算恢复携带完整已接受历史；运行中更新只追加增量后缀。即便没有模型 token，仍刷出 finished ID。BF16 MTP 状态保留 block784，与冻结基线相同。不存在 vLLM runner 或 draft handler。

MTP 缓存按输入 token 索引，首个有效位置为 1；边界 hidden feature 恢复前缀而不修改共享页。若已分配页容得下额外三次写入，proposal 图生成四个贪心 draft，中间无主机同步；否则逐步生成在分配/上下文边界裁剪。target 验证和 draft-head 预热在请求各 query 间共享 KV 读取，并保留独立因果 mask。

自有 kernel 完全覆盖已提交递归目的地，新请求从不可变零状态初始化，FA 读取按有效长度 mask。原先合并 FA/GDN 命名空间的清零提示已从 wire 协议删除；Python 从未消费该字段。计算图预热/捕获在正式回放前保存并恢复所有被修改的 FA/状态目的地。

## 服务协议扩展（2026-09-21）

`prepare` 携带 rpc_id、request_id 和归一化请求对象（messages、tools、effort/template 选项、输出格式、max_tokens、sampling、stop）。Python 校验、编译约束并注册请求，在 Rust 接纳前回复带 prompt_token_ids 的 `prepared` 或带类别的 `error`。只有明确的验证错误返回 HTTP 400；worker 内部错误返回 500。旧 `register` 仍用于 Run/Bench 的固定长度贪心模式。

服务 execute 输出可额外包含 text（增量解码文本）、finish_reason（stop/length 或 null）及累计 reasoning_tokens；Rust 把它们存在逐请求结果映射 `serving_outputs` 中。token ID 仍是 Rust 调度/KV 的权威输入。grammar mask 完全保留在 Python，由自有 target sampler 应用，包括推测验证行。Rust 等待回复时检测 worker 退出，并为 EOS、长度和取消使用现有 finished_request_ids 清理路径。
