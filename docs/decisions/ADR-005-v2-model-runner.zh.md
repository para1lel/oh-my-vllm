# ADR-005：仅使用 V2 Model Runner

日期：2026-09-21。状态：**已被取代**，由 ADR-006（独立运行时，同日）取代。
V2 runner 及其 `new_block_ids_to_zero` 清零路径已移除；该字段现为未使用的提示（ADR-007、审计 MNT-04）。保留作为历史记录。

## 决策

保留 GPUWorker 作为 Python 边界，要求使用 V2 `vllm.v1.worker.gpu.model_runner.GPUModelRunner`。Rust 保留调度、逻辑 KV 分配及权威的已接受 token 历史。移除旧 runner 恢复分支和未使用的旧 MTP 辅助代码，不提供旧版回退。不修改 vLLM 源码，集成缺口在本仓库实现。

V2 在添加新请求前移除被抢占请求，并且只追加缓存块更新。因此恢复使用 NewRequestData，携带完整、已接受的 `prefill_token_ids` 和替换块表。Rust 仅在接纳/恢复时发送完整历史，不含 draft；普通 decode 仍为增量。原始 prompt 单独保留，使输出预算和生成 token 数在重计算后不丢失。前缀命中保留已计算 token 位置。

V2 默认 draft handler 避免复制无约束 draft，返回 -1 占位值，但 Rust 需要真实无符号 ID。本地 handler 子类将启用传输标志的 batch 浅拷贝传入现有 CUDA copy/event 实现，不改变实际 runner 的 grammar/sampling 标志。这会给普通 MTP 增加 D2H/event 开销，必须通过真实性能检查。

混合缓存页还需要 V2 的 new_block_ids_to_zero 契约。Rust 只在 BlockPool 中记录真正的新分配，排除缓存引用和活跃推测槽。Python 将逻辑分配展开为全部预留的物理 stride ID，执行前调用现有 V2 清零路径。启动预热后也原地清零缓存 tensor，保留 CUDA Graph 地址。实际 FP64 探针发现旧预热内容导致注意力 NaN；清零后恢复有限输出，且满足原容差。运行时分配清零进一步防止页复用后的旧混合布局字节污染。

## 验收

保留全部离线和服务功能、MTP4 约束、真实 OMP 任务、前缀命中、抢占/重计算与取消。保持已有实际 kernel FP64 容差，不要求完整模型 token 相等。检查 HTTP 日志中的异常性能，不新增严格服务吞吐门槛。

只运行框架侧的九组 ordinary/MTP/prefix × bs1/2/4、32768→4096 测量。对比 bench/baseline/2026-09-19-acceptance.json 中冻结的 EngineCore 测量，不重跑或选择更有利的基线。记录产物 hash、运行时源码身份和二进制身份。editable vLLM 源码位于 039b2ad67da6d64f7c1835c4738c7fb545ad37fc，而包元数据仍报告 g6376c601e，与历史版本标签一致。这是历史对比，不是同源码配对测试。

全部九组历史性能，以及真实 FP64/文本/MTP/服务/OMP 检查均通过。实测结果和源码身份限制见[验收证据](../acceptance.zh.md)。
