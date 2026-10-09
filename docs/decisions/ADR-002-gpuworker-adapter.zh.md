# ADR-002: 历史 GPUWorker adapter

日期: 2026-09-19. 状态: 由 [ADR-006](ADR-006-independent-runtime.zh.md) 和 [ADR-007](ADR-007-ttft-and-cache-capacities.zh.md) 替代.

## 原始决策

首个原型使用 vLLM GPUWorker adapter, Rust 负责调度和逻辑 KV.
Adapter 将 Rust block table, 调度 token 和请求状态转换为 worker 输入.
缓存转换包括 physical-stride 映射和基于 `floor(num_gpu_blocks/3)` 的逻辑容量.
该方案支持目标 checkpoint 上早期调度器和 MTP 测试.

## 替代

独立运行时移除了 adapter 及其 vLLM 依赖.
Block ID 现在直接寻址 FA / GDN slot.
历史 CLI 容量单位保留, 旧 physical 映射已移除.
GDN 容量有独立限制, 用于避免反复分配未使用状态.
[原始证据](../../bench/evidence/2026-09-19-acceptance.json) 只保留对应历史范围.
当前 12 行分母和独立实现替代其早期验收.
