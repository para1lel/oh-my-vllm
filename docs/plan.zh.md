# 计划

[English](plan.md)。agent 以英文原文为准。

## 已完成

1. **2026-09-19 — 适配器。** Rust 负责调度和逻辑 KV 缓存，通过适配器对接已安装的 vLLM
   GPUWorker（ADR-001–003）。全部 9 组 EngineCore 测试通过。
2. **2026-09-21 — 服务与 V2 runner。**
   - OpenAI 兼容的 Chat/Responses 服务，通过真实 MTP4 和 oh-my-pi 验收（ADR-004）。
   - V2 Model Runner（ADR-005）。
3. **2026-09-21 — 独立运行时。** 模型、状态和 kernel 由项目自有，并基于独立库实现。构建、
   测试和推理均已移除 vLLM（ADR-006）。
4. **2026-09-22 — TTFT、长上下文和 TileLang。**
   - 刷新并冻结官方 vLLM 基线。
   - 新增按请求计的 TTFT 门槛，FA/GDN 容量分开配置（ADR-007）。
   - 支持 262144 token 上下文。
   - 把全部自有 Triton kernel 迁移到 TileLang。
5. **2026-09-22/23 — CUDA 迁移。**
   - 把全部自有 kernel 迁移到 B200 CUDA C++ 和内联 PTX，保留冻结的 TileLang 对照。
   - 全部 147 项算子用例和全部 12 组框架测试通过。
   - CUDA 现为默认后端（REQ-KERNEL-002）。

各阶段的证据索引见 [acceptance.zh.md](acceptance.zh.md)。

## 下一步

按批次修复 [2026-09-23 代码审计](audit-2026-09-23.zh.md) 发现的问题：

1. 服务可用性
2. 证据完整性
3. Kernel 与 worker 契约
4. 调度器加固
5. 经测量的性能改进
6. 清理

每个批次都必须保持 12 组性能门槛、数值容差、块大小 784，以及 Rust/Python 的职责划分。

## 不在计划内

- **超出范围：** CPU KV swap、PD/多节点、LoRA 和多模态执行。
- **只写文档（ADR-006）：** 未来的模型架构、其他 NVIDIA 后端、单机多卡和本地 DSpark
  草稿模型。ADR-006 描述了这些扩展，没有预留空接口。
