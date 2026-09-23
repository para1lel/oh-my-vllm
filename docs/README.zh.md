# 文档索引

[English](README.md)。agent 使用英文原文；英文修改时同步同目录 `.zh.md` 译本。

先阅读 [AGENTS.md](../AGENTS.zh.md) 和 [handoff.md](handoff.zh.md)。前者规定项目约束，后者记录当前状态和待办工作。agent 以对应英文版为准。

- [需求](requirements.zh.md)：模型、职责归属、功能和验收。
- [架构](architecture.zh.md)：当前运行时和模块职责。
- [设计](design.zh.md)：通信消息、逻辑缓存和调度器生命周期。
- [开发](development.zh.md)：conda 包装脚本、构建和带时间戳的日志。
- [测试](testing.zh.md)：CPU/GPU 测试套件、FP64 探针、功能组合和性能命令。
- [验收](acceptance.zh.md)：当前证据及历史产物的日期索引。
- [代码审计 2026-09-23](audit-2026-09-23.zh.md)：未修复问题和修复批次。
- [计划](plan.zh.md)：已完成阶段和下一步工作。
- [CUDA 开发](cuda-development.zh.md)：原生 kernel、冻结对照和算子门槛。
- [TileLang 开发](tilelang-development.zh.md)：冻结参考实现和 TileFoundry 工具。
- [性能分析](profiling.zh.md)：主机计时和 kernel 诊断。
- [服务](serving.zh.md)：OpenAI 兼容子集、客户端命令和服务检查。
- [贡献指南](../CONTRIBUTING.zh.md)：审查、检查和提交。
- 设计决策：[ADR-001](decisions/ADR-001-zmq-socket-type.zh.md) DEALER socket；
  [ADR-002](decisions/ADR-002-gpuworker-adapter.zh.md) GPUWorker 适配器（已被取代）；
  [ADR-003](decisions/ADR-003-mtp-state-slots.zh.md) MTP 状态槽和 BF16 SSM；
  [ADR-004](decisions/ADR-004-openai-serving.zh.md) Rust 优先的服务；
  [ADR-005](decisions/ADR-005-v2-model-runner.zh.md) V2 runner（已被取代）；
  [ADR-006](decisions/ADR-006-independent-runtime.zh.md) 独立运行时；
  [ADR-007](decisions/ADR-007-ttft-and-cache-capacities.zh.md) TTFT 门槛和独立容量。
