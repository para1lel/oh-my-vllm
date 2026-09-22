# 文档索引

[English](README.md)。agent 使用英文原文；英文修改时同步同目录 `.zh.md` 译本。

先阅读 [AGENTS.md](../AGENTS.zh.md) 和 [handoff.md](handoff.zh.md)。前者规定项目约束，后者记录当前证据、阻塞项和待执行测试。agent 以对应英文版为准。

- [需求](requirements.zh.md)：模型、职责归属、功能和验收。
- [架构](architecture.zh.md)：当前运行时和模块职责。
- [设计](design.zh.md)：通信消息、逻辑/物理缓存和调度器生命周期。
- [开发](development.zh.md)：conda 包装脚本、构建和带时间戳的日志。
- [测试](testing.zh.md)：实际路径 FP64、文本、功能和性能命令。
- [验收](acceptance.zh.md)：完整矩阵、原始证据和覆盖范围限制。
- [性能分析](profiling.zh.md)：主机计时和可选 kernel 诊断。
- [计划](plan.zh.md)：已完成的里程碑和剩余工作。
- [服务](serving.zh.md)：已实现的 OpenAI 兼容子集、客户端命令和真实 GPU 验收证据。
- [贡献指南](../CONTRIBUTING.zh.md)：审查、检查和提交。
- [ADR001](decisions/ADR-001-zmq-socket-type.zh.md)：选择 DEALER socket。
- [ADR002](decisions/ADR-002-gpuworker-adapter.zh.md)：已安装的 Worker 和物理 ID。
- [ADR003](decisions/ADR-003-mtp-state-slots.zh.md)：推测状态和 BF16 SSM。
- [ADR004](decisions/ADR-004-openai-serving.zh.md)：以 Rust 为主的双 API 服务方向。

普通/MTP 文本、实际路径探针和功能组合已经通过检查。原有九组性能测试全部通过。具体配置见 acceptance.md，历史情况见 handoff.md；不要仅依赖旧提交时的状态记录。
