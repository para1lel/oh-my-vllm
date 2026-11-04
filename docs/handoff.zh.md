# 工作状态

使用 `nightly` 进行开发.

## 当前路径

共享行为见[架构](architecture.zh.md), [运行时契约](requirements.zh.md)和[执行规则](execution.zh.md).
模型指南负责各自的实现细节和验收范围.

- [Qwen](models/qwen38/README.zh.md) 提供最近完成的 [HTTP 和 AgentX 结果](models/qwen38/results.zh.md).
- [DeepSeek](models/deepseek_v41/README.zh.md) 提供四 rank text target 和 DSpark5 路径.
  其[数值验收](models/deepseek_v41/evaluation.zh.md), 完整容量验收, HTTP 矩阵和 AgentX 验收尚未完成.
- [计量](models/deepseek_v41/accounting.zh.md)定义当前 trace 契约.
  [性能指南](models/deepseek_v41/performance.zh.md)定义参考计算和批准的工作负载.

## 下一步

- 在数值验收前恢复独立 DeepSeek 端点数据.
- 对选定的并行策略完成 DeepSeek 数值和状态验收.
- 在修改长上下文容量前测量 cache page, state endpoint, projection storage 和 temporary storage.
- 在数值和容量验收后完成批准的 DeepSeek HTTP 和 AgentX 工作负载.
- 对适用的算子修改使用固定的 [Qwen 算子用例](models/qwen38/operators.zh.md).

将模型路径和主机设置放在 `LOCAL.md` 中.
将原始记录放在仓库外部.
