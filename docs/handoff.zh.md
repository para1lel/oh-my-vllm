# 工作状态

在 `nightly` 上进行计划中的 [DeepSeek 扩展](deepseek-v41/design.zh.md).
服务在一张 B200 上运行 Qwen. DeepSeek 推理尚未启用.
架构和测量矩阵已获批准.
按[阶段计划](deepseek-v41/plan.zh.md)执行, 不再逐阶段请求批准.

## 当前状态

阶段 0 的参考和验收工作正在进行.
[检查点工具](deepseek-v41/checkpoint.zh.md)提供权重身份和惰性 CPU FP64 读取器.
[参考指南](deepseek-v41/reference.zh.md)分别说明 FP64 数学, 源精度量化和发布源码捕获.
[固定 fixture](deepseek-v41/fixtures.zh.md) 提供已审查的端点.
发布源码 GPU 捕获和复合验收限值仍待完成.

[长上下文生成](deepseek-v41/long-reference.zh.md)可从已保存状态恢复.
[分配工具](deepseek-v41/memory.zh.md)检查设计上限, 不证明运行时容量.

阶段 1 的[模型适配器](deepseek-v41/adapters.zh.md)及阶段 2 基础设施已完成.
[类型化执行接口](deepseek-v41/typed-execution.zh.md)在发布已接受状态前验证整个组的完成状态.
Qwen 保留单 worker 路径和缓存 coordinator.
DeepSeek 状态选择和物理模型执行是后续工作.

Qwen prefill 分派, 块调度和主机缓存启动修改已实现.

## 下一步

- 生成[长上下文 fixture](deepseek-v41/long-reference.zh.md).
- 在阶段 3 算子之前, 完成阶段 0 数值门槛和[分配边界](deepseek-v41/memory.zh.md).
- 保持已批准的[模型](deepseek-v41/model.zh.md), [状态](deepseek-v41/state.zh.md)和[采样](deepseek-v41/sampling.zh.md)契约.
- 后续实现修改遵循[测试范围](testing.zh.md)和[执行规则](execution.zh.md).
- 下次完整 HTTP 测量完成后, 替换 [Qwen 结果](deepseek-v41/qwen-results.zh.md).
- 遵守[交付流程](../CONTRIBUTING.zh.md#交付). 主机设置和原始产物不得放入受版本控制的文档.
