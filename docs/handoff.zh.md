# 工作状态

在 `nightly` 上继续工作, 不再逐阶段请求批准.
Qwen 与 DeepSeek 是对等[模型适配器](adapters.zh.md).
Qwen 在一张 B200 上提供推理. DeepSeek 服务推理尚未启用.

## 当前工作

共享代码提供模型描述符, 缓存策略, 类型化状态所有权和 rank 控制.
调度器更新依次遵循 decode 请求数, 整块数, 零散 token 数和零散后缀优先级.
适配器清理移除隐式 Qwen 默认值和过时迁移说明.
交付前, Qwen 数值与 HTTP 验证必须绑定最终运行时候选版本.

[DeepSeek 计划](models/deepseek_v41/plan.zh.md)跟踪尚未完成的模型工作.
已实现独立 CUDA 算子, rank 自有权重, 分组投影和共享 pinned Engram 加载器.
全模型集成前必须闭合复合数值限值和完整分配边界.
物理状态, 完整推理, DSpark, 功能验收和已批准测量矩阵仍待完成.

## 下一步

- 完成调度器和适配器清理, 独立审查及适用的 Qwen 回归.
- 闭合 DeepSeek 数值与分配门槛, 然后继续模型集成.

遵循[测试范围](testing.zh.md), [执行规则](execution.zh.md)和[交付流程](../CONTRIBUTING.zh.md#交付).
主机设置和进程记录保存在已忽略的 `LOCAL.md` 中. 原始产物保留在仓库外.
