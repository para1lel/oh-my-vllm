# 工作状态

在 `nightly` 上继续工作, 不再逐阶段请求批准.
Qwen 与 DeepSeek 是对等[模型适配器](adapters.zh.md).
Qwen 在一张 B200 上提供推理. DeepSeek 服务推理尚未启用.

## 当前工作

共享代码提供模型描述符, 缓存策略, 类型化状态所有权和 rank 控制.
模型选择显式提供默认值. Qwen 传输规则保留在其适配器中.
模型选择和就绪检查共用一个测试模块.

调度器依次最大化 decode 请求数, prefill 整块数和 prefill 零散 token 数.
随后最小化包含零散后缀的请求数. 额外 decode 验证使用剩余预算.

[DeepSeek 计划](models/deepseek_v41/plan.zh.md)跟踪尚未完成的模型工作.
已实现独立 CUDA 算子, rank 自有权重, 共享专家输入, 文本旋转系数表和共享 pinned Engram 加载器.
尚未启用的块通过预先准备的 TP4, TEP4 和 DEP4 collective 组合 HC, attention, 归一化和 MoE.
DEP4 MoE 的 gate 和共享专家仅在各请求所有者本地计算.

HC 直接使用已完成的 FP32 attention 和 MoE 结果, 无需单独的转换 kernel.
启用面向用户的推理前, 必须闭合复合数值限值和完整分配边界.
物理状态, 完整推理, DSpark, 功能验收和已批准测量矩阵仍待完成.

## 下一步

- 完成未启用的 DeepSeek 目标计算, 同时继续数值和分配检查.
- 集成物理状态和 DSpark, 然后完成容量与功能验收.
- 正确性验收后, 运行已批准的 DeepSeek HTTP 和 AgentX 矩阵.

遵循[测试范围](testing.zh.md), [执行规则](execution.zh.md)和[交付流程](../CONTRIBUTING.zh.md#交付).
主机设置和进程记录保存在已忽略的 `LOCAL.md` 中. 原始产物保留在仓库外.
