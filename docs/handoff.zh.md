# 工作状态

在 `nightly` 上进行计划中的 [DeepSeek 扩展](deepseek-v41/design.zh.md).
服务在一张 B200 上运行 Qwen. DeepSeek 推理尚未启用.
架构和测量矩阵已获批准.
按[阶段计划](deepseek-v41/plan.zh.md)执行, 不再逐阶段请求批准.

## 当前状态

阶段 1 的[模型适配器](deepseek-v41/adapters.zh.md)和阶段 2 的[分布式基础设施](deepseek-v41/distributed.zh.md)已实现.
DeepSeek 物理执行和状态选择仍待完成.
Qwen 保留单 worker 路径, 已修复 prefill 分派, 块调度和主机缓存启动.

阶段 3 已有使用[固定夹具和数值限值](deepseek-v41/fixtures.zh.md)的独立 CUDA 算子.
[Engram 加载器](deepseek-v41/memory.zh.md#4-权重字节与格式)在一个共享存储中提供完整编码表.
已实现 rank 自有投影和分组专家投影, 最终 head 折叠, 以及精确设备转换.
模型集成仍待完成.

[参考工具](deepseek-v41/reference.zh.md)区分 FP64 数学, 源精度量化和发布源码诊断.
阶段 0 的复合数值限值和物理分配边界仍未闭合.

## 下一步

- 在全模型集成前, 完成复合数值限值和[分配边界](deepseek-v41/memory.zh.md).
- 在适用门槛关闭后继续其余算子和 Engram 模型集成.

遵循[测试范围](testing.zh.md), [执行规则](execution.zh.md)和[交付流程](../CONTRIBUTING.zh.md#交付).
将主机设置和进程记录保存在已忽略的 `LOCAL.md`.
将原始产物保存在仓库之外.
