# 模型适配器抽取

## 当前范围

阶段 1 抽取从 Python 适配器入口开始.
[注册表](../../python/oh_my_vllm/models/registry.py) 根据配置中的 architecture 和 model type 选择已实现的模型系列.
目录名称不用于选择模型系列.
目前只注册 Qwen.
DeepSeek 配置仍会导致启动错误, 不会通过 Qwen 后端执行.

[适配器接口](../../python/oh_my_vllm/models/api.py) 提供 `ready`, `handle` 和 `close` 方法.
桥接层拥有 ZMQ DEALER 和 msgpack 传输.
适配器不拥有传输.
`stub` 路径仍直接选择测试占位实现.
所有其他路径均使用注册表.
原有 Qwen 执行, 缓存, 采样和媒体实现保持不变.

此入口不是完整的阶段 1 抽取.
模型描述符一致性检查, 共享状态契约和 Qwen 专用控制代码迁移将在后续步骤完成.
这些接口不启用分布式执行或 DeepSeek 功能.

## 选择与启动

加载器拒绝超过 1 MiB 的配置输入.
它拒绝重复 JSON 键和非对象配置值.
architecture 列表必须仅包含已注册的 architecture.
model type 也必须匹配.
未知 architectures 和不兼容格式会在导入 GPU 后端前报错.
请求的能力必须是所选适配器能力的子集.

[Qwen 入口](../../python/oh_my_vllm/models/qwen38/adapter.py) 使用系列键 `qwen38`.
原始检查点声明的 architecture 为 `Qwen3_5ForConditionalGeneration`, model type 为 `qwen3_5`.
这些原始标识符保持不变.
检查点必须使用动态 E4M3 FP8 量化和 128-by-128 权重块.
Qwen 入口在导入后端前, 将所使用的配置字节与冻结的 Qwen 配置 SHA256 比较.
现有权重加载器在加载期间再次检查检查点.

仅通过注册表选择并不能证明权重兼容性或数值验收通过.
它不检查权重载荷, 不分配缓存状态, 也不更改上下文上限.
Qwen 后端保留原有启动检查和数值门槛.
[阶段计划](plan.zh.md) 指定后续抽取与回归工作.

## Rust 描述契约

[model-contract crate](../../crates/model-contract/src/lib.rs) 为后续 readiness 一致性检查提供版本化类型.
它定义 `ModelDescriptor`, `CacheSchema`, `ParallelPlan` 和 `SpeculationSpec`.
反序列化拒绝未知字段和枚举值.
消费者使用描述符前, 调用 `validate()`.
仅反序列化不会验证版本或结构关系.

描述符绑定检查点, 配置和 tokenizer 的 SHA256 身份, 以及词表, 上下文, 能力, 格式, 缓存组, 并行策略和推测信息.
身份验证只检查语法, 不检查文件内容.
缓存组标识存储, 参与层, token 单位, 容量, 以及自有或共享生产者.
验证器拒绝重复组, 缺失生产者, 自引用和环.
元数据数量与名称长度上限约束结构验证.
传输代码必须在反序列化分配存储前设置输入字节上限.

并行计划使用逻辑 ranks, 而不是物理 GPU 索引.
TP1 使用一个 rank, TP4, TEP4 和 DEP4 各使用四个 ranks.
只有 DEP4 按 rank 划分请求.
推测信息将预测阶段数与草稿数量和验证行数分开.
MTP 可以使用一个阶段和四个草稿, DSpark 可以使用三个阶段和五个草稿.
验证容量等于最大草稿数量加一.

模型系列注册表必须验证所支持的布局, 维度和模型系列特定组合.
Rust 类型不启用执行, 不验证权重内容, 也不替代 rank readiness 和状态事务.
当前 worker 消息尚不包含这些描述符.
