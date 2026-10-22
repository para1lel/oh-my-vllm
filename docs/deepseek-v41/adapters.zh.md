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
Qwen 计算和已接受状态规则保持不变.

这不是完整的阶段 1 抽取.
共享状态契约, Rust 模型特定控制和完整的 Qwen HTTP 回归矩阵是必要的后续步骤.
这些接口不启用分布式执行或 DeepSeek 功能.

## Python 所有权

[Qwen 目录](../../python/oh_my_vllm/models/qwen38)拥有权重, 缓存布局, 模型计算, 采样, MTP4, 请求状态, 分词器和媒体处理.
其中的[卸载实现](../../python/oh_my_vllm/models/qwen38/offload.py)拥有神经模型快照和恢复.
[stub 模块](../../python/oh_my_vllm/models/stub.py)拥有显式测试后端的模拟卸载.

共享 worker 保留传输, 执行策略, 元数据上传, 重复检查, 追踪记录, 主机内存预算和约束机制.
其传输辅助函数仍验证现有的成对缓存线协议格式.
这些辅助函数不提供与模型无关的状态事务.

[Qwen 工具语法](../../python/oh_my_vllm/models/qwen38/tool_grammar.py)向共享约束提供模型特定的 token 语法.
启用工具时, 必须显式提供语法回调.
缺少回调会导致错误, 不会选择 Qwen 语法.
JSON 对象和 JSON schema 约束不需要工具语法回调.

旧 `worker.checkpoint` 模块仅导入 Qwen checkpoint 类.
此兼容路径使固定的 Qwen 参考生成器保持不变.
其他模型模块使用新的模型系列路径.
数值夹具和验收门槛保持不变.

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

[model-contract crate](../../crates/model-contract/src/lib.rs) 为 readiness 一致性检查提供版本化类型.
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
Ready 消息包含模型描述符.

## Python 描述契约

[Python 验证器](../../python/oh_my_vllm/models/descriptor.py) 使用与 Rust 相同的版本化具名映射字段.
它拒绝未知字段, 无效值, 重复组, 无效生产者链以及超出类型范围的整数.
布尔值不能替代整数.
这些检查仅使用 CPU 数据.

[Qwen 构建器](../../python/oh_my_vllm/models/qwen38/descriptor.py) 使用经验证的原始配置字节以及已加载的检查点和分词器身份映射.
它不读取检查点文件, 也不导入 GPU 后端.
调用方必须在提供身份前验证文件字节.
已解析配置必须与经验证字节一致.
值类型也必须一致.

每个身份映射将文件标签关联到小写 SHA256 摘要.
检查点映射使用 `config` 和 `index` 标签, 以及所有已加载分片的文件名.
分词器映射包含 `tokenizer.json`, `generation_config.json` 和 `chat_template.jinja`.
辅助程序将键排序, 编码为没有末尾换行的紧凑 ASCII JSON.
SHA256 输入以一个版本化域开头, 以 JSON 字节结尾:

```python
b"oh-my-vllm/checkpoint-identity/v1\0"
b"oh-my-vllm/tokenizer-identity/v1\0"
```

修改文件名, 摘要或域都会改变身份.
修改缓存容量不会改变检查点或分词器身份.
辅助程序将所有提供的身份条目纳入哈希.
它不证明文件内容或内存容量.

Qwen 提供现有的 262144 token 上限, TP1, 以及一阶段, 四草稿, 五验证行的 MTP 上限.
MTP 上限不会在普通解码模式中启用 MTP 执行.
配置的缓存成员仅在 MTP4 模式中包含预测器 FA 数组.

两个缓存组使用版本化的 Qwen 布局 ID.
FA 存储使用 BF16.
GDN 布局包含 FP32 递归状态, BF16 卷积历史和 BF16 边界隐藏状态数组.
容量为已分配的调度器池计数.
池计数包含编号为零的空块或空槽.

私有槽, 临时槽, 主机存储和执行工作区不会增加调度器容量.
布局保留现有成对预留和提交规则.
它们不是独立状态事务或共享生产者存储.

worker 在缓存分配后使用构建器.
注册表提供经验证的原始配置字节, 不重新读取文件.
在 Ready 前, 检查点读取器对索引中的所有分片求哈希, 并再次验证文件字节.
此步骤防止后续视觉权重懒加载改变检查点文件集合.
分词器也验证其原始资源字节.
Ready 返回描述符的私有副本.

在 Ready 后, 读取器在每次张量读取前后验证分片文件状态.
它在 Ready 后首次读取每个分片前再次验证冻结的摘要.
视觉权重加载后, 在模型能使用新权重前进行完整身份检查.
仅凭文件时间戳不能证明文件内容未变.

## 就绪协议

Init 必须指定 `descriptor_version=1`.
Python 在构建适配器前拒绝缺失或不支持的版本.
Ready 必须提供描述符.
旧消息不能使用默认描述符值.
Rust 在构建调度器前验证模式和已实现的模型系列.

它将缓存布局, 层成员, 块单位和已分配容量与所选模式比较.
旧的 FA 和 GDN 容量字段必须与描述符中的对应组一致.
请求上下文上限不得超过模型支持的上限.
Qwen 保留 TP1, 普通解码和 MTP4.

显式 `stub` 后端具有独立测试身份和仅含元数据的缓存布局.
其 `f32` 权重格式条目是测试协议哨兵, 不是已加载的模型权重.
其能力条目规定服务测试模拟, 不表示神经网络推理或媒体解码.
Rust 仅为 `stub` 路径接受此描述符.
Qwen 不能在加载错误后使用它.

就绪协议不提供数值或 GPU 容量验收.
[阶段计划](plan.zh.md) 规定 Qwen 回归检查和模型特定控制提取的后续步骤.
