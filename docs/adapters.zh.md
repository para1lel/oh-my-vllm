# 模型适配器

每个受支持的模型族在自己的[模型目录](models)中维护实现和验收状态.
只有已实现的适配器才能启动服务.
显式 `stub` 后端提供服务测试模拟, 不提供模型推理.

## 职责

| 层次 | 职责 |
|---|---|
| Rust 服务 | HTTP, CLI, 准入, 传输, worker 生命周期 |
| Rust 调度器 | 请求队列, token 预算, 取消, 经过验证的状态提交 |
| Rust 模型策略 | 默认值, 能力, readiness, 输出解析, 缓存分配 |
| Python 模型适配器 | 权重, 计算, 物理状态, 编解码器, 媒体, 推测 |
| 共享 Python worker | ZMQ/msgpack, 执行策略, 传输, 约束, 重复检查, trace |
| 算子 | 原生计算, 执行计划, workspace 所有权 |

[Python 接口](../python/oh_my_vllm/models/api.py)提供 `ready`, `handle` 和 `close`.
适配器不负责传输.
[缓存策略](cache-policy.zh.md)将调度与模型存储语义分离.
[状态 arena](state-arena.zh.md) 和 [rank 控制](distributed.zh.md)提供类型化所有权和组内一致性.
这些接口不证明模型正确性或 GPU 容量.

模型代码放在 `python/oh_my_vllm/models/<family>/` 及对应的 Rust 模型模块中.
不将模型维度, token 语法或推测数量放在共享机制中.
只有计算和资源契约一致时才共享算子.
接口应支持有收益的融合和已准备的设备元数据.

## 选择与 readiness

[Python 注册表](../python/oh_my_vllm/models/registry.py)和 [Rust 选择器](../crates/zmq-worker/src/models/policy.rs)使用原始配置字节.
Architecture 和 model type 选择适配器; 目录名和公开 HTTP 名称不参与选择.
未知架构, 不支持的格式和能力不匹配在 GPU 导入或进程启动前导致启动错误.
只有字面量路径 `stub` 选择占位实现.

配置输入上限为 1 MiB.
Python 拒绝重复 JSON 键和非对象配置值.
各模型族验证支持的格式和 checkpoint 身份.
选定适配器提供默认值并验证显式选项.
公开的模型别名不能改变所选模型族.

Init 指定 `descriptor_version=1`.
Ready 在构造调度器前提供经过验证的描述符.
[Rust 类型](../crates/model-contract/src/lib.rs)和 [Python 验证器](../python/oh_my_vllm/models/descriptor.py)使用相同的版本化命名 map 字段.
它们拒绝未知字段, 不支持的版本, 无效值, 重复分组和无效 producer 链.
Python 布尔值不能替代整数.

描述符绑定 checkpoint, 配置, tokenizer, 词表, 上下文, 能力, 格式, 缓存分组, 并行策略和推测方式.
共享验证检查结构. 模型族策略检查支持的组合与已加载容量.
元数据数量和名称长度限制约束验证工作量.
传输层还在反序列化之前限制字节数.

当前 Qwen 适配器使用 TP1 和一个 rank.
推测将预测阶段与草稿数和验证容量分开.
验证容量等于最大草稿数加一.
描述符不能启用尚未实现的执行模式.

## 状态和输出

Rust 仅在输出和状态验证通过后发布结果.
分布式执行还检查全部必要 rank 的完成状态.
Python 将已接受状态与推测写入分离, 并应用模型专属恢复规则.
传输或 kernel 仍可能使用存储时, 取消不能释放存储.
Graph 和传输生命周期遵循[执行规则](execution.zh.md).

经过验证的模型族选择增量输出解析器.
Client 在其生命周期中保持该模型族, 并在替换 worker 时检查.
工具 grammar 来自适配器; 启用工具却缺少 grammar 回调会导致错误.
JSON object 和 schema 约束不需要工具 grammar 回调.
只有已接受 token 能改变约束和请求历史.

## 身份和兼容

身份验证必须检查实际消费的字节, 不能只检查摘要语法或文件时间戳.
Checkpoint 和 tokenizer 身份使用按文件标签排序的摘要 map, 编码为无末尾换行的紧凑 ASCII JSON.
SHA256 输入以一个版本化 domain 开始:

```python
b"oh-my-vllm/checkpoint-identity/v1\0"
b"oh-my-vllm/tokenizer-identity/v1\0"
```

文件名, 摘要或 domain 改变会改变身份.
缓存容量不改变 checkpoint 或 tokenizer 身份.
各加载器在 Ready 前和延迟使用权重前验证消费的文件.

冻结的 Qwen FP64 生成器使用 `worker.checkpoint` 导入.
它不是共享运行时 checkpoint 接口.
验收期间保持冻结源码身份和数值数组不变.
删除无用的迁移路径和测试; 只有当前消费者仍使用时才保留兼容代码.
