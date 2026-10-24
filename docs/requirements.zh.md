# 运行时契约

Rust 负责服务, 准入, 调度, 逻辑缓存和经过验证的状态提交.
Python 负责模型计算和物理张量.
进程间使用 ZMQ DEALER 和 msgpack.
开发和推理使用不同的依赖文件.

模型适配器具有对等地位.
各适配器提供配置, 能力, token 编解码器, 计算, 缓存策略和推测规则.
不得将一个模型的维度或语法作为另一模型的默认值.
参见[适配器契约](adapters.zh.md), [Qwen 适配器](models/qwen38/README.zh.md)和 [DeepSeek 适配器](models/deepseek_v41/README.zh.md).

模型算子使用 CUDA C++ 或 CuTe DSL.
原生 CUDA 库可以提供基础设施和计算.
不添加 DeepGEMM, FlashInfer 等外部算子库或 vLLM 依赖.
算子接口必须允许有收益的融合和 PDL.
优化期间保持模型专属的数值和状态规则.

调度和资源所有权使用 [GPU 执行规则](execution.zh.md).
包含 prefill 的 Execute 对全部分片使用 Eager 执行.
仅含 decode 的 Execute 使用 CUDA Graph 录制和重放.
调度器依次最大化 decode 请求数, prefill 整块数, 零散 token 数.
在这些优先级之后, 最小化包含零散后缀的请求数.

## 服务功能

适配器通过共享服务实现以下功能.
只有模型专属验收完成后才能声明支持该功能.

| 范围 | 契约 |
|---|---|
| HTTP | 模型列表, Chat JSON/SSE, Responses JSON/类型化 SSE, 诊断 |
| 已存响应 | 检索, 删除, 续接, 存储限制, `store=false` |
| 离线 | `run`, `bench`, token/prompt 文件, 到达控制, 前缀复用, 固定输出长度 |
| 采样 | Temperature, top-k, top-p, seed, frequency/presence/repetition penalties, 循环保护 |
| 停止 | EOS, `ignore_eos`, 停止字符串, 最小输出, 最大输出, 上下文限制 |
| 推理 | Effort 选择, thinking budget, 增量解析, usage 统计 |
| 工具 | Auto/none/required/named 选择, 并行调用控制, 历史, 流式参数 |
| 结构化输出 | JSON object/schema, token 掩码, 独立推测 grammar 状态 |
| 媒体 | 有序文本/图片/视频, 限制, 帧时间戳, 历史, 前缀隔离 |
| 缓存 | 整块和部分前缀, copy-on-write, 抢占, 主机换出, 恢复, 淘汰 |
| 生命周期 | 截止时间, 断连取消, 背压, worker 失败/替换, 正常关闭 |

每个请求只有一个 completion.
不支持 logprobs, 后台执行和自动截断.
启用的工具与结构化输出模式不能组合使用.
模型专属渲染不得静默改变公共 API 语义.

使用[测试和记录范围](testing.zh.md)以及各模型的固定数值限值.
[贡献流程](../CONTRIBUTING.zh.md)管理交付.
