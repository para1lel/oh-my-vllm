# oh-my-vllm

面向单张 B200 GPU 和 Qwen3.8-27B-FP8 的 Rust 推理框架.
Rust 负责 HTTP 服务, 请求调度和逻辑 KV 缓存.
Python 通过第三方 GPU 库和项目内核完成模型计算.
两个进程通过 ZMQ DEALER socket 交换 msgpack 消息.

目标模型包含 48 层 GDN 和 16 层 FA.
运行时使用 784-token 块, 支持输入与输出合计 262144 token.
支持普通解码, MTP4, DSpark, 前缀复用, 重计算抢占和约束生成.
HTTP 服务提供 Chat Completions 和 Responses API, 包括流式响应和函数工具.

## 启动

按 [开发指南](docs/development.zh.md) 安装环境并设置模型位置.
使用当前服务器上存在的 checkpoint 绝对路径:

```bash
export OH_MY_VLLM_MODEL="/path/to/Qwen3.8-27B-FP8"
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker serve
```

服务默认监听 `127.0.0.1:8000`.
[服务合同](docs/serving.zh.md) 说明限制, 工具和请求示例.
[测试指南](docs/testing.zh.md) 说明正确性和性能验证流程.

DSpark 使用不同的草稿 checkpoint 和相同的目标模型:

```bash
export OH_MY_VLLM_DRAFT_MODEL="/path/to/Qwen3.8-27B-DSpark"
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --speculative-mode dspark serve
```

[架构](docs/architecture.zh.md) 说明草稿算法和目标模型验证.

## 性能实测

当前十五行矩阵使用普通解码, MTP4, MTP4 prefix-hit 和 DSpark.
每组 batch 为 1/2/4, 输出为 4096 个 token. 新一轮采集待完成.

普通解码和 DSpark 使用 32768 个输入 token.
MTP4 使用 32768, 131072, 以及命中 32144 个前缀 token 的 32768 输入组.

每个阶段完整预热两次, 测量五次.
三倍理论下界和 10% 波动门槛继续有效, 保留 prefix-hit prefill 倍率的既定例外.
[需求](docs/requirements.zh.md) 说明计时, [验收](docs/acceptance.zh.md) 标明实测源码范围.

## 长期 agentic 服务

当前测试为每种模式保留 4 个固定 oh-my-pi 会话, 至少持续 1 小时.
2 个会话使用 Chat, 2 个使用 Responses. Session ID 和历史保持连续.
不设请求数下限, 不单独预热. 首次请求的编译计入测量.

OMP 使用 `qwen3` tokenizer, `262144` context window 和运行时 provider usage 决定自动压缩.
任务和摘要使用 `max`, 映射为服务的 `xhigh`. 模型请求共享 4 并发上限.

新的持久会话采集及资源审查待完成.
最终图片展示框架开始处理的时刻, 以及输出 token 数除以处理墙钟时间, 包含 reasoning token.
图片按任务和压缩用途标记, 不提供 API 颜色图例.

此重型测试不设性能门槛, 日常命令默认跳过.
[测试](docs/testing.zh.md#长时-agentic-服务) 说明流程和内存审查.

## 阅读

- [需求](docs/requirements.zh.md): 范围和验收条件.
- [架构](docs/architecture.zh.md): 职责, 数据流和实现决策.
- [验收](docs/acceptance.zh.md): 实测结果及源码身份.
- [开放工作](docs/handoff.zh.md): 当前状态和后续任务.
- [文档索引](docs/README.zh.md): 指南和决策记录.
- [交互教程](code-journey/README.zh.md): 13 个中文章节, 使用真实源码和调度实验.
- [贡献规则](CONTRIBUTING.zh.md): 检查和提交流程.

[验收索引](docs/acceptance.zh.md) 标明每次测量的源码和测试范围.
框架性能采用十五行 prefill / decode 测量和理论资源下界.
项目构建, 测试和推理均不使用 vLLM 实现.
正式验证使用完整原始记录.

服务器专属配置记入被忽略的 `LOCAL.md`.

[English](README.md)
