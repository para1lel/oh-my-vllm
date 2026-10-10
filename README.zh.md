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

这十五项负载使用源码 `fabcede8c6e0e6e4fa2c90d5d97b1f98d9c7dfdb`, 每项在一张 B200 上运行.
每个请求保留 4096 个输出 token. 每项负载完整预热两次, 测量五次.
Prefix-hit 请求复用 32144 个 token, 其他行使用冷前缀.

Prefill 从请求提交到首 token, decode 从首 token 到末 token.
每次重复分别取请求中最长的阶段区间.
Decode TPS 为 `4095 * batch / median(decode_wall_seconds)`.
理论百分比为 `100 * median(decode_bound_seconds) / median(decode_wall_seconds)`.
它表示实测吞吐占理论最大吞吐的比例.

| 模式 | 输入 | Batch | Prefill (s) | Prefill 倍率 | Decode TPS | 理论占比 (%) | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| ordinary | 32768 | 1 | 1.268106 | 2.871 | 115.1 | 43.5 | 通过 |
| ordinary | 32768 | 2 | 2.555269 | 2.892 | 219.4 | 45.0 | 通过 |
| ordinary | 32768 | 4 | 5.224045 | 2.956 | 375.2 | 44.4 | 通过 |
| mtp4 | 32768 | 1 | 1.310482 | 2.883 | 395.2 | 45.2 | 通过 |
| mtp4 | 32768 | 2 | 2.615408 | 2.877 | 630.4 | 45.5 | 通过 |
| mtp4 | 32768 | 4 | 5.328181 | 2.930 | 1010.4 | 43.5 | 通过 |
| prefix | 32768 | 1 | 0.037446 | 3.723 | 115.5 | 43.7 | 通过 |
| prefix | 32768 | 2 | 0.061593 | 3.062 | 219.7 | 45.0 | 通过 |
| prefix | 32768 | 4 | 0.120962 | 3.007 | 399.7 | 47.3 | 通过 |
| ordinary | 131072 | 1 | 7.059579 | 2.501 | 104.1 | 48.1 | 通过 |
| ordinary | 131072 | 2 | 14.201501 | 2.516 | 166.5 | 48.0 | 通过 |
| ordinary | 131072 | 4 | 28.825513 | 2.553 | 219.2 | 44.2 | 通过 |
| dspark | 32768 | 1 | 1.279684 | 2.862 | 174.4 | 38.8 | 通过 |
| dspark | 32768 | 2 | 2.540217 | 2.841 | 246.0 | 38.8 | 通过 |
| dspark | 32768 | 4 | 5.292443 | 2.959 | 430.4 | 41.5 | 通过 |

各阶段须通过有效门槛. Prefix-hit prefill 记录倍率, 三倍倍率不作为门槛.
其 10% 波动检查与 decode 门槛继续有效.
[验收](docs/acceptance.zh.md) 给出原始摘要, 失败尝试, 独立审查及验证范围.

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
