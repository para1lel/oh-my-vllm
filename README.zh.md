# oh-my-vllm

本推理框架使用 Rust 实现服务, 调度和逻辑状态.
Python 模型适配器通过 ZMQ DEALER 和 msgpack 调用原生 CUDA 算子.
各适配器负责自己的模型配置, 计算, 缓存布局和 token 语法.

安装, 命令和 HTTP 请求见[服务指南](docs/serving.zh.md).
服务提供 Chat 和 Responses API, 流式输出, 工具和结构化输出.

[English](README.md) | [贡献流程](CONTRIBUTING.zh.md)

## 模型适配器

| 适配器 | 执行 | 状态和文档 |
|---|---|---|
| Qwen3.8-27B-FP8 | 单张 B200, 普通解码和 MTP4, 总计 262144 tokens | [已实现的适配器](docs/models/qwen38/README.zh.md), [最新 HTTP 结果](docs/models/qwen38/results.zh.md) |
| DeepSeek-V4.1-Flash | 四张 B200, TP4/TEP4/DEP4, DSpark5, 总计 1048576 tokens | [适配器设计](docs/models/deepseek_v41/README.zh.md), [实施计划](docs/models/deepseek_v41/plan.zh.md); 服务推理尚未启用 |

适配器具有相同的架构地位. 实现和验收状态可以不同.
未知或尚未实现的模型导致启动错误.

## 开发

使用[适配器契约](docs/adapters.zh.md), [GPU 执行规则](docs/execution.zh.md)和[测试范围](docs/testing.zh.md).
模型目录包含各自的数学定义, 固定数值限值, 测量流程和最新完整结果.
[交接](docs/handoff.zh.md)给出当前工作和下一步.
