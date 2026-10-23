# oh-my-vllm

本程序提供在单张 NVIDIA B200 上运行 Qwen3.8-27B-FP8 的 Rust 服务.
Python worker 加载模型, 执行文本, 图片和视频推理.
支持普通解码或 MTP4, 并提供约束解码.

安装, 命令和 HTTP 请求见[服务指南](docs/serving.zh.md).
服务提供 Chat 和 Responses API, 流式输出, 工具和结构化输出.
[Host KV cache](docs/kv-offload.zh.md) 使用预留的 pinned 内存, 支持请求换出和前缀复用.

[English](README.md) | [贡献流程](CONTRIBUTING.zh.md)

## 执行和检查

Worker 使用 [GPU 执行规则](docs/execution.zh.md)管理算子调度和资源所有权.
[全模型数值检查](docs/operators/evaluation.zh.md)将固定端点与直接 PyTorch FP64 计算进行比较.
[算子检查](docs/operators/microbench.zh.md)使用仓库内的 fixture 和固定的理论延迟倍率门槛.

## 测量和计划扩展

[Qwen 结果](docs/deepseek-v41/qwen-results.zh.md)包含最新完成的 HTTP 矩阵及其测量身份.
测量命令和定义见 [HTTP 流程](docs/performance.zh.md)和 [AgentX 流程](docs/agentx.zh.md).

[DeepSeek 架构](docs/deepseek-v41/design.zh.md)给出 `nightly` 上计划中的四 GPU 扩展.
DeepSeek 推理尚未启用.
