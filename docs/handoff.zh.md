# 工作状态

继续在 `nightly` 上工作. 仓库保留 Qwen 服务主线, 并移除未完成的模型工作.

## 当前服务主线

Qwen 在一张 B200 上通过 Rust HTTP 服务, Rust 调度器和 Python worker 提供推理.
worker 使用 ZMQ DEALER 和 msgpack 边界.
服务支持 Chat 和 Responses API, 流式输出, 工具, 结构化输出, 媒体, prefix cache, MTP4 和自适应 CUDA Graph 选择.
显式 `stub` 后端只用于服务测试.

调度器依次选择 decode 工作, 完整 prefill block 和剩余 token.
缓存策略负责 Qwen FA 和 GDN 存储分配.
Rust 在发布结果前验证输出, 状态, 传输和 worker 生命周期.

## 下一步

- 共享运行时代码变更后, 运行 Qwen HTTP 和 AgentX 回归矩阵.
- 模型专属规则放在 `python/oh_my_vllm/models/qwen38/` 和 `crates/zmq-worker/src/models/qwen38/`.
- 原始报告, 主机设置和临时进程记录保留在仓库外.

使用[测试范围](testing.zh.md), [执行规则](execution.zh.md)和[交付流程](../CONTRIBUTING.zh.md).
