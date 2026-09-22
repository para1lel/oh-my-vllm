# ADR-004 — Rust 优先的 OpenAI 兼容服务

**日期：** 2026-09-21
**状态：** 已接受、已实现，真实 MTP4 验收已通过

## 背景

用户希望本地 oh-my-pi 通过 HTTP 调用引擎，完成 agentic 仓库介绍任务。此前 HTTP 不在项目范围内。

## 决策

同时支持 OpenAI Chat Completions 和 Responses，包括流式工具调用和多轮工具结果。外部入口和生命周期优先由 Rust 负责。保留 Rust 调度器/KV 所有权、Python GPUWorker、单张 B200、块大小 784 及现有 ZMQ/msgpack 边界。

支持可配置思考强度、无状态历史重放和存储型 Responses 续接。存储响应位于有界、可过期的内存中，无需重启持久化。使用真实 oh-my-pi 运行测试两种 API。

## 影响

HTTP 成为已实现范围。服务添加在线请求接纳、流式响应、EOS/stop 处理和取消传播。模型专有的工具/思考输出需转换成各 API 的协议表示。API 兼容细节和思考映射见 serving.md；本决策不承诺全部 OpenAI 平台能力。

用户随后授权实现：默认 medium，high→xhigh，兼容 MTP 的 token 约束，以及检查性能但不增加 HTTP 吞吐门槛。Axum 和引擎 actor 拥有请求生命周期；Python 只负责本地模板准备和 XGrammar mask 状态。存储 Responses 默认 TTL 一小时、最多 1,000 条记录和 256 MiB 序列化载荷，不持久化。不支持的协议/schema 功能显式失败。支持子集见 [serving.md](../serving.zh.md)，验收证据见 [handoff.md](../handoff.zh.md)。
