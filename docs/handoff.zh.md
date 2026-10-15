# 工作状态

Rust 服务已有 HTTP API, CLI, 调度器, 逻辑 KV cache 和 worker 生命周期控制.
Python 占位实现与后续推理 worker 使用相同的进程及协议边界.

不加载模型权重, 不使用 GPU.

下一步: 在 Python worker 中添加模型计算.
保持[运行时契约](requirements.zh.md)及[进程边界](architecture.zh.md).

所有英文文档继续遵循 ASD-STE100. 不在本文档中记录测试结果.
