# 工作状态

Rust 提供 HTTP API, CLI, 调度器, 逻辑 KV cache 和 worker 生命周期控制.
服务模式为普通解码和 MTP4.
HTTP 输入可包含文本, 图片和视频.

Python worker 是占位实现, 不加载模型权重, 不使用 GPU.
Rust 通过推理 worker 的进程和协议边界发送媒体引用.

[算子契约](operators/model.zh.md)包含 18 个内部阶段及 7 个计划中的 serving 边界.
负载包含六类请求模板, 每类使用一个 decode context.

[算子开发工具](operators/development.zh.md)提供 CUDA C++ 和 CUTLASS FFN 基础版本.
构建工具与推理加载器分开.
其余 serving provider 尚不可用.

完整算子实现尚未完成. 60% 性能要求仍有工作待完成.
下一步: 优化 FFN, 添加其余 provider, 并将模型计算接入 worker.
每个 provider 都要有数值检查和完整边界测量.

工作使用 [main 和 Issue 分支](collaboration.zh.md).
主机路径和产物位置保存在被忽略的本地配置中.
保持[运行时契约](requirements.zh.md)及[进程边界](architecture.zh.md).

所有英文文档继续遵循 ASD-STE100. 不在本文档中记录测试结果.
