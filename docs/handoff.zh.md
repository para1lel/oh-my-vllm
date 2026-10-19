# 工作状态

在 `main` 上工作.
使用 [GPU 执行规则](execution.zh.md).
Worker 使用传统算子调度及历史实测实现中的 native kernel.
包含 prefill 的 Execute 对所有分片使用 Eager 执行.
仅含 decode 的 Execute 录制并重放 CUDA Graph.
Tile 调度器, task plan 和协议 fixture 已移除.

Rust 提供 HTTP, 调度, 逻辑 KV 和 worker 生命周期控制.
Python 通过 ZMQ DEALER 和 msgpack 提供模型执行.
Graph 重放使用调用者的 stream.
Graph 关闭在释放资源前等待 GPU 工作.
录制失败会释放临时资源 lease.

构建入口是 `development.native_build`.
[数值工具](operators/evaluation.zh.md)将短输入, 单次迭代的全模型 logits 与 PyTorch FP64 计算比较.
保留冻结的参考及其身份检查.
单算子容差测试及其参考 provider 已移除.
旧算子计时, 公开比较和峰值校准工具也已移除.

性能工作使用 24 种 [HTTP 配置](performance.zh.md)及标称资源计费.
历史测量与当前源码的证据分开保存.
只有完整 HTTP 测量支持变更时, 才优化 native 算子.
保持数值精度, 状态所有权和服务行为不变.
变更影响 MTP, vision 或长上下文时, 扩展全模型输入.

Host 路径和 artifact 位置保存在被忽略的本地配置中.
