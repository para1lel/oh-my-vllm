# GPU 执行

## 算子调度

Rust 负责请求, 调度, 逻辑 KV 身份和 state commit.
Python 通过 CUDA 算子及现有 ZMQ DEALER 和 msgpack 边界执行模型.
[算子边界](operators/fusion.zh.md)规定计算和私有状态所有权.

Worker 为每条 Execute 消息选择一次执行方式.
只要任一已调度行包含 prefill, 该 Execute 的所有分片都使用 Eager 执行.
仅含 decode 的 Execute 使用 CUDA Graph 录制和重放.
此规则也控制采样, MTP context, proposal 和 acceptance.
Vision 和权重准备使用 Eager 执行.

Eager 和 CUDA Graph 执行使用相同的 native 算子.
Python 按模型依赖顺序调用这些算子.
CUDA kernel 可以融合操作并在内部使用矩阵 tile.

## Graph 资源

Workspace 拥有已录制操作的权重, cache 引用, metadata 和临时 tensor.
权重准备和 workspace 构造在录制前分配存储.
具有延迟 native 初始化的操作在录制前执行 warmup.
采样, MTP acceptance 和已准备的 draft chain 无需 warmup 即可录制.
录制不会执行这些操作的 RNG 更新.

Graph 存在期间, 缓冲区地址保持固定.
Metadata 在重放前于调用者的 stream 上更新.
重放使用同一个 stream.

录制使用独立的 nonblocking stream, 并在退出时恢复调用者的 stream.
录制失败会释放失败前创建的资源.
录制成功后, 资源 lease 转交给 workspace.
关闭时先等待 GPU 工作, 再销毁 graph executable, 最后释放其 owner 和 lease.
执行失败沿用现有 worker 移除和状态清理规则.

## 数值与状态契约

遵守[模型契约](operators/model.zh.md)及[全模型数值评估](operators/evaluation.zh.md).
保留[测试范围](testing.zh.md)中的服务, graph 资源, 内存安全和状态所有权检查.
