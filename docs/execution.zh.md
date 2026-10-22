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

## 依赖 kernel 启动

小规模 FFN 和 GDN 操作在归一化与输入投影之间使用 [Programmatic Dependent Launch](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).
它们也在输出量化与输出投影之间使用 PDL.
65 到 1024 行的 FFN prefill 在这两个边界使用 PDL.
其融合输入投影包括激活和输出量化.
更大的 prefill 操作保留串行启动.

Stream capture 将这些选定的依赖记录在 CUDA Graph 中.
Consumer 可以在 producer 完成执行前准备自身存储并读取固定权重.
读取 producer 提供的输入或 scale 前, consumer 等待 producer 完成执行.

启动信号允许 consumer 启动. Consumer 读取 producer 的输出前必须等待.

在输出合并和层边界, 一个 kernel 必须完成执行后下一个 kernel 才能启动.
这些规则阻止在前序 kernel 完成执行前复用临时存储.
PDL 不改变状态选择或 Eager 与 Graph 执行规则.

## Graph 资源

Workspace 拥有已录制操作的权重, cache 引用, metadata 和临时 tensor.
权重准备和 workspace 构造在录制前分配存储.
具有延迟 native 初始化的操作在录制前执行 warmup.
采样, MTP acceptance 和已准备的 draft chain 无需 warmup 即可录制.
录制不会执行这些操作的 RNG 更新.

Graph 存在期间, 缓冲区地址保持固定.
Metadata 在重放前于调用者的 stream 上更新.
重放使用同一个 stream.

Metadata 和采样输入使用可复用的页锁定 host 缓冲区上传.
每次上传在返回前复制一份 CPU 输入.
调用者 stream 上的事件保护这份副本, 直到传输完成.
复用或释放存储前, 必须等待该事件.
不要在 Graph 录制期间更新这些输入.

Device 在有容量上限的池中保留已完成传输的 host 上传缓冲区.
新缓冲区使用容量分级. 事件完成后, 变更后的 workspace 可以复用这些缓冲区.
Device metadata 使用按 stream 顺序管理的临时存储池.

MTP context graph 拥有固定的隐藏状态输入缓冲区.
Worker 在同一 stream 上重放前, 将 target 隐藏状态复制到这些缓冲区.
Context 复用取决于算子 plan, 不取决于 target workspace 地址.

录制使用独立的 nonblocking stream, 并在退出时恢复调用者的 stream.
录制失败会释放失败前创建的资源.
录制成功后, 资源 lease 转交给 workspace.
关闭时先等待 GPU 工作, 再销毁 graph executable, 最后释放其 owner 和 lease.
执行失败沿用现有 worker 移除和状态清理规则.

## 数值与状态契约

遵守[模型契约](operators/model.zh.md)及[全模型数值评估](operators/evaluation.zh.md).
保留[测试范围](testing.zh.md)中的服务, graph 资源, 内存安全和状态所有权检查.
