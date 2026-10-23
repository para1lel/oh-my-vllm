# 原生集合通信所有者

## 范围与设置

[Python API](../../python/oh_my_vllm/operators/runtime/collectives.py#L76) <!-- source: class Communicator: --> 为[集合通信计划](collective-plan.zh.md)提供原生传输.
Rust 控制 bootstrap, 操作顺序, rank 截止时间和设备组终止.
此传输不启用 DeepSeek 模型执行.
仅导入 API 不会加载 NCCL.
可选原生模块使用仓库的 TVM FFI ABI3 加载器.

每个 rank 进程必须只看见一个 CUDA 设备, 其本地 ordinal 为 0.
`local_device()` 返回 CUDA 设备数量和 UUID.
Worker 必须在 bootstrap 之前将此 UUID 与预期的设备组 UUID 比较.
`versions()` 返回 NCCL 头文件和已加载库的版本代码.
`unique_id()` 返回 128 个不透明字节.
Supervisor 向所有 rank 发送相同的 ID.

在捕获之前构造 `Communicator(world, rank, unique_id)`.
在外部截止时间内轮询 `comm.poll()` 至 `"ready"`.
初始化使用已安装的 NCCL 配置和非阻塞模式.
相关工作使用一个控制线程及调用方 stream.

原生调用没有内部等待循环.
库调用仍可能超过截止时间.
Supervisor 必须具备终止进程的路径.

## 逻辑缓冲区与私有存储

`comm.prepare(kind, source, destination)` 接受 `all_reduce`, `all_gather` 和 `reduce_scatter`.
归约采用求和.
`comm.routing(routes)` 接受有序的 `Route("send", peer, tensor)` 和 `Route("recv", peer, tensor)` 条目.
零元素条目保持其位置和匹配调用.

支持的类型为 `float32`, `bfloat16`, `float16`, `int32`, `int64` 和 `uint8`.
提供位于 `cuda:0` 的连续一维拥有存储所有权的张量.
原生检查拒绝错误的数量, 类型, 步幅, 设备, 边界及存储重叠.
经过检查的视图地址可以不按 16 字节对齐.
远端数量一致性由规范计划负责.

[存储检查](../../python/oh_my_vllm/operators/runtime/cuda/collective_contracts.cuh#L67) <!-- source: inline int64_t collective_staging_bytes( --> 考虑 NCCL LL 和 LL128 对包围逻辑数据的字节区间的读取.
非空缓冲区仅在地址和字节长度均按 16 字节对齐时使用直接路径.
其他非空缓冲区使用私有存储, 容量经过检查并向上取整到 16 字节.
分配地址按 16 字节对齐.
原生所有者将整个私有分配初始化为零.
零元素缓冲区不使用私有存储.

现有 Device 分配导出函数提供私有存储.
Device 分配统计包含此存储.
[准备好的所有者](../../python/oh_my_vllm/operators/runtime/cuda/collective_owner.cuh#L42) <!-- source: struct CollectivePlanState { --> 持有调用方张量和私有张量.
它在 NCCL 提交之前复制逻辑输入字节.
它在主机提交完成之后复制逻辑输出字节.

填充不改变线上数量, 张量形状, 路由顺序或规范摘要.
存储地址和复制大小在捕获之前固定.
每次重放都会再次执行输入复制.

## Eager 与捕获完成

Eager 执行先调用 `prepared.run()`.
读取结果或复用缓冲区之前, 轮询 `comm.poll()` 至 `"ready"`.
此轮询在调用 CUDA 之前检查 NCCL 状态.
主机就绪后, 它提交输出复制和完成事件.
Eager 的 `"ready"` 证明包括这些复制在内的设备工作已经完成.

捕获之前完成 Eager 预热.
为每个操作调用一次 `prepared.prepare_capture()`.
所有 rank 按相同顺序捕获操作.
对每个操作使用以下顺序:

```python
try:
    prepared.run()
    while True:
        status = comm.poll_enqueue()
        if status == "ready":
            break
        if status != "pending":
            raise RuntimeError(status)
        check_external_deadline()
    prepared.finish_enqueue()
except BaseException as error:
    try:
        report_error_with_deadline(error)
    finally:
        os._exit(1)
```

`poll_enqueue()` 只检查 NCCL 主机进度.
它不调用 CUDA, 也不证明设备完成.
`finish_enqueue()` 在原始捕获 stream 上提交输出复制.
它没有等待循环.
此收尾过程完成之前, 下一个操作会被拒绝.

捕获回调返回之前, 必须完成最后一个操作的收尾.
捕获期间不要使用 `comm.poll()`.

重放之后, 在调用方 stream 上调用 `comm.record_completion()`.
读取结果, 复用缓冲区或发送 RankResult 之前, 轮询 `comm.poll()` 至 `"ready"`.
在整个图上下文期间持有完整 workspace 和准备好的对象.

## 故障与关闭

在捕获回调内部处理捕获提交, 轮询和收尾错误.
发生错误, 超时或取消时, 只尝试有时间上限的 CPU/IPC Error 报告.
然后通过 `os._exit(nonzero)` 终止 rank.
不要返回或通过 Device 捕获清理路径抛出异常.
在此路径上不要调用 CUDA, abort 或释放缓冲区.

如果报告或原生代码停滞, supervisor 截止时间仍须有效.
Rust 必须终止并回收设备组的所有 rank, 然后才能释放设备组租约.

正常关闭时, 销毁源图和所有图可执行体.
在截止时间内轮询 `comm.abort_ready()` 至 true.
False 表示图租约仍然存活.
此查询只读取回调标志.
然后调用 `comm.abort()`, 再轮询至 `"aborted"`.

Abort 之后, 此轮询还会等待未使用的捕获预留释放.
此顺序完成之后, 释放准备好的对象和 workspace 张量.

图回调只设置完成标志.
调用线程上的清理释放所有者.
未完成或失败的工作持有其所有者, 直到显式清理或进程退出.
强制退出不能作为正常退出的 Compute Sanitizer 证据.
