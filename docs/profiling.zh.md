# Profiling 与诊断

通过 profiling 解释已测时延. 正式计时按 [性能契约](requirements/performance.zh.md) 关闭 profiling.
原始 trace 放在外部存储, 绑定源码的摘要放在 [验收索引](acceptance.zh.md).

## 主机与设备时间

[Worker 日志](../python/oh_my_vllm/worker/logging_utils.py) 记录请求, step 和主机耗时.
[执行记录](../python/oh_my_vllm/performance/execution.py) 将模型工作连接到理论 DAG.
使用 CUDA Events 或 profiler trace 归因 GPU 时间. 主机 enqueue 时间不等于 GPU 执行时间.
并发 kernel 的耗时不能相加作为 wall latency. 记录 stream, 重叠和依赖.

## 稳定测量

测量重复中不包含编译, graph capture 或 cache 淘汰.
采集前后检查源码, 已加载模块, checkpoint, 环境和 GPU 身份.
保留失败与中断尝试. 报告不可用计数器.

[Capture 事务](../python/oh_my_vllm/worker/graph_capture.py#L31) <!-- source: def capture_graph( -->
保持 graph 存储与 workspace 寿命一致. [运行时身份](../python/oh_my_vllm/worker/runtime.py) 记录已加载实现.

## 算子调优

测量完整算子, 包含必需的复制, reduction, scale 打包和输出转换.
交错比较 candidate 与仓库 baseline. 先检查数值和 mutation 契约.
profiling 与计时分开, 选用实现后检查完整框架.
融合会改变资源压力和重叠. 根据整体收益的测量选用.

使用 [算子 harness](../development/kernels/) 和 [内核指南](kernels.zh.md).
实现的调度与阈值放在 [CUDA 源码](../python/oh_my_vllm/kernels/cuda_backend/) 中, 以选中 candidate 的证据为依据.

## 内存与故障

使用 [worker 诊断](../python/oh_my_vllm/worker/model_runner.py) 和 [graph 所有权](architecture/compilation.zh.md) 做内存归因.
RSS, allocator 统计, tensor 字节数和 CUDA reserved 字节数衡量不同量.
一小时观察不能证明内存平台或该小时之后的稳定性.
allocator OOM 恢复与稳定服务采集分开运行. profiling 或失败后释放自有进程.
