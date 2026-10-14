# GPU 测量方法

分别采集框架墙钟延迟, 设备时序图和硬件计数器.
第一项用于验收. 另外两项用于解释行为.
参见 [正式协议](../requirements/performance.zh.md) 和 [命令](../testing/performance.zh.md).

## 确定工作负载

记录源码, 已加载模块, 编译器, checkpoint, 形状, 进程出生身份和 GPU 身份.
计时前完成编译和图捕获. 比较之间将持久状态恢复到基线.
测量完整算子, 包括转换, 复制和归约.
完整框架采集之前, 交错测量候选和基线的配对样本.

合成输入可能改变数值分支. 选择特化路径前, 记录模型输入对应的分支频率.
[夹具](../../development/kernels/fixtures.py) 提供可复现的算子输入, 不代表模型输入分布.

## 阅读时序图

对小规模诊断工作负载使用 [有界 Torch profiler](../profiling.zh.md#小窗口-torch-profiler) 或 Nsight Systems.
通过关联信息和流身份, 将 CPU launch 和图节点连接到设备区间.
CPU 标注区间可能在其 GPU 工作结束之前结束. 不要只按 CPU 区间筛选 kernel.
[Nsight Systems 指南](https://docs.nvidia.com/nsight-systems/AnalysisGuide/) 解释排队时间和 GPU 执行时间.

PDL 允许有依赖关系的 kernel 重叠, 包括同一流中的 kernel.
区间可能包含依赖等待. 相加 kernel 时长因此可能重复计算同一墙钟区间.
区间并集表示被覆盖的 GPU 时间, 不代表依赖关键路径或排除等待的计算时间.
使用已完成请求的区间计算阶段延迟. 参见 [CUDA PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

## 单独采集计数器

从少量计数器开始. 记录 profiler 版本, 重放模式, 采集轮数, 时钟策略和缓存策略.
Nsight Compute 可能串行化 launch, 并重放 kernel 以采集计数器.
重放轮数不是独立计时样本. 缓存恢复和时钟控制可能改变工作负载.

如果执行必须依赖并发, 使用适当的 range replay 模式.
改变重放策略之前, 阅读 [Nsight Compute profiling 指南](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

已测试命令使用 `--cache-control none --clock-control none`.
该配置不控制时钟和缓存. 它不能证明各轮缓存内容相同或时钟固定.
profiler 时长只用于诊断. 不要同时运行 Torch profiler 和 Nsight Compute.

| 问题 | 观测项 |
| --- | --- |
| 独立工作是否足够? | 总 warp 数, grid 大小, waves 和工作分布 |
| 资源限制是什么? | 寄存器, 共享内存, 驻留 block 和 spill |
| 指令是否就绪? | eligible warp, issue 活跃度和依赖 stall |
| 哪种计算资源? | Tensor, FP32 和特殊函数管线活跃度 |
| 哪种内存资源? | HBM/L2 字节数, 带宽, 共享内存流量和冲突 |

保留计数器的单位和分母. 按指令归一化的 stall 比率不是墙钟时间百分比.
SM throughput 是管线指标, 不代表实测 FLOPS 除以模型峰值.
仅凭 occupancy 无法确定延迟原因. 参见 [NVIDIA compute triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/).
说明采集没有提供哪些计数器.

## 已复现的观测

[绑定源码的诊断摘要](../evidence/optimization-observations/verification.json) 记录原始报告和输入范围.
Nsight Compute 2026.1.1 采集 32144-token GDN 夹具一次 scan 的计数器, 使用 16 轮重放采集.
其 grid 有 192 个 block 和 768 个 warp, GPU 有 148 个 SM. 共享内存允许每 SM 驻留两个 block.

报告的 0.65 waves 等于 `192 / (148 * 2)`. 它不表示 35% 的 SM 空闲.
平均每 SM 有 5.251 个 active warp, occupancy 为 8.205%, register spill 为零.
本次调用的 Tensor 活跃度为 0.815%. 未采集分支频率.

如果不增加独立 warp 总数, 改变 block 数不能证明延迟隐藏改善.
较小 block 可能改善工作分布或共享内存驻留. 需要配对测量检验该假设.
profiler 的 12.456-ms 时长不是正式算子计时.

融合 normalization/FP8 gate-up 夹具有 32144 个 token, 宽度 5120, 输出列数 34816.
其中一个 CUTLASS kernel 的 Tensor Core 活跃度为 57.649%, FMA 管线活跃度为 31.319%, 使用 18 轮重放采集.
其 profiler 时长 5.457-ms 不包含算子的其他 kernel.
将状态 scan 依赖和 block-scale 累加作为独立的调研方向检查.
这些观测不能证明完整框架加速.

### 模型输入分支

一次独立的带观测模型运行使用一个请求, 将 token 42 重复 32768 次, 并手动提供缓存表.
它在各 GDN 层输入上重算当前分类器, 不改变原始 prefill 结果.
捕获和重放后, 48 层中的 46 层全部 chunk 使用精度路径.
48 层的 chunk/head 对计数为精度路径 1153392, constant-key 路径 26256, 矩阵路径为零.
同步之后才读取捕获的观测 buffer.

这只是 ordinary 模型的一种输入, 不代表 prompt 分布或正式调度器采集.
未采集 checkpoint 字节哈希和物理层 ID.
运行包含捕获和观测成本. 其时长不是验收计时.
用这些分支计数确定精度路径调研的优先级, 不扩大输入范围.

## 选择修改

说明一个假设, 以及提供依据的计数器或依赖关系.
保持数值, 舍入, 状态和重放合同不变. 缺少等价证据时, 拒绝更快的候选.
比较完整链路, 然后关闭 profiling, 重复适用的框架行.
保留失败尝试和源码身份. 每次采集后停止任务拥有的 GPU 进程.
