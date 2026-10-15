# 算子评测

使用[模型契约](model.zh.md)及 [fusion 计划](fusion.zh.md).
[负载模型](../../development/kernels/workload.py)提供六个请求模板.
验收使用其中的 7 个 serving 边界. 不对 18 个内部阶段应用门槛.

原生 provider 提供这些 serving 边界. 完整负载验收是必要的.
可用工具见[开发命令](development.zh.md).

## Shape 分布

| 模板 | 权重 | Batch | 文本输入 | 输出 | 模式 | 媒体 |
|---|---:|---:|---:|---:|---|---|
| 短文本 | 0.20 | 1 | 784 | 128 | 普通 | 无 |
| 文本批次 | 0.20 | 4 | 32768 | 256 | 普通 | 无 |
| 长文本 | 0.20 | 1 | 131072 | 512 | MTP4 | 无 |
| 图片 | 0.15 | 1 | 1024 | 256 | 普通 | 448x448 |
| 图片批次 | 0.15 | 4 | 4096 | 256 | MTP4 | 896x896 |
| 视频 | 0.10 | 1 | 1024 | 128 | MTP4 | 8 帧, 448x448 |

权重是简化请求模型, 不是真实生产流量的测量值.
模型使用等长输入的 batch 和均衡 prefill 分块.

MTP 每个请求有五行验证输入. 在模型中, 每个 MTP 步骤保留四个 token.
其中接受 3 个 proposal, 并拒绝第 4 个 proposal.
拒绝采样提供 1 个 token.
这只是简单的负载模型, 不改变 MTP 接受流程.

每个模板的 decode 只使用一个上下文, 取其输出序列的中点.
其长度为 `input_tokens + output_tokens // 2`.

根据请求权重, 层数及调用次数计算算子概率.
每个算子分别归一化概率.

仅在执行 shape 相同时合并 case.
不在性能分布中使用边界检查.
对 MTP 接受分支及其他边界条件做数值检查, 不赋予性能权重.

## 性能门槛

使用单 GPU dense B200 峰值.
Tensor Core 峰值为 FP8 4.5 PFLOPS, BF16 2.25 PFLOPS 及 TF32 1.125 PFLOPS.
Scalar 峰值为 FP32 75 TFLOPS 和 FP64 37 TFLOPS. HBM 带宽为 7.7 TB/s.
[NVIDIA B200 参考](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/_downloads/4b3d3302fa4fc6854b595ccff50b9932/RA11334001-DSPB200-ReferenceArch.pdf)标识平台.

[NVIDIA HGX 规格](https://www.nvidia.com/en-au/data-center/hgx/)提供八 GPU 的 FP64 及 sparse TF32 数据.
将这些数据除以 8 得到单 GPU 数据. 将 sparse TF32 数据除以 2 得到 dense 计算数据.
使用[中间精度契约](model.zh.md#中间精度).
分别应用 FP32 和 FP64 上限. 没有经过验证的共享资源模型时, 不将两者时间相加.

每个完整算子使用以下工作量模型:

```text
t_lower = max(
    necessary_HBM_bytes / HBM_peak,
    FP8_work / FP8_peak + BF16_work / BF16_peak + TF32_work / TF32_peak,
    FP32_work / FP32_peak,
    FP64_work / FP64_peak
)
eta(shape) = t_lower(shape) / measured_seconds(shape)
E_eta(operator) = sum(probability(shape | operator) * eta(shape))
PASS(operator) = accuracy_pass AND complete_coverage AND E_eta(operator) >= 0.60
```

使用[工作量计算](../../development/kernels/work.py)得到每个 serving 边界的必要计算量及 HBM 字节数.
从跨越 serving 边界的张量及必要的持久状态访问推导 HBM 字节数.
计入必要的冷权重读取. 不累加内部阶段之间张量的 HBM 字节数.

将全部临时存储访问包含在计时操作中.
接受操作选择状态 slots, 不复制完整的 recurrent 或卷积状态.

FP8, BF16 和 TF32 共享 Tensor Core 资源. 不将它们的限制按独立资源计算.
Padding 和不必要的复制不能抬高理论下界.

不在理论下界中加入实测启动耗时或经验效率因子.
工作量模型不接收实测或校准延迟参数.
保留各 shape 的耗时及效率. 不以总耗时之比替代期望效率.

缺少实现, 精度失败, case 缺失或混用源码标识都会阻止验收.
报告包含最大归一化容差比. 该比值必须不大于 1. 必要的保护检查必须通过.
效率超过 1.05 时必须检查工作量模型.

如果 provider 缺失或失败, 以错误结束评测.
不用参考实现替代它.

## 测量

通过 GPU wrapper 获取一张空闲 B200.
测量前构建实现.
测量前准备缓冲区.

计时前检查精度. 所有运行之间恢复可变状态.
在调用者的 stream 上, 在完整操作前后使用 CUDA events.
Eager 和 CUDA Graph 计时包含全部必要 launches.

分别测量 eager 执行与 CUDA Graph replay.
执行两次 warmup 运行和五次正式测量运行.

HBM roofline 验收使用冷权重.
每次计时调用前, 使用大于 B200 L2 容量 2 倍的 buffer 运行 L2 eviction.
在调用者的 stream 上将此 eviction 放在开始事件之前.
计时长度不包含 eviction.

完整模型的权重工作集大于 L2.
冷权重策略适用于这个条件.
记录 eviction 方法, buffer 大小及 L2 容量.
不将权重已热的独立运行用作 HBM roofline 验收证据.
Warmup 运行不免除冷权重规则.

记录中位数和波动范围. 所有测量保存在仓库外的位置.
失败记录保存在相同位置.
包含源码, 二进制, 编译器, 依赖, device, clock 及 shape 标识.

使用 Nsight Systems 定位高耗时 launch 和 GPU 空隙.
精度通过后对选定 launch 使用 Nsight Compute.

采集耗时, Tensor Core 吞吐, HBM 流量, cache 命中率, occupancy, registers, spills 及 warp stalls.
不可用的 counter 使用 "unavailable" 标签.
Counter 权限不足时也使用这个标签.
不把缺失的 counter 记录为零.

不将 profiling 运行用于验收计时.
每次运行后停止自己启动的进程.
