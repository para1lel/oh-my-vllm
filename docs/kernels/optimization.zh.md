# 算法调研和实测经验

框架延迟门槛仍待通过. 外部 benchmark 不能确定哪个算法最适合本项目的合同.
参见 [测量方法](../profiling/measurement.zh.md), [数值合同](../requirements/kernels.zh.md) 和 [诊断证据](../evidence/optimization-observations/verification.json).
调研快照和临时探针保存在仓库外. 不要安装外部 Python 算子提供者.

## Gated DeltaNet

递推由衰减状态和 recalled value 产生的秩一修正组成.
逐 token scan 存在顺序状态依赖. 增加线程不能消除该依赖.
[Gated DeltaNet 论文](https://arxiv.org/html/2412.06464v3) 给出用于 chunk 并行矩阵计算的扩展 WY 表示.
[FLA 的 WY 源码](https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/ops/gated_delta_rule/wy_fast.py) 是具体的算法参考.
仓库 [chunk CUDA](../../python/oh_my_vllm/kernels/cuda_backend/operators/gdn_chunk.cuh) 和 [精度 scan](../../python/oh_my_vllm/kernels/cuda_backend/operators/gdn_precision.cuh) 提供当前实现.

[FlashQLA](https://github.com/QwenLM/FlashQLA/tree/da06429d54b0f577de0a638f451ac8f0b395e0ac) 提供 Blackwell 调度和上下文并行状态传递.
其公开计时在指定输入上比较不同算法. 它们不是本项目的测量或数值准入证据.
研究矩阵暂存和依赖重叠. CUDA 移植前检查中间精度, 抵消, 状态传递和 BF16 边界.

仅凭 decay 不能证明移除初始状态符合相对误差容差.
当 beta 和 values 为零时, 精确最终状态为 `exp(sum_decay) * initial_state`.
以零状态开始的截断返回零. 如果参考范数超过比较 floor, 相对误差为 100%.
即使状态绝对值很小, 该结论仍成立.

FlashQLA 的 [decay 相关上下文路径](https://github.com/QwenLM/FlashQLA/blob/da06429d54b0f577de0a638f451ac8f0b395e0ac/flash_qla/ops/gated_delta_rule/chunk/blackwell/cp_fwd.py) 可能在 decay 截断后省略初始状态修正.
零 decay 保留完整传递修正. 将截断视为未经验证的数值修改, 而不是只改变调度的优化.

## Block-scaled FP8

本项目保留 checkpoint 和激活的 FP32 block scale.
[CUTLASS Blackwell blockwise 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/81_blackwell_gemm_blockwise/README.md) 提供对 FP32 累加器进行软件缩放.
它还说明如何为窄矩阵选择转置操作数.
这些是 [自有 GEMM 实现](../../python/oh_my_vllm/kernels/cuda_backend/groupwise_fp8.cu) 可用的原生 CUDA 方向.
将 scale 工作, 转换和最终 epilogue 与矩阵乘一起测量.

[DeepGEMM SM100 接口](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/README.md) 使用 packed UE8M0 scale.
其二次幂 scale 转换不等价于任意 FP32 缩放.
不要复制该数值路径或改变 checkpoint 权重. persistent 调度和数据暂存是独立的调研方向.

## Attention

[FlashAttention-4](https://arxiv.org/html/2603.05451v1) 研究 Blackwell 的 Tensor Core, 共享内存和指数吞吐量增长不均衡的问题.
其流水线设计连接这些资源, 而不是假设仅矩阵吞吐量决定延迟.
使用该资源模型检查 [Torch SDPA prefill](../../python/oh_my_vllm/kernels/ragged_attention.py) 和 [仓库 CUDA decode](../../python/oh_my_vllm/kernels/cuda_backend/operators/decode_attention.cuh).
近似指数或省略重缩放需要独立数值证明. 论文结果不能证明项目舍入边界符合要求.

## 本地实验

[诊断摘要](../evidence/optimization-observations/verification.json) 绑定输入, 源码, 原始哈希, 样本和限制.
这些实验没有选出新的生产实现.
后续 [仿射扫描诊断](affine-diagnostics.zh.md) 记录矩阵布局收益, 完整仿真成本和最终状态抵消失败.

### 原生 FP64 仿真

隔离的 cuBLAS handle 以有限随机 FP64 输入和 CPU FP64 参考, 对七个矩阵形状测试三种模式.
全部 21 项比较通过探针容差. 这不能证明完整 GDN 行为或极端输入符合要求.
Eager 仿真约耗时 108–1050 微秒, native 模式约耗时 3–26 微秒.
采样的 performant-mode 时序图使用 native FP64 kernel.
其他形状缺少该时序图证明.

[cuBLAS 仿真指南](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html#fixed-point) 解释 mantissa 选择和 workspace 限制.
未改变的输出 sentinel 不能标识实际执行路径.
在所测小矩阵上, 完整仿真链更慢, 未被选用.
该结果不排除较大矩阵使用仿真.

### 三层 wavefront

一项诊断在三个 GDN 模型层中比较完整层执行, 串行 token 切片和并行切片.
32768 个 token 的每种配置的三十个交错样本给出完整层执行 84.250 ms, 并行切片 79.456 ms.
这是 5.69% 的局部计时改善, 不是完整框架测量或正式配对协议.

串行和并行切片相对完整层执行具有相同的已记录误差摘要.
hidden 输出 NRMSE 为 1.790%, 最后一个 convolution 状态不逐位相等.
这不能确定存在并发 race 或正式 GDN 状态失败.
完整等价性未经证明. 未选用该候选.
测试并发调度之前先解决串行切片等价性.

### 当前硬件观测

scan 夹具的 Tensor Core 活跃度和 occupancy 低, register spill 为零.
FP8 夹具有已记录的 Tensor Core 和 FMA 活跃度.
[计数器记录](../profiling/measurement.zh.md#已复现的观测) 给出单位, 形状和范围.
低 occupancy 不能证明较小 block 会改善延迟.
调优特定分支路径之前, 记录模型输入的数值分支频率.

## 后续实验

1. 记录模型输入的路径频率, 在源码和设备指令中定位 scan 依赖 stall.
2. 对专用精度调度与当前完整算子进行比较. 保持递推, 归约顺序和状态边界相同.
3. 检查 FP32 scale 累加和窄矩阵调度, 不转换为 UE8M0.
4. 集成前, 用抵消, 非零初态和零 decay 案例检查 WY/状态传递算法.

选用前执行正式数值矩阵和适用的重放测试.
选用后重复框架测量. 保持方程, 容差和两倍门槛相同.
