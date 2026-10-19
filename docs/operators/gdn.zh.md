# GDN 执行计划

使用[模型契约](model.zh.md)和[完整 serving 边界](fusion.zh.md).
Worker 分别提供 scan 和分块 prefill 路径.

## 选择依据

根据每条序列的 token 数, batch 大小及每个 CTA 的状态工作量选择算法.
一个 CTA 是一个 CUDA 线程块.
一个 value tile 包含某个状态 head 中连续的 value 行, 以及全部 128 个 key 元素.
独立 CTA 数量为 `batch * 48 * (128 / value_rows_per_tile)`.
仅凭 CTA 数量不能得出执行时间或 occupancy.

已建模的 recurrent 路径每条序列有 1 或 5 个 token.
已建模的 prefill 路径每条序列至少有 784 个 token.
分块路径选择最长序列超过 128 个 token 的 prefill batch.
序列长度可以不同. 空序列不产生计算块.
其他输入使用 scan 路径.

## 短序列

初始设计使用按 token 顺序更新, 并行处理 value tile 的方式.
只加载一次 BF16 状态. 跨 token 更新保持 FP32 live carry.
保存必要的私有 BF16 状态. 保持读取 slot 不变.

包含 1 或 2 条等长序列的 batch 对 2 至 5 个 token 使用固定 query 数量.
独立输入使用主机元数据检查. 交错布局的服务输入保留 GPU 元数据检查.
[门控辅助函数](../../python/oh_my_vllm/kernels/cuda_backend/gdn_recurrent_core.cuh#L137)<!-- source: __device__ inline float2 fast_gate_pair(float alog, float activation, float projected_b) { --> 在限定输入范围内使用快速路径.
其他输入使用完整计算. 单 token 路径和通用路径使用同一个辅助函数.

[FlashInfer BF16 实现](https://github.com/flashinfer-ai/flashinfer/blob/deca0e2959d521fe37efffdb6074cd357e836042/flashinfer/gdn_kernels/gdn_decode_bf16_state.py)提供此设计参考.
其选择规则使用状态 head 数量和 token 数量.
其单 token 路径和多 token 路径使用不同的选择规则.

项目形状使用以下初始候选:

| Batch | 每条序列的 token 数 | 每个 tile 的 value 行数 | CTA 数量 |
|---|---:|---:|---:|
| 1 | 1 或 5 | 16 | 384 |
| 4 | 1 或 5 | 32 | 768 |

这些候选来自源码规则, 不是项目性能结果.
当 batch 提供足够的并行工作量时, 每个 CTA 处理更多 value 行可以减少重复的 Q/K 和 gate 计算.
比较应包含 head norm, 输出 gating, quantization 及 projection.
单独的 norm 启动可能增加完整操作的耗时.

## 长序列

Prefill 初始设计使用 64-token 计算块.
[FlashQLA Blackwell 实现](https://github.com/QwenLM/FlashQLA/tree/5538c35f0df33ef804eacd90c77d9d9087c1d4e4/flash_qla/ops/gated_delta_rule/chunk/blackwell)提供此设计参考.
它求解块内三角系统, 并使用 Tensor Core 矩阵乘计算状态更新及输出.
它保持 FP32 live carry, 并使用 BF16 矩阵操作数.
实现前, 在模型契约中给出 prefill 操作数类型和舍入点.
该规范应使用固定版本的 vLLM prefill 源码作为依据.

在序列结束位置和检查点导出位置拆分计算块.
一个 784-token 区间包含十二个 64-token 块及一个 16-token 尾块.
导出 BF16 检查点后继续保持 FP32 live carry.

[Prepare kernel](../../python/oh_my_vllm/kernels/cuda_backend/gdn_wy_native_prepare.cuh#L60)<!-- source: __global__ __launch_bounds__(256, 4) void native_prepare_bf16_inverse_kernel( --> 计算因果 QK 系数, 并将其保存为 BF16.
Output kernel 读取此 buffer.
[Workspace 定义](../../python/oh_my_vllm/kernels/gdn.py#L23)<!-- source: def chunk_workspace(rows, batch, max_queries, verify): --> 将此 buffer 计入运行时分配和启动内存预留.

[三角逆实现](../../python/oh_my_vllm/kernels/cuda_backend/gdn_wy_bf16_inverse.cuh#L68)<!-- source: template <class Storage> __device__ inline void inverse_blocks(Storage &s) { --> 使用 FP32 求解 8 行对角块.
它读取 BF16 矩阵项, 并将逆矩阵块保存为 BF16.
BF16 分块矩阵乘使用 FP32 累加.

当 batch, head 及 value tile 提供的并行工作量不足时, 序列分区可以提供更多并行工作量.
选择前应计入初始状态准备及修正成本.
[FlashQLA 分区选择器](https://github.com/QwenLM/FlashQLA/blob/5538c35f0df33ef804eacd90c77d9d9087c1d4e4/flash_qla/ops/gated_delta_rule/chunk/cp/autocp/decision.py)会将这些成本与不使用分区的执行方式进行比较.
不得通过丢弃传入状态或更早的 token 来缩短执行时间.

## 工作量计算

Scan 和分块工作量使用共享的 Python 选择规则.
在 64-token 计算边界, checkpoint 导出位置和序列末尾划分真实 token 行.
不计入填充行或分组 Q/K head 的重复工作.

对一个计算块, 令 `m` 为真实行数, `D=128`, `H=48`, `G=16`.
三角逆以外的 BF16 矩阵工作量为 `6*H*D*D*m + H*D*m*(3*m-1) + 2*G*D*m*m` FLOPs.
其中包含状态投影, 状态更新, 输出, 三角 W/U 乘积, KK 和因果 QK.
W/U 计数不包含单位对角线对应的复制.

FP32 计数包含 live state 更新, 对角块求逆, gate 缩放和输出合并.
对于三角逆, 令 `T=m*(m-1)*(m-2)/3`, 对 `i=0..7` 令 `d_i=min(8,max(0,m-8*i))`.
对角块工作量为 `F=sum_i d_i*(d_i-1)*(d_i-2)/3`.
三角逆贡献 `H*F` FP32 FLOPs 和 `H*(T-F)` BF16 FLOPs.
这些计数不包含与结构性零值和单位值的乘积, 也不包含填充行.
Scan 路径的工作量模型保留原有 FP32 逐 token 递推计数.

## 实现门槛

使用 CUDA C++ 或 CuTe DSL. 外部源码用于设计参考.
替代实现通过数值检查及完整边界比较前, 保持当前正确路径.

使用现有建模形状选择项目参数. 仅在选择问题确有需要时增加中间长度.
Eager 和 graph 计时应包含所有启动, 状态导出, 临时存储及尾块工作.
每次比较期间保持负载概率和 roofline 计算不变.
