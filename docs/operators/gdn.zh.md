# GDN 执行计划

使用[模型契约](model.zh.md)和[完整 serving 边界](fusion.zh.md).
本计划不改变当前 CUDA 实现.

## 选择依据

根据每条序列的 token 数, batch 大小及每个 CTA 的状态工作量选择算法.
一个 CTA 是一个 CUDA 线程块.
一个 value tile 包含某个状态 head 中连续的 value 行, 以及全部 128 个 key 元素.
独立 CTA 数量为 `batch * 48 * (128 / value_rows_per_tile)`.
仅凭 CTA 数量不能得出执行时间或 occupancy.

已建模的 recurrent 路径每条序列有 1 或 5 个 token.
已建模的 prefill 路径每条序列至少有 784 个 token.
该模型不提供长度为 6 到 783 时的算法比较依据.

## 短序列

初始设计使用按 token 顺序更新, 并行处理 value tile 的方式.
只加载一次 BF16 状态. 跨 token 更新保持 FP32 live carry.
保存必要的私有 BF16 状态. 保持读取 slot 不变.

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

当 batch, head 及 value tile 提供的并行工作量不足时, 序列分区可以提供更多并行工作量.
选择前应计入初始状态准备及修正成本.
[FlashQLA 分区选择器](https://github.com/QwenLM/FlashQLA/blob/5538c35f0df33ef804eacd90c77d9d9087c1d4e4/flash_qla/ops/gated_delta_rule/chunk/cp/autocp/decision.py)会将这些成本与不使用分区的执行方式进行比较.
不得通过丢弃传入状态或更早的 token 来缩短执行时间.

## 实现门槛

使用 CUDA C++ 或 CuTe DSL. 外部源码用于设计参考.
替代实现通过数值检查及完整边界比较前, 保持当前正确路径.

使用现有建模形状选择项目参数. 仅在选择问题确有需要时增加中间长度.
Eager 和 graph 计时应包含所有启动, 状态导出, 临时存储及尾块工作.
每次比较期间保持负载概率和 roofline 计算不变.
