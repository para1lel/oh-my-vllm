# 模型算子契约

[阶段清单](../../python/oh_my_vllm/kernels/specification.py)给出内部语义阶段的输入, 输出, 计算和状态写入.
维度来自 checkpoint 配置和张量头.

使用清单中的配置 SHA256 标识模型.
不改变 checkpoint 值或 FP8 scale.
加载器可将 BF16 norm 权重, scales, A_log 和 dt_bias 无损提升到 FP32.

## 语言模型

| 属性 | 值 |
|---|---|
| Hidden / FFN / 词表 | 5120 / 17408 / 248320 |
| 层数 | 48 GDN + 16 FA |
| GQA | 24 query heads, 4 KV heads, 宽度 256 |
| GDN | 16 key heads, 48 value heads, 宽度 128 |
| 卷积 | 10240 channels, 宽度 4 |
| Cache | Block 784, BF16 循环状态, align 模式 |
| 上下文 / 步骤 | 262144 / 32768 tokens |
| FP8 | E4M3, 128-by-128 权重 scale, 动态 128-element 激活 scale |
| MTP | 一层 FA, 四个 proposals |

适用的 norm 权重使用 checkpoint offset.
GDN gate norm 使用 checkpoint 权重, 不加 offset.

打包 GQA 投影有 14336 列.
每个 query head 的 256 个 query 列后紧接 256 个 gate 列.

最后 2048 列为 K 和 V.
GQA 输出 gate 使用 sigmoid. GDN 输出 gate 使用 SiLU.

MRoPE 使用 head 的 256 个元素中的 64 个, theta 10000000, 交错 sections `(11,11,10)`.
图像和视频 token 保持时间及空间位置.

GDN 使用以下公式, value head 对应的 key head 为 `h // 3`:

```text
q = q * rsqrt(sum(q*q) + epsilon)
k = k * rsqrt(sum(k*k) + epsilon)
g = -exp(A_log) * softplus(a + dt_bias)
beta = sigmoid(b)
S_decay = exp(g) * S
delta = beta * (v - S_decay @ k)
S_next = S_decay + outer(delta, k)
o = S_next @ q / sqrt(128)
```

保持推理路径中的 BF16 舍入点.
Prefill 和 recurrent 验证行使用 FP32 live carry.
持久 checkpoints 使用 BF16 存储.

在每个 784-token 边界和最终行保存 checkpoint.
验证写入私有状态. 只有已接受行可以替换已接受状态.
序列边界与读取, 写入及 checkpoint slots 标识每个请求的卷积状态.

接受 token 时, 选择已保存的状态 slots.
不为接受 token 复制完整的 recurrent 或卷积状态.

MTP context 第 `p` 行配对 target hidden 第 `p-1` 行和第 `p` 个 token.
位置零没有 MTP 行. 前缀恢复时提供边界 hidden 行.
图片和视频行在 MTP 输入 norm 中使用各自的 vision embedding.

Context fill 为全部 context 行投影 K 和 V.
只有选定端点需要 Q, gate, attention 输出及 FFN 输出.

GQA 输入变体分别使用完整 14336 列, K/V 2048 列或 Q/gate 12288 列.
MTP target 验证为每个请求产生最多五行词表输出.
未启用的 draft 项构成 token ID 为 `-1` 的后缀.
每个请求在有效 draft 之后的 target 行直接提供 bonus 分布, 不减去 proposal 分布.
第一个 draft 在端点使用 Q/gate-only 变体.
其余 3 个 draft 使用完整的 MTP 输入及层路径.

## 视觉模型

| 属性 | 值 |
|---|---|
| 层数 / hidden / FFN | 27 / 1152 / 4304 |
| Attention | 16 heads, 宽度 72, 非因果 |
| Patch | 2 帧乘 16 乘 16 像素, RGB |
| 空间合并 | 2 乘 2 patches |
| 位置表 | 2304 乘 1152 |
| 输出 | 5120 列 |
| 权重 | BF16, 带 bias |
| 激活 | Block MLP: GELU tanh 近似. Merger: exact GELU |

如果时间 patching 还需要一帧, 再使用一次最后一帧.
使用 checkpoint 的图像归一化和 resize 限制.

按空间 grid 插值绝对位置表.
对 Q 和 K 应用空间 RoPE.

Attention 连接同一时间帧组内的 patches.
不连接不同图片或帧组.

merger 在 2-by-2 gather 前归一化每个 1152-element patch.
两次投影的 shape 为 `4608x4608` 和 `5120x4608`.

将输出 embeddings 写入对应语言 token 行.
保持文本, 图片和视频之间的输入顺序.

## 中间精度

数值行为与 [vLLM commit e8e1e961](https://github.com/vllm-project/vllm/commit/e8e1e96196725f097eae2394cbc458b1f5a679b8) 对齐.
采用其 BF16 模型路径及项目的 cache 设置. 此源码参考不引入 vLLM 依赖.
使用源码中的类型和舍入点, 但所选 Tensor Core 路径可以采用获准的较低精度.

数值检查允许时, 使用 FP8 或 BF16 Tensor Core 操作数.
实现可以将中间操作数和预处理后的 scales 舍入为 Tensor Core 格式.
原始 checkpoint 张量保持不变. 不为通过数值检查而提高精度.
使用[评估流程](evaluation.zh.md)中的全模型 logits 门槛.

超过 128 行的 FFN 输入使用 MXFP8 Tensor Core 投影.
超过 128 行的 GDN 输入使用 MXFP8 输入投影.
对于 prefill 及 `max_queries > 1` 的验证, 其输出投影也使用 MXFP8.
单查询验证保留先前的输出投影.
权重预处理转换派生的 E4M3 编码, 并提供值为 2 的幂的 UE8M0 scales.
[归一化 kernel](../../python/oh_my_vllm/kernels/cuda_backend/mx_fp8_norm.cuh#L7)<!-- source: norm_mx_kernel(const --> 直接从其 BF16 结果生成 MXFP8 操作数.

原始 checkpoint 张量和 scales 保持不变.

超过 128 行的完整 FA 边界使用 MXFP8 输入投影.
当 attention 只有一个分区时, 使用 MXFP8 输出投影.

FA 在相同逻辑槽中保留 BF16 K/V 及其派生的 E4M3 副本.
每次 BF16 cache 写入后, 在 attention 读取前更新派生副本.
K 使用由其 RMS 权重确定上界的逐层 scale. V 为每个 token 和 head 使用一个 scale.
最多含 32 行且每个序列最多含 5 个 query 的 FA batch 使用 E4M3 Q/K/V/P, 以及 FP32 softmax 和累加.
每个 query 的概率 scale 仅使用该分区内对它可见的 V scales.

至少含 1024 个 query 的 prefill 使用 E4M3 Q/K 和 BF16 P/V.
其他 attention batch 使用 BF16 操作数.
这些选择不会改变固定的 [HTTP 参考计算](../performance.zh.md#下界计算).

| 区域 | 计算 | 存储结果 |
|---|---|---|
| 其他 FP8 投影 | E4M3 操作数, FP32 scales 和累加 | BF16 |
| MXFP8 FFN 和 GDN 投影, 超过 128 行 | E4M3 操作数, UE8M0 scales, FP32 累加 | BF16 |
| 所选的 MXFP8 FA 投影, 超过 128 行 | E4M3 操作数, UE8M0 scales, FP32 累加 | BF16 |
| BF16 投影 | BF16 操作数, FP32 累加 | BF16 |
| 残差加法和 RMS | FP32 求和及统计量. 对未舍入的和执行归一化 | BF16 残差及归一化结果 |
| Q/K RoPE | BF16 归一化 Q/K 及 cos/sin. FP32 乘积和求和 | BF16 |
| GDN Q/K norm | FP32 | Prefill 使用 BF16. Recurrent 行使用 FP32 live values |
| GDN scan | FP32 live carry, 乘积和求和 | BF16 checkpoints |
| GDN chunked prefill | BF16 矩阵操作数. FP32 累加, 状态衰减和 live carry | BF16 checkpoints |
| GDN head norm 和 gate | BF16 attention 结果, 随后使用 FP32 RMS, weight, SiLU 和乘积 | 一个 BF16 结果 |
| FA attention 和 gate | 所选的 FP8 或 BF16 操作数, FP32 softmax 及累加 | FP8 前的 BF16 attention, sigmoid 和乘积 |
| Vision attention | BF16 Q/K/V/P, FP32 softmax 及累加 | BF16 |
| MTP dual RMS | FP32 统计量和增益计算 | BF16 packed input |
| MTP FC, 全部行数 | BF16 操作数, FP32 累加 | BF16 |
| Sampling 接受和 residual CDF | FP32 概率, 比值, masses 和求和. FP64 随机阈值 | 整数选择 |

分块 GDN prefill 遵循固定版本的[求解及投影类型](https://github.com/vllm-project/vllm/blob/e8e1e96196725f097eae2394cbc458b1f5a679b8/vllm/third_party/flash_linear_attention/ops/chunk.py#L38).
三角系统结果和经过 gate 缩放的 K/V 操作数, 在 W/U 矩阵乘前使用 BF16 存储.
W/U 输出也使用 BF16 存储.
三角逆可以使用 BF16 中间存储及 BF16 分块矩阵操作数, 并使用 FP32 累加.
对角块求解保持 FP32. 状态衰减, live carry, 完整 token 历史及私有 checkpoint 要求保持不变.

[Carry 路径](https://github.com/vllm-project/vllm/blob/e8e1e96196725f097eae2394cbc458b1f5a679b8/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py#L154)在矩阵乘前将状态操作数及缩放后的新 value 舍入为 BF16.
状态更新先缩放 FP32 新 value, 再转换为 BF16.
独立的输出导出使用 BF16.
状态衰减和 carry 更新保持 FP32. BF16 checkpoint 导出不得替换 live carry.

[输出路径](https://github.com/vllm-project/vllm/blob/e8e1e96196725f097eae2394cbc458b1f5a679b8/vllm/third_party/flash_linear_attention/ops/chunk_o.py#L94)使用 BF16 状态快照和新 value, 使用 FP32 累加.
它在新 value 的矩阵乘前, 将带因果掩码和 gate 缩放的 QK 矩阵舍入为 BF16.
[投影源码](https://github.com/vllm-project/vllm/blob/e8e1e96196725f097eae2394cbc458b1f5a679b8/vllm/third_party/flash_linear_attention/ops/wy_fast.py#L75)定义 gate 缩放的舍入点.

固定版本的 [RMS 参考](https://github.com/vllm-project/vllm/blob/e8e1e96196725f097eae2394cbc458b1f5a679b8/vllm/ir/ops/layernorm.py#L44) 区分 FP32 归一化输入和 BF16 残差存储.
[GDN decode 路径](https://github.com/vllm-project/vllm/blob/e8e1e96196725f097eae2394cbc458b1f5a679b8/csrc/gdn/fused_gdn_decode_kernel.cu#L226) 将 recurrent Q/K 保持为 FP32.
[视觉注意力依赖](https://github.com/vllm-project/flash-attention/blob/b273a02621d3c94bc747df67204f591b67f38fae/flash_attn/cute/flash_fwd_sm100.py#L2522) 在 PV 前将在线概率舍入为 BF16.
全模型参考全程使用 FP64. 它不复制中间低精度舍入.

[HTTP 参考计算](../../development/serving/accounting.py)使用固定精度假设进行性能计费.
临时存储和重复工作不能抬高理论下界.

## 数值检查

使用[全模型评估流程](evaluation.zh.md).
参考以 checkpoint 权重及其原始 scales 进行直接 PyTorch FP64 计算.
在短输入和一次 decode 迭代后比较完整词表.
固定局部用例及其经审查的数值门槛使用[算子 microbench 契约](microbench.zh.md).

保持上述状态所有权, checkpoint 和精度契约.
服务边界使用[融合计划](fusion.zh.md).
