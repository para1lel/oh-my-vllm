# 模型算子契约

[阶段清单](../../python/oh_my_vllm/operators/specification.py)给出内部语义阶段的输入, 输出, 计算和状态写入.
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
| Cache | Block 784, FP32 循环状态, align 模式 |
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

保持下文规定的关键中间类型和算子输入输出类型.
Prefill 和 recurrent 验证行使用 FP32 live carry.
持久 recurrent checkpoints 使用 FP32 存储. 卷积状态保持 BF16.

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

数值行为与 [vLLM commit 86129933](https://github.com/vllm-project/vllm/commit/86129933fb2ddc5f1e4d9ce32c4062277c7ac2af) 对齐.
采用其默认 BF16 模型路径及 FP32 循环状态. 此源码参考不引入 vLLM 依赖.
原始 checkpoint 张量保持不变.

采用与其默认编译路径相同的关键中间类型和算子输入输出类型.
舍入方式可以不同.
遵守这些类型及数值门槛. 使用简单且执行成本低的运算.
使用[评估流程](evaluation.zh.md)中的全模型 logits 门槛.

所有行数的 FP8 投影均使用 checkpoint 中的 E4M3 权重和 FP32 分块 scales.
激活量化为每个 128 元素组使用动态 FP32 scale.
不要将权重或 scales 转换为 MXFP8.
锁定版本的 [vLLM 排除规则](https://github.com/vllm-project/vllm/blob/86129933fb2ddc5f1e4d9ce32c4062277c7ac2af/vllm/utils/deep_gemm.py#L27) 因精度问题, 在 B200 上为 `qwen3_5_text` 禁用 DeepGEMM.

FA 使用 BF16 Q/K/P/V 操作数, 以及 FP32 softmax 和累加.
持久 K/V 状态使用 BF16.
锁定的 vLLM 源码在 B200 上默认使用 FlashInfer 后端.

FFN 从 BF16 投影输出开始, 用 FP32 计算 SiLU 及其乘积.
与[融合激活](https://github.com/vllm-project/vllm/blob/86129933fb2ddc5f1e4d9ce32c4062277c7ac2af/csrc/activation_kernels.cu#L676)一致, 在 E4M3 量化前将乘积转换为 BF16.
BF16 值可保留在寄存器中.
此融合乘积的激活 scale 下限为 `1e-30 / 448`.

FFN 残差输出保持 FP32, 直到下一 attention 输入或最终归一化.
下一 attention 输入先加上 FFN value, 再将其残差存为 BF16.
归一化使用此转换前的 FP32 和.
这保持了 vLLM 默认编译路径的中间精度.

| 区域 | 计算 | 存储结果 |
|---|---|---|
| FP8 投影 | E4M3 操作数, FP32 scales 和累加 | BF16 |
| BF16 投影 | BF16 操作数, FP32 累加 | BF16 |
| 残差加法和 RMS | FP32 求和及统计量. 对未舍入的和执行归一化 | FP32 FFN 残差. BF16 attention 残差及归一化结果 |
| Q/K RoPE | BF16 归一化 Q/K 及 cos/sin. FP32 乘积和求和 | BF16 |
| GDN Q/K norm | FP32 | Prefill 使用 BF16. Recurrent 行使用 FP32 live values |
| GDN scan | FP32 live carry, 乘积和求和 | FP32 checkpoints |
| GDN chunked prefill | BF16 矩阵操作数. FP32 累加, 状态衰减和 live carry | FP32 checkpoints |
| GDN head norm 和 gate | BF16 attention 结果, 随后使用 FP32 RMS, weight, SiLU 和乘积 | 一个 BF16 结果 |
| FA attention 和 gate | BF16 操作数, FP32 softmax 及累加 | FP8 前的 BF16 attention 和乘积. FP32 sigmoid |
| Vision 位置插值 | BF16 权重, 乘积和求和 | BF16 |
| Vision attention | BF16 Q/K/V/P, FP32 softmax 及累加 | BF16 |
| MTP dual RMS | FP32 统计量和增益计算 | BF16 packed input |
| MTP FC, 全部行数 | BF16 操作数, FP32 累加 | BF16 |
| Sampling 接受和 residual CDF | FP32 概率, 比值, masses 和求和. FP64 随机阈值 | 整数选择 |

vLLM 在 B200 上的默认 GDN prefill 后端为 FlashInfer.

项目的分块实现使用 FLA 的[求解及投影类型](https://github.com/vllm-project/vllm/blob/86129933fb2ddc5f1e4d9ce32c4062277c7ac2af/vllm/third_party/flash_linear_attention/ops/chunk.py#L38).
三角系统结果和经过 gate 缩放的 K/V 操作数, 在 W/U 矩阵乘前使用 BF16 存储.
W/U 输出也使用 BF16 存储.
三角逆可以使用 BF16 中间存储及 BF16 分块矩阵操作数, 并使用 FP32 累加.
对角块求解保持 FP32. 状态衰减, live carry, 完整 token 历史及私有 checkpoint 要求保持不变.

[Carry 路径](https://github.com/vllm-project/vllm/blob/86129933fb2ddc5f1e4d9ce32c4062277c7ac2af/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py#L154)在矩阵乘前将状态操作数及缩放后的新 value 舍入为 BF16.
状态更新先缩放 FP32 新 value, 再转换为 BF16.
独立的输出导出使用 BF16.
状态衰减, carry 更新及持久 checkpoint 导出保持 FP32.

[输出路径](https://github.com/vllm-project/vllm/blob/86129933fb2ddc5f1e4d9ce32c4062277c7ac2af/vllm/third_party/flash_linear_attention/ops/chunk_o.py#L94)使用 BF16 状态快照和新 value, 使用 FP32 累加.
它在新 value 的矩阵乘前, 将带因果掩码和 gate 缩放的 QK 矩阵舍入为 BF16.
[投影源码](https://github.com/vllm-project/vllm/blob/86129933fb2ddc5f1e4d9ce32c4062277c7ac2af/vllm/third_party/flash_linear_attention/ops/wy_fast.py#L75)定义 gate 缩放的舍入点.

固定版本的 [RMS 参考](https://github.com/vllm-project/vllm/blob/86129933fb2ddc5f1e4d9ce32c4062277c7ac2af/vllm/ir/ops/layernorm.py#L44) 区分 FP32 归一化输入和 BF16 残差存储.
[GDN decode 路径](https://github.com/vllm-project/vllm/blob/86129933fb2ddc5f1e4d9ce32c4062277c7ac2af/csrc/gdn/fused_gdn_decode_kernel.cu#L226) 将 recurrent Q/K 保持为 FP32.
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
Serving 边界使用[算子设计](design.zh.md#serving-边界).
