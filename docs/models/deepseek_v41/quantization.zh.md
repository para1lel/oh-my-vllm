# 量化契约

Adapter 按发布的 [DeepSeek reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py) 执行 block quantization.
Reference source 的 SHA-256 为 `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`.
本契约适用于 activation, window KV, compressed KV, index key 和 query.
本契约不改变原始 weight value 或 scale.

## 数学

设 `A` 为一组中最大的 FP32 绝对值.
使用以下 normalization scale:

| Format | Normalization scale |
|---|---|
| FP8, 未取整 | `max(A, 1e-4) * (1 / 448)` |
| FP8, 已取整 | `max(A, 1e-4) * (1 / 448)` 向上取二次幂 |
| FP4, UE8M0 scale | `max(A, 6 * 2**-126) * (1 / 6)` 向上取二次幂 |
| FP4, E4M3 scale | `E4M3(max(A, 6 * 2**-9) / 6)` |

向上取二次幂使用 FP32 的 exponent 和 mantissa bit.
它不使用经过舍入的 `log2` 结果.
FP4 E4M3 scale 必须先舍入, 再归一化数值.
UE8M0 scale storage 向上舍入.
当保存的 scale 使用 UE8M0 时, 未取整的 FP8 normalization 仍使用原始 FP32 scale.

E4M3 和 E2M1 数值使用最近偶数舍入.
将归一化的 FP8 数值限制在 `[-448, 448]`.
将归一化的 FP4 数值限制在 `[-6, 6]`.
E2M1 magnitude code `0` 至 `7` 表示 `0, 0.5, 1, 1.5, 2, 3, 4, 6`.

偶数 channel 放在低 nibble.
负零保留其符号.
全零 FP4 group 保留非零 scale.

In-place 调用先将归一化数值舍入到所选格式, 再反量化.
结果转回输入 dtype.
Encoded 调用返回 E4M3 数值或 packed E2M1 byte, 并单独返回 scale.
FP4 encoded storage 的宽度是逻辑宽度的一半.

FP4 projection 的 activation 使用 FP8, 而不是 FP4.
Native projection 使用每 32 个 channel 一组的 activation scale.
每个 128-channel activation scale 在四个 native group 中重复.
Reference FP8 projection 沿 row 和 column group 两个维度展开 weight scale.
FP32 scale 输入使用 reference projection path, 而不是 native UE8M0 byte interface.

## 执行

[Model quantizer](../../../python/oh_my_vllm/models/deepseek_v41/quantization.py) 选择 native BF16 path 或 reference computation.
[Native interface](../../../python/oh_my_vllm/operators/transform/api.py) 规定 shape, format, alignment 和 overlap rule.
它使用 caller 的 CUDA stream.
每个 warp 在 in-place write 之前读取一个完整 group.
Native quantization 不分配临时 device storage.

Tensor preparation 可以生成一个对齐的 contiguous input copy.
Encoded output 和 scale 使用独立 storage.

Metadata failure 在 launch 之前发生.
无效 device value 或 scale 产生无效 scale code.
Native attention 在接受 output 之前检查这些 code.
Input 必须产生 interface 规定的有限 reconstruction result.

Native attention 中的 window KV 和 compressed KV 保持分离.
Reference path 在计算之前连接两张 BF16 table.
Short-query native selection 和 attention status fallback 保留现有规则.

## 验收

[Quantization 检查](../../../tests/test_deepseek_quantization.py) 使用独立的数值和 encoded byte 字面量.
它们检查 scale rule, 最近偶数中点舍入, 负零, exponent boundary 和 projection scale group.
CUDA case 检查 stream order, invalid metadata, empty history 和 temporary storage bound.
这些检查不能替代独立的 full-model endpoint fixture, 或批准的 HTTP 和 AgentX workload.
