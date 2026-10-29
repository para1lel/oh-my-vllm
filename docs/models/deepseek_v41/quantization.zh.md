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

[Router](../../../python/oh_my_vllm/models/deepseek_v41/model.py) 在 score calculation 之前, 使用 FP32 input 和 weight 执行 projection.
Expert selection 之前不要将 router logits 舍入到 BF16.

Tensor-parallel shared-expert down projection 保留 FP32 partial sum, 直到 rank reduction 完成.
将归约结果舍入到 BF16.
Attention output projection 保留其 FP32 rank reduction 之前的 local BF16 舍入.

Sparse attention 按原顺序处理 selected index, 每个 block 包含 64 个 slot.
QK 和 probability-V product 使用 BF16 operand, 并以 FP32 累加.
Running maximum 从 `-1e30` 开始, 不包含 sink.
Maximum 改变时, 每个 block 对之前的 FP32 numerator 和 denominator 重新缩放.
未舍入的 exponential 计入 denominator.
与 V 相乘之前, 将这些 exponential 舍入到 BF16.

所有 selected block 完成后, 将 sink exponential 加入 denominator.
将 numerator 除以 denominator.
将 output 舍入到 BF16.
保留 duplicate index 和 masked slot 的原始位置.
Empty row 和 fully masked row 返回零, 即使 finite sink exponential 下溢也如此.

在 Sinkhorn mixing 中, 第一次 column normalization 之前, 给 row softmax value 加上 `hc_eps`.
发布模型使用 `hc_eps=1e-6`.

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

## Reference 执行

使用隔离的 development environment 执行发布的 TileLang reference.
不要将 TileLang 加入 inference dependency.
对于此 quantizer source, TileLang 0.1.8 可能在所有 input store 完成前复用 input shared storage.
在 reference 副本中, 每个 quantizer 写入 output shared storage 前都要使用 CUDA block barrier.
在 `act_quant_kernel` 和 `fp4_quant_kernel` 中, 紧接 `T.copy(y_local, y_shared)` 之前加入 `T.sync_threads()`.
不要改变 equation, cast 或原始 checkpoint file.

在每个 reference manifest 中记录 original source, patch, modified source, compiler 和 runtime identity.
生成 expected array 前, 使用 Compute Sanitizer Racecheck 检查 packed 和 in-place format.
除非重复执行的 target 和 DSpark endpoint 逐位一致, 否则不要冻结 array.
将此 backend 标识为带 synchronization patch 的 reference, 而不是未修改的发布版本.
不要导入 candidate module 来生成 expected array.
