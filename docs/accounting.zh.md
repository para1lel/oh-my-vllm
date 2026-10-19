# HTTP 参考计费假设

版本 `http-reference-v1` 使用[规范化调度](performance.zh.md#下界计算)及[标称速率](execution.zh.md#http-性能).
这是保守资源估算, 不是最小物理延迟证明.
[计算源码](../development/serving/accounting.py)规定维度和公式.

## 内存

每个逻辑投影 pass 只计算一次矩阵权重, 由其 batch 共享.
FP8 矩阵每元素使用一字节, 每个 128-by-128 区域使用 FP32 scale.
BF16 矩阵每元素使用两字节.
每个 pass 只计算一次各归一化向量.
中间 hidden state, logits 和 activation buffer 不计内存分数.

FA 对每个请求和 pass 读取一次历史 KV.
新 query KV 可以直接交给 attention, 不再读取 HBM.
接受的 KV 只计一次写入, 使用 4 个 KV head, 宽度 256, K 和 V 均为 BF16.
MTP draft attention 从位置 1 开始. 首个 draft 读取缓存的 context endpoint.
后续 draft 计算新 KV.

当请求起点大于零时, GDN 读取一份 BF16 recurrent 和 convolution state.
计算该 pass 中每个不同的 784-token checkpoint 和接受终点.
终点上的 checkpoint 只计一次.
Verification 前进接受的 proposal 数加一行.
每个 GDN 层有 48 个 128-by-128 recurrent head, 以及 10240 个各存储 3 个值的 convolution channel.

私有 candidate 写入和后续 state 复制不计分.
此省略不取消 private-state 正确性检查.

## 算术

矩阵乘法按规定 operand 精度计算, 每个 multiply-add 计两次操作.
计算涵盖全部 64 个 target 层和完整词表投影.
全部 verification 行都计入, 包括 proposal 被拒绝的情况.
FA 计算完整因果 attention pair, 每个 pair 和 query head 计三次 FP32 操作.
这些标量操作代表 score scaling, probability summation 和 probability normalization.

GDN prefill 使用固定的 64-row chunk, 在 784-token checkpoint 处分割.
其参考 block recurrence 计算 BF16 乘积及 FP32 对角求逆和 state update.
Decode 和 verification 使用固定的 FP32 recurrent 公式.
Runtime 算法变化不能改变此参考选择.

RMS normalization 每元素计四次操作, 每行另计两次标量操作.
Residual addition 每元素计一次操作.
FFN activation 每个 intermediate 元素计两次乘积.
GDN convolution 每 token 和 channel 计七次操作.

FA head normalization 使用宽度 256. Rotary product 使用规定的 64 维.
Attention output gate 每元素计一次乘积.
MTP context, draft 和词表投影遵循记录的因果位置.

## 省略项

HTTP 用例使用不带 penalty 的 greedy sampling.
Selection comparison, sorting, integer indexing, random generation 和 greedy MTP acceptance 不计浮点分数.
估算省略 sigmoid evaluation, transcendental instruction, datatype conversion, quantization 和 scale application 算术.
它也省略上述条目之外的 GDN head normalization, gate 和 input normalization 算术.
省略的操作没有经验吞吐系数.

Trace 没有用于 embedding-region 去重的完整 input token ID.
因此 embedding 内存读取不计分.
Checkpoint hidden storage, MTP endpoint storage, metadata 和返回 token 流量也被省略.
这些保守省略可能降低报告效率.
当实现增加工作量时, 它们不能提高效率.

Vision 和初始化不在这些纯文本 HTTP 计时范围内.
它们的 task 覆盖和正确性是独立的最终验收条件.
修改这些假设必须更新计费版本, 并使旧性能证据失效.
