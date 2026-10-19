# 全模型数值评估

使用[模型契约](model.zh.md)及[开发命令](development.zh.md).
[GPU 执行规则](../execution.zh.md)规定执行方式.

## 数值范围

使用固定的 16-token 文本提示和一次 decode 迭代.
提示为 `The quick brown fox jumps over the lazy dog. Explain why the sky is blue`.
[采集代码](../../development/model/capture.py)规定其 token IDs.
对照包含从 embedding, 全部 64 层到最终词表投影的全部 248320 个 logits.
不比较中间算子, 也不使用单算子容差.

独立[参考生成器](../../development/model/reference.py)直接使用 PyTorch FP64 计算.
它按原始 128-by-128 scales 将 checkpoint E4M3 权重解码为 FP64.
BF16 checkpoint 值精确转换为 FP64.
之后的所有算术, 激活, 归一化, attention 和 recurrent state 均使用 FP64.
参考不使用激活量化, BF16 状态存储或中间低精度舍入.
它不使用推理 kernel 或旧算子输出.

参考先计算提示并保留 FP64 状态.
提示中最大的 logit 选出单次 decode 迭代的固定输入 token.
实现完成自身提示计算后使用同一 token.
因此, decode 对照包含累计的提示误差和完整 decode 计算.
比较使用 Eager prefill 和 CUDA Graph decode.
报告记录执行策略.
不执行生成 token 的循环.

Prefill 变化时, 使用 `--mode prefill` 比较同一提示的末行.
保存的 artifact 包含两个 endpoint 向量及其 checkpoint identity.
这些短文本检查不能证明长上下文, MTP 或 vision 的数值行为.
这些路径变化时扩展全模型输入及 endpoint.
协议和状态所有权检查与数值对照分开.

## 证据与门槛

将冻结的 [FP64 artifact](../../development/model/reference.npz) 保存在仓库中.
记录其 SHA256, checkpoint identity, 参考源码 identity, PyTorch 版本和输入 token IDs.
拒绝发生变化的 artifact, checkpoint 文件, 输入, 非有限 logits 或不完整词表向量.
不要用推理输出替换参考.
生成参考时不要覆盖已有 artifact.

报告给出 NRMSE, 最大绝对误差, 余弦相似度及两侧的 top-1 token IDs.
NRMSE 为 logits 差值的 L2 norm 除以 FP64 logits 的 L2 norm.
余弦相似度使用完整向量, 不做中心化.
Top-1 是否相同仅用于诊断, 因为小误差可能交换接近的 logits.

[对照策略](../../development/model/compare.py)规定固定的全模型门槛和 artifact identity.
门槛为 NRMSE 至多 0.05, 余弦相似度至少 0.999, 最大绝对误差至多 0.5.
三项门槛必须全部通过. 对照失败时以错误退出.
这些门槛约束该短用例. 它们不能证明其他输入的准确性.
实现变化后不要自动调整这些门槛.
推理向量及报告保存在仓库外.
