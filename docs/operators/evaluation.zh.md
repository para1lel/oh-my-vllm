# 全模型数值评估

使用[模型契约](model.zh.md)及[开发命令](development.zh.md).
[GPU 执行规则](../execution.zh.md)规定执行方式.

## 数值范围

默认用例使用固定的 16-token 文本提示和一次 decode 迭代.
提示为 `The quick brown fox jumps over the lazy dog. Explain why the sky is blue`.
[采集代码](../../development/model/capture.py)规定其 token IDs.
对照包含从 embedding, 全部 64 层到最终词表投影的全部 248320 个 logits.
不比较中间算子, 也不使用单算子容差.
[算子 microbench 套件](microbench.zh.md)使用已存储的 PyTorch fixture 提供额外的局部检查.

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
下方新增用例检查长上下文, MTP 和 vision 计算.
协议和状态所有权检查与数值对照分开.

## 新增固定用例

[用例定义](../../development/model/cases.py)规定存储的输入及完整词表输出.
常规测试加载仓库 fixture, 不使用 PyTorch.

| 用例 | 固定输入 | 词表输出 |
|---|---|---|
| `mtp-restored` | 784-token prefix, 五行验证输入, 两行被选中 | 全部五行验证输出, 恢复后的 decode, 首次 MTP draft, 下一次 MTP draft |
| `image-video` | 178-token prefix, 一张图像, 三帧视频 | Prefill, decode, 首次 MTP draft |
| `context-32769` | 32768-token prefix 和一个 decode token | Prefill 和 decode |

恢复用例在验证前清除请求的私有状态引用.
验证读取逻辑 784-token checkpoint.
状态选择提交前两行输入.
下一次 target 计算读取该选中状态.
MTP context 使用存储的边界 hidden 行及选中的验证行.
两次 draft 输入使用固定 token.

媒体用例检查图像和视频 transformer, merger, 语言 embedding 替换, 多轴位置和 MTP embedding 替换.
三帧视频检查末帧复制.
长上下文用例在 decode 前检查完整因果 attention 及累计 recurrent state.

[新增参考生成器](../../development/model/reference_cases.py)直接使用 PyTorch FP64 公式.
Query 分块限制 attention 内存占用, 同时保留每个 query 的完整因果 key 范围.
生成器将输入, 完整 endpoint 数组, checkpoint identity 和参考来源信息保存为压缩分片.
每个分片至多 32 MiB.
Fixture 索引通过 SHA256 绑定每个 manifest.

将 `MODEL_ROOT` 设为 checkpoint 目录.
将 `LOGITS_REPORT` 设为新的外部 JSON 路径.
运行一个新增用例:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.model --model "$MODEL_ROOT" --case mtp-restored --report "$LOGITS_REPORT"
```

使用其他用例名运行对应固定检查.
每条命令比较列出的所有 endpoint.
验证对照逐行独立检查.

显式生成 fixture 时, 将 `MODEL_FIXTURES` 设为新的外部目录.
使用独立参考环境:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.model.reference_cases --model "$MODEL_ROOT" --output "$MODEL_FIXTURES"
```

将 fixture 复制到仓库前, 检查生成器, manifest 哈希及完整 endpoint 数组.
验收期间不要生成预期数组.

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
每个 endpoint 行都必须通过全部三项门槛. 对照失败时以错误退出.

| 用例 | 最大 NRMSE | 最小余弦相似度 | 最大绝对误差 |
|---|---|---|---|
| 短提示 | 0.05 | 0.999 | 0.5 |
| `mtp-restored` | 0.09 | 0.996 | 1.6 |
| `image-video` | 0.06 | 0.998 | 1.3 |
| `context-32769` | 0.07 | 0.998 | 1.1 |

短用例保留原有门槛.
新增门槛采用 [vLLM commit 86129933](https://github.com/vllm-project/vllm/commit/86129933fb2ddc5f1e4d9ce32c4062277c7ac2af) 以及选定的项目实现, 使用同一 checkpoint, 仓库存储的输入及 FP64 参考.
对每个用例, 取其所有 endpoint 行中最大的 NRMSE 和绝对误差, 以及最小的余弦相似度.
分别按 0.01, 0.1 和 0.001 的步长向外取整.
各用例独立设定门槛. 取这两个实现中较大的误差上限和较小的余弦下限.

选定的项目实现使用 FP32 FFN residual, 按列乘 beta 的 GDN inverse, 以及带缩放的 BF16 GDN attention 系数.
此校准在算子迁移前固定门槛. 不修改参考数组或算子数值门槛.

vLLM 校准使用默认 BF16 路径, FP8 投影, FP32 recurrent state 和 O2 编译.
其数值 adapter 为各 cache group 分配互不重叠的 cache blocks.
对于媒体 MTP, 它使用存储的三轴位置编译模型.
此 adapter 检查模型计算.
它不检查默认使用一轴位置的 V2 proposal buffer.

将 adapter, 环境 identities, 输入 hashes, 生成代码 hashes 和完整报告保存在仓库外的校准 artifacts 中.

这些门槛约束固定用例. 它们不能证明其他输入的准确性.
实现变化后不要自动调整这些门槛.
推理向量及报告保存在仓库外.
