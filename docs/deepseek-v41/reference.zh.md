# DeepSeek FP64 参考构建

## 范围

[阶段计划](plan.zh.md)控制 native 算子调优之前的参考构建.
这些工具提供 checkpoint 身份, 解码后的权重, Engram 哈希, FP64 目标模型和预测器数学实现, 以及 CPU 文本参考生成器.
[固定夹具](fixtures.zh.md) 提供短文本端点和提示词编码用例.
目前尚未固定 DSpark, 媒体和长上下文端点夹具.
未声明运行时数值验收.
[checkpoint 指南](checkpoint.zh.md)规定身份和行加载检查.

将参考与推理代码和 native 算子分开.
使用 `requirements/deepseek-reference.lock` 指定的固定环境.
其 PyTorch 和 NumPy 版本与 `requirements/model-reference.lock` 一致.
它还提供用于 Engram 归一化的固定分词器库.
推理运行时不得导入 PyTorch 或这些开发模块.

## 算术

将原始 BF16, FP8 和 packed FP4 checkpoint 数值直接解码为 FP64.
在 FP64 中乘以其原始尺度.
对 projections, activations, 归一化, residuals, attention 和持久数学状态使用 FP64.
不得在该参考中加入运行时 activation 量化或 BF16/FP32 中间舍入.

[数学函数](../../development/model/deepseek_v41/reference_math.py)实现 hyper-connections, 相邻元素对旋转变换, sparse attention sinks 和 routing.
Hyper-connection 残差系数的两个轴依次为源副本和目标副本.
RMS 权重乘以归一化数值, 不添加 Qwen 式的权重偏移.
有限次 Sinkhorn 迭代在每个指定归一化步骤中包含源实现的 epsilon.
Attention sink 参与 softmax 分母, 但其 value 为零.

专家分数完全相同时, 优先选择较小的全局专家 IDs.
本地 `torch.topk` 调用不规定稳定的相等分数顺序.
该显式项目规则防止 rank 放置改变分数相等时的专家选择.
Correction bias 用于选择专家.
无偏分数决定 routing 权重.

[源代码契约](model.zh.md)和[状态契约](state.zh.md)规定其余模型方程.
算子和全模型限值必须在调优之前固定.
这些数学函数不设置或放宽这些限值.

## 逐层流式构建

将原始 packed 权重保留在 checkpoint 文件中.
通过已检查的权重读取器, 每次加载一个 projection 或选定专家.
在下一个 projection 之前释放临时 FP64 权重存储.
将已接受的 activations 和数学 cache 状态与临时权重分开.

按选定 IDs 读取 embedding 和 Engram 行.
以有界片段流式读取词表 head 行, 然后组装全部 129280 个 logits.
不得用 top-k logits 代替完整端点向量.
不需要完整的 FP64 模型副本.

对短 prefill 参考计算全部 40 个目标层.
保留其 FP64 状态供下一个 decode token 使用.
使用最高的 prefill logit 选择该固定 decode 输入.
将选定 token 与两个完整端点向量一起记录.
草稿, 恢复, 媒体和长上下文参考是独立用例.

将每个 artifact 绑定到 checkpoint, 分词器, 参考源码, 依赖版本, 输入 tokens 和输出数组身份.
拒绝已有输出路径, 不覆盖参考 artifacts.
将原始生成记录保存在仓库之外.
仅在生成器和身份通过独立审查后, 添加已审查的端点 fixtures.

## 短文本生成

[生成器](../../development/model/deepseek_v41/reference.py)将固定的 16-token 提示绑定到本地分词器.
它在计算之前重新计算完整 checkpoint 身份.
它在计算之后检查 checkpoint 文件状态和参考源码身份.
输入文件或源文件变化会导致错误, 而不是输出 artifact.

输出包含两个完整的 FP64 词表向量及其 metadata.
Metadata 绑定依赖, 参考源码, checkpoint 身份, 提示 tokens, 选定 decode token 和数组 digests.
该命令仅使用 CPU 执行.
它不生成 DSpark, 媒体或长上下文 fixtures.

将 `REFERENCE_ENV` 设为仓库之外的新环境目录.
将 `MODEL_ROOT`, `CHECKPOINT_MANIFEST` 和 `REFERENCE_OUTPUT` 分别设为 checkpoint, 完整 manifest 和新的外部 NPZ 路径.

```bash
scripts/with-env.sh uv venv "$REFERENCE_ENV"
scripts/with-env.sh uv pip sync --python "$REFERENCE_ENV/bin/python" \
  requirements/deepseek-reference.lock
scripts/with-env.sh "$REFERENCE_ENV/bin/python" \
  -m development.model.deepseek_v41.reference \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$REFERENCE_OUTPUT"
```

## 数学检查

将 `REFERENCE_PYTHON` 设为固定参考环境中的 Python 可执行文件.
以下小型检查仅使用 CPU 张量:

```bash
scripts/with-env.sh "$REFERENCE_PYTHON" -m unittest \
  development.model.deepseek_v41.test_reference_math \
  development.model.deepseek_v41.test_reference_attention \
  development.model.deepseek_v41.test_reference_model \
  development.model.deepseek_v41.test_reference_dspark
```

这些检查对混合方向, 归一化, 旋转元素对, attention sinks 和 routing bias 使用手算数值.
它们不是全模型数值凭据.
普通 CPU contract hook 不导入 PyTorch.
后续 GPU 参考生成遵循 [GPU 执行规则](../execution.zh.md).

## DSpark 参考

[预测器](../../development/model/deepseek_v41/reference_dspark.py) 提供三个 FP64 阶段和五行双向草稿.
`append_main` 投影原始目标特征, 仅更新三个有界主窗口.
`forward` 追加一个显式边界行, 并返回完整基础 logits, 加入 Markov 偏置的 logits, 已选 IDs 和原始置信度.
可选的固定提案 IDs 支持有条件的端点检查.
默认选择使用较小 ID 优先的 argmax, 而不是随机服务采样.
每次调用均不修改其输入状态.

预测器注意力位置和输出采样位置相差一.
待处理 tokens, 验证, 已接受特征更新, RNG 和精确修正遵循[采样约定](sampling.zh.md).
其 NumPy 参考不使用模型权重或 native 算子.
这些小型 CPU 检查不是本地 checkpoint 的 DSpark 端点 fixtures.
