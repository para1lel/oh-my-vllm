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
配置解析和分词器构建使用经过验证的字节缓冲区, 而不是重新读取路径.
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
  development.model.deepseek_v41.test_reference_dspark \
  development.model.deepseek_v41.test_reference_vision
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

## DSpark 恢复生成

[用例生成器](../../development/model/deepseek_v41/reference_cases.py) 重复固定短提示词的 token IDs.
此前缀具有 129 个 tokens.
此前缀跨越 128-token 窗口边界, 并留下一个未满的 ratio-2 压缩器组.
目标模型计算此前缀, 然后为五行 DSpark 提案提供特征.
验证消费待处理 token 和全部五个提案.

参考从原始前缀状态单独计算前两个验证输入.
它对照完整验证中对应行检查特征和 logits.
其目标 argmax 提供固定修正输入.
这是强制状态选择用例, 而不是随机拒绝事件.
[采样参考](sampling.zh.md) 单独检查拒绝概率.

下一次提案前, 仅已接受的目标特征更新预测器主窗口.
下一个边界特征仅追加一次.
目标模型随后从所选状态计算修正输入.
产物包含 prefill, 两个提案块, 验证和恢复解码的共 18 个完整词表行.
它还包含每个提案块的五个原始置信度值.

生成器记录检查点身份, 源码身份, 依赖版本, 全部输入 IDs 和全部输出数组哈希.
发布前, 它检查所用控制字节和输入状态.
排他链接发布完整的外部 NPZ 文件, 不替换已有文件.
目前尚无固定的恢复产物.

将 `CASE_OUTPUT` 设为新的外部 NPZ 路径.
使用固定参考环境:

```bash
scripts/with-env.sh "$REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.reference_cases \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$CASE_OUTPUT"
```

## 视觉数学

[视觉参考](../../development/model/deepseek_v41/reference_vision.py) 接受带显式 patch 网格的已归一化 FP64 patches 或已归一化 FP64 RGB pixels.
它计算视觉层输出, 使用分半 2D RoPE 和完整双向注意力.
合并器使用零填充的 3-by-3 分组, 通道优先的收集顺序和精确 erf GELU.
学习得到的起始, 换行和结束 embeddings 构成完整的语言模型图像跨度.

Query 分块限制分数存储, 不丢弃 keys.
权重读取按 projection 行设置上界.
参考使用 CPU FP64 算术, 不进行源实现的 BF16/FP32 舍入.
它不包含图像加载, resize, 归一化, 视频帧选择或语言模型图像插入.
小型数学检查不提供固定的本地检查点媒体端点.

## 图像准备

[图像参考](../../development/model/deepseek_v41/reference_image.py) 接受编码后的图像字节, 而不是路径或 URLs.
其网格计算使用源实现的最小像素面积, patch 对齐舍入, token 上限和可选的宽图像 resize 规则.
Pillow 提供源实现的 RGB 转换, resize 和灰色填充.
不进行 EXIF 旋转.
输出包含 FP64 归一化 patches, 以及有序的起始, 图像, 换行和结束 token 类型.
FP64 归一化不包含源实现的 FP32 算术和最终 BF16 转换.

此工具拒绝超过 64 MiB 的编码输入, 超过 67108864 pixels 的解码图像, 以及超过 512 MiB 的准备后 FP64 数组.
这些是参考资源上限, 不是新的服务媒体约定.
它不执行视觉或语言模型.

组合图像和视觉检查时, 使用 `requirements/deepseek-media-reference.lock` 指定的独立环境.
其版本与文本参考锁定文件一致, 另加入 Pillow 12.3.0.
不要修改正在生成参考的进程使用的包.
将 `MEDIA_REFERENCE_ENV` 设为新的外部环境目录:

```bash
scripts/with-env.sh uv venv "$MEDIA_REFERENCE_ENV"
scripts/with-env.sh uv pip sync --python "$MEDIA_REFERENCE_ENV/bin/python" \
  requirements/deepseek-media-reference.lock
scripts/with-env.sh "$MEDIA_REFERENCE_ENV/bin/python" -m unittest \
  development.model.deepseek_v41.test_reference_image \
  development.model.deepseek_v41.test_reference_vision
```

## 媒体提示展开

[提示展开辅助函数](../../development/model/deepseek_v41/reference_media.py) 接受 tokenizer IDs 和有序的编码图像字节.
每个图像占位符变为一个完整的源图像跨度.
跨度内每个位置保持配置的图像 token ID, 通过独立类型区分起始, 图像, 换行和结束.
文本位置的类型为 -1.
辅助函数保持图像顺序, 并记录每个跨度的起始位置.
它拒绝图像数量不匹配, 缺少视觉支持, 无效 token IDs, 以及展开后超过上下文容量的提示.

此参考辅助函数的合计 patch 存储上限为 512 MiB.
此上限不包含单张图像准备所需的临时存储.
展开后的提示 IDs, 类型和 patch 数组均为只读.
辅助函数不运行视觉模型, 不插入 embeddings, 也不预留输出 tokens.
调用者必须在服务接纳前确保输入输出合计长度不超过其上限.
