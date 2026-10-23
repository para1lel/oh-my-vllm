# DeepSeek 参考构建

## 范围

[阶段计划](plan.zh.md)控制 native 算子调优之前的参考构建.
这些工具提供 checkpoint 身份, 解码后的权重, Engram 哈希, FP64 目标模型和预测器数学实现, 以及 CPU 文本参考生成器.
[固定夹具](fixtures.zh.md) 提供短文本, DSpark 恢复和媒体端点, 以及提示词编码用例.
目前尚未固定长上下文端点夹具.
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

## 源精度量化

独立的[量化 oracle](../../development/model/deepseek_v41/reference_quantization.py) 仅使用 NumPy 检查 activation 和 cache 转换.
它不替代 FP64 数学参考, 也不设置复合算子的验收限值.
`activation_fp8`, `indexer_fp4` 和 `compressed_kv_fp4` 接受 FP32 数组, 先将输入舍入到 BF16.
它们返回编码字节, 尺度字节, 以及用 FP32 表示的 BF16 量化往返数值.

每组分别包含 32, 32 和 16 个通道.
UE8M0 尺度计算使用 FP32 倒数乘法.
E4M3 尺度计算使用 FP32 除法.

转换采用最近舍入, 中点取偶数.
Packed 通道 `2j` 位于低 nibble.
非有限输入, BF16 溢出, 无效形状和不支持的输入 dtype 均导致错误.
该模块记录发布版 kernel 的源文件身份.

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
  development.model.deepseek_v41.test_reference_media_model \
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
[用例生成器](../../development/model/deepseek_v41/reference_cases.py) 构建[固定恢复夹具](fixtures.zh.md#dspark-续算).
该夹具指南规定输入, 状态选择和完整输出.
生成器在排他发布前检查所用控制字节, 文件状态和源码身份.

将 `CASE_OUTPUT` 设为新的外部 NPZ 路径.
使用固定参考环境:

```bash
scripts/with-env.sh "$REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.reference_cases \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$CASE_OUTPUT"
```

## 媒体参考

固定输入, 完整端点, 环境选择和生成命令见[媒体指南](media-reference.zh.md).
[图像辅助函数](../../development/model/deepseek_v41/reference_image.py) 提供归一化 FP64 patches 和有序的图像跨度类型.
[视觉参考](../../development/model/deepseek_v41/reference_vision.py) 提供包含已学习起始, 换行和结束行的完整 embeddings.
[提示辅助函数](../../development/model/deepseek_v41/reference_media.py) 在展开占位符时保持图像顺序.
[目标参考](../../development/model/deepseek_v41/reference_model.py) 在 hyper-connection 展开前插入这些 embeddings.

图像位置不接收 Engram 贡献, 并使用视觉路由偏置.
仅允许在第一次 prefill 调用期间插入.

参考资源上限不改变服务媒体约定.
调用者必须在服务接纳前预留输出容量.
不要修改正在运行的参考生成器所用的环境或源文件.

## 发布源码基线

独立的[捕获工具](../../development/model/deepseek_v41/source_capture.py) 记录经过验证的发布源码产生的短上下文 FP32 logits.
它使用固定的短提示词和续算输入.
[加载器](../../development/model/deepseek_v41/source_weights.py) 从原始 checkpoint 读取有界切片.
它不创建另一份转换后的 checkpoint.
模型参数来自经过验证的根配置字节.
捕获记录一项内存修正, 在每次 forward 调用前发布 index owner 的 cache.

此基线使用发布版 MP4 放置: 专家归属各 rank, 共享专家和视觉权重复制, Engram 切片驻留设备.
它不实现框架的 TP4, TEP4, DEP4 或 pinned-host Engram 约定.
完整向量误差, 余弦相似度和 top-1 一致性属于诊断数据.
它们不设定复合数值限值, 也不替代 FP64 参考.

使用 `requirements/deepseek-baseline.lock` 指定的独立环境.
将 `BASELINE_ENV` 设为该外部环境目录.
将 `BASELINE_OUTPUT` 设为新的外部 NPZ 路径.
使用短文本生成一节中的模型和 manifest 变量.
将设备选择设置保存在 `LOCAL.md`.
通过四 GPU wrapper 运行命令:

```bash
scripts/with-env.sh uv venv "$BASELINE_ENV"
scripts/with-env.sh uv pip sync --python "$BASELINE_ENV/bin/python" \
  requirements/deepseek-baseline.lock
PYTHONDONTWRITEBYTECODE=1 OH_MY_VLLM_GPU_COUNT=4 \
  scripts/with-env.sh scripts/with-gpu.sh \
  "$BASELINE_ENV/bin/python" -B -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  -m development.model.deepseek_v41.source_capture \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$BASELINE_OUTPUT" --run-gpu
```

发布前, 四个 rank 的完整向量必须一致.
源码, checkpoint 和环境身份必须保持不变.
JSON 文件包含来源信息和诊断数据.
NPZ 文件是完成标记.
不会替换已有输出.
