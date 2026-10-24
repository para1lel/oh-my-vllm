# DeepSeek 参考构建

## 范围

[阶段计划](plan.zh.md)控制 native 算子调优之前的参考构建.
这些工具提供 checkpoint 身份, 解码后的权重, Engram 哈希, FP64 目标模型和预测器数学实现, 以及 CPU 文本参考生成器.
[固定夹具](fixtures.zh.md) 提供短文本, DSpark 恢复, 媒体和长上下文端点, 以及提示词编码用例.
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

[数学函数](../../../development/model/deepseek_v41/reference_math.py)实现 hyper-connections, 相邻元素对旋转变换, sparse attention sinks 和 routing.
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

独立的[量化 oracle](../../../development/model/deepseek_v41/reference_quantization.py) 仅使用 NumPy 检查 activation 和 cache 转换.
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

[生成器](../../../development/model/deepseek_v41/reference.py)将固定的 16-token 提示绑定到本地分词器.
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
后续 GPU 参考生成遵循 [GPU 执行规则](../../execution.zh.md).

## DSpark 参考

[预测器](../../../development/model/deepseek_v41/reference_dspark.py) 提供三个 FP64 阶段和五行双向草稿.
`append_main` 投影原始目标特征, 仅更新三个有界主窗口.
`forward` 追加一个显式边界行, 并返回完整基础 logits, 加入 Markov 偏置的 logits, 已选 IDs 和原始置信度.
可选的固定提案 IDs 支持有条件的端点检查.
默认选择使用较小 ID 优先的 argmax, 而不是随机服务采样.
每次调用均不修改其输入状态.

预测器注意力位置和输出采样位置相差一.
待处理 tokens, 验证, 已接受特征更新, RNG 和精确修正遵循[采样约定](sampling.zh.md).
[用例生成器](../../../development/model/deepseek_v41/reference_cases.py) 构建[固定恢复夹具](fixtures.zh.md#dspark-续算).
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

[媒体生成器](../../../development/model/deepseek_v41/reference_media_case.py) 提供[固定媒体端点](fixtures.zh.md#媒体端点).
三张合成 RGB 图片尺寸为 17 乘 11, 11 乘 17 和 13 乘 13 像素.
第一张为静态图片. 其余两张代表 `<0.0 seconds>` 和 `<0.5 seconds>` 的视频帧.
该用例不测试视频解码或帧提取.

[视觉参考](../../../development/model/deepseek_v41/reference_vision.py) 包含已学习的起始, 换行和结束 embeddings.
目标模型在 hyper-connection 展开前, 按提示词顺序插入完整图片跨度.
图像位置不接收 Engram 贡献, 并使用视觉路由偏置.
仅第一次 prefill 调用插入图片.
Decode 使用其已接受状态, 不加入新图片.
元数据绑定 PNG 字节, 提示词, patches, 视觉 embeddings, token map 和完整端点向量.

使用 `requirements/deepseek-media-reference.lock` 对应的独立环境.
将 `MEDIA_REFERENCE_PYTHON` 设为其 Python 可执行文件.
将 `MEDIA_OUTPUT` 设为新的外部 NPZ 路径.

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$MEDIA_REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.reference_media_case \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$MEDIA_OUTPUT"
```

默认使用八个 CPU 线程, projection-row 上限为 2048.
小规模 `test_reference_media_case` 检查不重新生成完整模型端点.

## 长上下文参考

[长上下文生成器](../../../development/model/deepseek_v41/reference_long_case.py) 将短提示词重复至 65537 tokens.
它在 16385 和 65537 tokens 处存储完整 prefill 和 decode 向量.
这些长度跨过第 20 层候选池的 2048 个八位置块, 以及原始 65536-token 旋转位置编码上下文.
它们不能证明 1048576-token 容量或 64 个请求准入.

连续 prefill 分块最多包含 4096 tokens, 并在每个端点处准确结束.
全部来源 KV, 窗口, 部分压缩器和 Engram 历史在分块间延续.
不采用位置偏移, 历史截断或近似重放.
每个端点的 argmax 在独立分支上选择一个 decode token.
该分支不成为下一个端点提示词的一部分.
`load_long()` 通过[夹具读取器](../../../development/model/deepseek_v41/fixtures.py) 加载四个固定 FP64 数组.

使用现有的 `REFERENCE_PYTHON`, 模型和 manifest 变量.
将 `LONG_OUTPUT` 和 `LONG_STATE` 设为不同的外部 NPZ 路径.

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.reference_long_case \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$LONG_OUTPUT" --checkpoint-state "$LONG_STATE" \
  --threads 8 --projection-row-limit 2048 --chunk-size 4096
```

这些设置与冻结产物一致. 不同分块大小可能改变归约顺序.
每个完成的分块和端点分支保存完整续算状态和已完成端点.
身份变化, 形状无效或数据损坏会导致错误.
工具在原子替换前同步文件, 并保留 `<stem>.previous.npz`.
它不会自动回退.

中断后重复相同命令.
如需选择上一版本状态, 将 `LONG_STATE` 设为该路径.
不使用 `--checkpoint-state` 时不保存续算状态.
状态检查点不是数值夹具.
小规模 `test_reference_long_case` 检查覆盖续算, 身份, 数据损坏和发布失败.

## 发布源码基线

独立的[捕获工具](../../../development/model/deepseek_v41/source_capture.py) 记录经过验证的发布源码产生的短上下文 FP32 logits.
它使用固定的短提示词和续算输入.
[加载器](../../../development/model/deepseek_v41/source_weights.py) 从原始 checkpoint 读取有界切片.
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
