# DeepSeek 媒体参考生成

## 范围

[媒体生成器](../../development/model/deepseek_v41/reference_media_case.py) 计算包含一张静态图片和两张视频帧图片的完整 CPU FP64 语言模型端点.
输入帧是合成的已解码图片, 不是压缩视频文件.
此用例不验证视频解码器, 帧提取或服务的媒体处理.
[视频契约](video.zh.md) 规定帧选择和时间戳块.

生成器提供参考构建, 不提供运行时验收.
已审查的端点现存于[固定媒体夹具](fixtures.zh.md#媒体端点).
普通检查通过 NumPy 加载此夹具, 无需生成参考或访问检查点.
该夹具不能证明服务数值, 性能或容量通过验收.
长上下文参考输出和模型专用数值限制仍待完成.
[参考指南](reference.zh.md) 规定算术, 权重流式读取和独立参考环境.

## 固定输入

三张非均匀 RGB 图片的尺寸分别为 17 乘 11, 11 乘 17 和 13 乘 13 像素.
其 PNG 字节以 base64 元数据保存, 并附带 SHA256 身份.
第一张是静态图片.
其余图片是每秒两帧的两张合成视频帧.
时间戳块为 `<0.0 seconds>` 和 `<0.5 seconds>`.

经验证的本地提示词编码器提供渲染文本和有序图片记录.
生成器拒绝不同的图片顺序.
分词器使用经验证的检查点字节, 不自动填充或截断.
图片辅助程序把每个占位符展开为完整图片跨度.
提示词必须为解码保留一个位置.

元数据保留消息, 编码器选项, 渲染文本, 占位符 ID, 展开后的 ID, token 类型, 图片网格和 patch 身份.
它还保留帧索引和合成帧的范围限制.

## 计算和发布

视觉参考按提示词顺序计算全部三个图片跨度.
其中包含学习到的起始, 换行和结束嵌入.
目标模型参考在第一次预填充调用中插入这些嵌入.
最高预填充 logit 选择一个解码输入.
解码从已接受的预填充状态继续, 不再输入图片.
每个端点包含全部 129280 个 FP64 logits.

生成器在计算前重新计算完整检查点身份.
它按该身份验证每个使用的控制文件.
它记录参考源码哈希, 依赖版本, 分词器映射身份, 视觉嵌入身份和端点身份.
直到发布时, 检查点文件状态和参考源码身份都必须保持不变.
排他链接发布完整的外部 NPZ 文件, 不替换已有文件.
无效端点形状, 非有限值和已存在的输出路径都会导致错误.

生成期间不得修改参考源码文件或环境.
将原始产物保存在仓库外.
仅在独立身份和输出审查后添加固定夹具.

## 命令

使用 `requirements/deepseek-media-reference.lock` 对应的独立环境.
将 `MEDIA_REFERENCE_PYTHON` 设为其 Python 可执行文件.
将 `MODEL_ROOT` 设为本地检查点目录.
将 `CHECKPOINT_MANIFEST` 设为其完整身份清单.
将 `MEDIA_OUTPUT` 设为新的外部 NPZ 路径.

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$MEDIA_REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.reference_media_case \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$MEDIA_OUTPUT"
```

此命令仅使用 CPU 执行.
默认配置使用八个 CPU 线程, 投影行上限为 2048.
小型构建检查不生成完整检查点端点:

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$MEDIA_REFERENCE_PYTHON" \
  -m unittest development.model.deepseek_v41.test_reference_media_case
```
