# DeepSeek 固定视频提示词夹具

## 范围

[视频编码模块](../../development/model/deepseek_v41/video_codec.py) 提供三个补充提示词用例.
[原有二十个编码用例](fixtures.zh.md#提示词编码) 保持不变.
这些用例检查[视频辅助程序](video.zh.md) 展开所选帧之后的文本, token IDs 和图片记录顺序.
静态图片和帧图片使用不透明的夹具 URL.
不打开图片字节.

这些夹具不解码视频, 不处理图片像素, 不展开图片跨度, 也不计算模型输出.
它们不能证明服务, 数值, 性能或容量通过验收.
解码器集成, 长上下文参考和数值限制待完成.

## 固定身份

[冻结产物](../../development/model/deepseek_v41/video_codec.json) 包含 24704 bytes.
其 SHA256 为 `21553a1d0c51f5361a202ed34048c8d4b39fab2cb8a64856699c653439975f7b`.
它使用 schema 1 和 reference kind `local-python-video-prompt-codec-cpu`.
它绑定完整检查点清单 SHA256 `0f42736d2943ec4725cb33d9e860a10ed994ba1d33e50d79c191847d3facea67`.

生成过程按照该清单验证使用的编码器, 分词器和分词器配置字节.
它执行经验证的编码器字节, 并通过 `Tokenizer.from_buffer(verified_bytes)` 构建分词器.
它禁用自动特殊 token 插入, 填充和截断.
源编码器为没有 context 的请求提供 BOS.
生成过程不重新扫描权重载荷.

产物存储完整清单, 所用文件身份, 八个历史源码身份和依赖版本.
它还存储完整输入, 视频元数据, 所选索引, 期望图片顺序, 渲染文本, token IDs 和返回的媒体.
三个 JSON 哈希绑定用例输入, 完整用例输出和源码记录.
这些哈希使用排序后的键, 分隔符 `(",", ":")`, `ensure_ascii=True` 和 `allow_nan=False`, 不含末尾换行.

`load_video_codec()` 仅使用 Python 标准库.
它先检查固定文件哈希, 再读取 JSON.
它检查内嵌清单, 所用文件绑定, JSON 哈希, 分词选项, 用例边界和有序媒体.
它不导入生成器辅助程序, 分词器, NumPy, 参考计算或推理运行时.
历史源码记录标识构建过程, 不会将其与未来的实现文件比较.
加载器固定文件哈希的变更不会改变已记录的历史生成器身份.

## 用例

每个用例包含四个图片记录和四个图片占位 token.
图片占位 token ID 为 129264.
三个完整 token 序列共包含 131 tokens.

| 用例 | 内容和媒体顺序 | Tokens |
|---|---|---:|
| `mixed-still-video` | 文本, 静态图片, 文本, 两对时间戳/帧, 文本, 静态图片, 文本 | 40 |
| `rounded-timestamp-ties` | 文本, 四对时间戳/帧, 文本 | 46 |
| `context-video-continuation` | 混合提示词及其助手回复之后的重复时间戳帧提示词 | 45 |

混合用例以每秒两帧选择源帧 0 和 1.
其时间戳为 `<0.0 seconds>` 和 `<0.5 seconds>`.
返回的图片顺序为 `still-before`, `mixed-frame-0`, `mixed-frame-1` 和 `still-after`, URL 前缀为 `fixture://`.

重复时间戳用例以每秒 100 帧选择全部四个源帧.
索引 0, 1, 2 和 3 均格式化为 `<0.0 seconds>`.
四个不同记录保持源顺序.
其 URL 名称为 `tied-frame-0` 到 `tied-frame-3`.
相同的时间戳文本不会合并帧记录.

续写用例将此前的混合提示词和助手回复保留在显式 context 中.
渲染文本和返回图片仅包含新一轮内容.
其 tokens 等于重复时间戳用例移除初始 BOS token 之后的 tokens.
调用方分别保留此前的文本和媒体.

## 构建和检查

使用 `requirements/deepseek-media-reference.lock` 对应的环境.
将 `MEDIA_REFERENCE_PYTHON` 设为其 Python 可执行文件.
将 `MODEL_ROOT` 设为本地检查点目录.
将 `CHECKPOINT_MANIFEST` 设为上述身份对应的完整清单.
将 `VIDEO_CODEC_OUTPUT` 设为新的外部 JSON 路径.
生成补充产物:

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$MEDIA_REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.video_codec \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$VIDEO_CODEC_OUTPUT"
```

固定用例集不接受外部用例输入.
每个输入最多 32768 JSON bytes.
每段渲染文本最多 65536 UTF-8 bytes.
每个用例最多 4096 tokens 和八个图片记录.
总 token 上限为 8192, 产物上限为 1 MiB.
所用文件遵循共享编码读取器的 64 MiB 上限.

生成过程在发布前检查所用文件和已记录源码是否改变.
它在输出目录中写入并刷新临时文件.
排他链接发布完整文件, 不执行替换.
已有或竞争创建的输出保持不变.
临时文件在成功或失败后移除.
仅在有意更新夹具时, 将已审查的外部字节复制到仓库中.

运行聚焦检查:

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$MEDIA_REFERENCE_PYTHON" \
  -m unittest development.model.deepseek_v41.test_video_codec
```

检查包含固定值, 图片顺序, 重复时间戳, context, 经验证字节的生成, 损坏, 变更, 边界和输出所有权.
它们不运行解码器或模型.
