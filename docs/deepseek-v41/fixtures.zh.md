# DeepSeek 固定参考夹具

## 范围和身份

[夹具读取器](../../development/model/deepseek_v41/fixtures.py) 用 SHA256 绑定四个固定产物.
普通检查无需 PyTorch, GPU 执行或访问检查点即可加载它们.
它们是参考输入和期望输出, 而不是服务性能结果.
不要用运行时输出替换它们.
模型和算子数值限制是[验证计划](validation.zh.md) 中的独立门槛.

| 夹具 | 内容 | SHA256 |
|---|---|---|
| [短文本](../../development/model/deepseek_v41/reference.npz) | 两个完整 FP64 词表向量 | `3bda8a9f03c797e65222a6eb85c82b758636514591f70f855e8f691a1f1fead0` |
| [提示词编码](../../development/model/deepseek_v41/codec.json) | 二十个带完整 token IDs 的编码用例 | `30a3079291e9326961f3d79937fe25bac3235f49ad79c250edc6763619fe7a75` |
| [DSpark 续算](../../development/model/deepseek_v41/dspark.npz) | 十八个完整 FP64 词表行和两个置信度向量 | `c78d89d66d2ff38999af2aa44b845c6d2fea1702280c217a92e7a0d693ee6fa8` |
| [媒体端点](../../development/model/deepseek_v41/media.npz) | 静态图片和视频帧图片之后的两个完整 FP64 词表向量 | `a847254bb8a1b858d67eeb7405edb4a5435c24f2280585ed18f6b4b0eee86223` |

四个产物均绑定检查点清单 SHA256 `0f42736d2943ec4725cb33d9e860a10ed994ba1d33e50d79c191847d3facea67`.
该清单记录 48 个本地权重分片及已标识控制文件的完整身份.
产物还记录各自的生成器源码和依赖版本.
固定产物哈希将这些记录与输出绑定.
后续源码变更不会在测试期间重新生成期望值.

## 短文本

[构建指南](reference.zh.md) 规定 FP64 方程和固定提示词.
产物包含全部 129280 个 prefill logits 和全部 129280 个续算 logits.
prefill argmax 选择 token 16 作为固定续算输入.
续算使用相同的 prefill 状态, 而不是全新的模型状态.

元数据记录全部 16 个提示词 token IDs, 数组哈希, 检查点身份, 源码哈希和 CPU 执行设置.
权重解码和后续全部模型算术使用 FP64, 不进行激活量化.
此产物不覆盖 DSpark 端点, 媒体, 长上下文或 native 舍入.
该产物的存在不能证明运行时通过数值验收.

## DSpark 续算

`load_dspark()` 仅通过 NumPy 加载固定的 17870563-byte `dspark.npz`.
该文件包含已验证外部 reference 的原始字节.
它使用 schema 1, 用例 `dspark-restored` 和 reference kind `independent-pytorch-fp64-no-activation-quantization`.
加载器先将完整文件哈希与固定身份比较, 再读取数组数据.
它还验证存档的完整检查点清单, 固定输入, 状态选择, 形状, 有限 FP64 值和数组元数据哈希.
其形状契约独立固定, 不随后续生成器变更而改变.

输入前缀有 129 tokens.
它将 16-token 短提示词重复八次, 再添加该提示词的第一个 token.
目标模型 prefill 选择待处理 token 4787.
前五个提议为 `[13769, 46012, 54994, 1060, 270]`.
六个验证输入为 `[4787, 13769, 46012, 54994, 1060, 270]`.

| 数组 | 形状 |
|---|---|
| `prefill` | `(129280,)` |
| `draft_logits` | `(5, 129280)` |
| `draft_confidence` | `(5,)` |
| `verify_logits` | `(6, 129280)` |
| `restored_logits` | `(129280,)` |
| `next_draft_logits` | `(5, 129280)` |
| `next_draft_confidence` | `(5,)` |

全部十八个 logit 行包含完整词表.
两个置信度向量各包含五个原始 FP64 值.
它们独立于词表 logits.

该用例从验证中强制选择两个输入的前缀, 相当于接受一个草稿.
它独立重算该前缀, 而不是使用被拒绝的验证状态.
目标模型 argmax 提供修正 token 46012.
下一个待处理位置为 131, 位置从零开始计数.
后续五个提议为 `[54994, 1060, 270, 41638, 6397]`.
下一个草稿块仅使用已接受的特征, 目标模型续算使用选中的状态.

此用例不是一次采样拒绝事件.
它提供独立 CPU FP64 参考值, 而不是服务 DSpark 执行.
长上下文端点和模型专用数值阈值是后续门槛.
Phase 0 尚未完成.

文件哈希将历史生成器源码和依赖记录与这些输出绑定.
普通检查不将这些源码记录与当前生成器文件比较.
后续生成器编辑不会改变此固定 reference.
[构建指南](reference.zh.md) 给出生成器及其源码检查.

## 媒体端点

`load_media()` 仅通过 NumPy 加载固定的 2012626-byte `media.npz`.
该文件逐字节复制已审查的外部 reference.
它使用 schema 1, 用例 `still-and-video-frames` 和 reference kind `independent-pytorch-fp64-no-activation-quantization`.
加载器先检查完整文件哈希, 再读取数组.
它还检查内嵌的完整检查点清单, 完整输入身份, 历史源码记录和压缩 token 映射身份.

输入身份包含消息, 编码器选项, 渲染文本, PNG 字节, token IDs, token 类型, 图片网格和 patch 身份.
它还包含解码 token, 帧选择和视觉嵌入身份.
其 SHA256 为 `968e863392fe7ec3531e4df6a7d43851a295f070c486fec80565abc63a7449bc`.
21 个历史源码记录的 SHA256 为 `c099eb57db022052006567e726dbd3cd4f56c5bebd35eaf7bce45a3ed2383dd7`.

两个身份均使用 UTF-8 JSON, 键排序, 分隔符 `(",", ":")` 和 `allow_nan=False`, 不含末尾换行.
JSON 使用 `ensure_ascii=True`.
测试不将历史源码记录与当前实现文件比较.
完整文件哈希还绑定已记录的依赖版本和 CPU 设置.

37-token 占位符提示词展开为 624 tokens.
三个视觉嵌入按图片顺序具有形状 `(200, 5120)`, `(206, 5120)` 和 `(184, 5120)`.
prefill argmax 选择解码 token 9544.
解码从已接受的 prefill 状态继续.

| 数组 | 形状 | 小端 FP64 字节的 SHA256 |
|---|---|---|
| `prefill` | `(129280,)` | `7edd90ed9122b9dab48dffbcc231fd06a284591bc4f04cfbf65ecf2384edfc38` |
| `decode` | `(129280,)` | `b97f1fd3e53a8e90145a5db261ffa44b78228beb427f337145836336422b6c31` |

加载器检查所有数组都存在, 使用 FP64, 并具有有限值.
其形状, dtype 和哈希元数据必须与数组一致.
[媒体构建指南](media-reference.zh.md) 规定合成的已解码静态图片和视频帧.
此夹具不能证明解码器, 服务, 性能或容量通过验收.
长上下文参考和数值限制仍待完成.

## 提示词编码

[编码生成器](../../development/model/deepseek_v41/codec_reference.py) 执行经过验证的本地 Python 编码源码.
它先对照完整检查点清单验证所用编码器, 分词器和分词器配置的字节.
它不重新计算权重载荷哈希.
发布输出之前, 它检查输入文件和生成器文件状态.
发布不能替换已有输出文件.

分词通过 `Tokenizer.from_buffer` 使用已验证的字节, 而不是重新读取路径.
禁用自动特殊 token 插入, 填充和截断.
Python 编码器在适用位置提供 BOS.
每个用例存储输入, 渲染文本, 完整 IDs 和返回的有序媒体记录.
生成器不获取图像数据, 也不执行图像预处理.

用例覆盖聊天, 推理强度别名及数值, 推理移除, 推理保留, 工具, 系统消息, 助手续写和图像.
工具对话按本地规则保留此前推理.
显式提供 context 时, 编码器仅返回续写文本及其中的新图像.
调用方分别保留此前上下文和媒体.
这些夹具不证明 Jinja 等价性.
已标识的 Python 编码器仍是权威来源.

### 工具图像顺序修正

原始 `tool-result-images` 用例暴露一个本地源码缺陷.
编码器在按调用顺序排序工具结果之前收集图像.
以相反顺序提供结果时, 渲染的占位符和返回的图像记录对应不同结果.
此用例记录源行为用于诊断, 而不是获准的服务行为.

`tool-result-images-canonical` 用例在编码前按调用顺序提供完整工具结果记录.
它具有相同的渲染文本和 token IDs, 以及正确的图像顺序.
服务适配器必须在提取媒体之前规范化完整工具结果记录.
它必须拒绝缺失, 重复或未知的调用身份, 不能只静默移动图像记录.
此修正改变源实现的顺序, 不改变语言模型或图像模型算术.

## 上游检查点边界

DeepInfer 修订版 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` 默认使用 `-1017` 检查点.
比较使用[检查点修订版 1bae0e1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170).
其配置和张量索引的字节与本地文件相同.
48 个分片的大小也相同.
每个上游 Git LFS 整文件 SHA256 都与对应本地分片哈希不同.

相同布局不能证明张量值相同.
不同的整文件哈希也不能指出哪些张量值不同.
未批准上游检查点替换或数值结果复用.
所有运行时权重和参考均使用本地检查点身份.
将上游元数据和比较记录保存在仓库之外.

## 命令

运行固定产物和生成器检查:

```bash
scripts/with-env.sh python -m unittest \
  development.model.deepseek_v41.test_fixtures \
  development.model.deepseek_v41.test_reference_cases \
  development.model.deepseek_v41.test_codec_reference
```

将 `MODEL_ROOT`, `CHECKPOINT_MANIFEST` 和 `CODEC_OUTPUT` 设为本地检查点, 已验证清单和新的外部 JSON 路径.
生成编码产物:

```bash
scripts/with-env.sh python -m development.model.deepseek_v41.codec_reference \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$CODEC_OUTPUT"
```

有意替换夹具之前, 审查源码变更和生成身份.
不要在数值验收期间重新生成夹具.
