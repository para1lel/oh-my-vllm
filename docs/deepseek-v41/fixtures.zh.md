# DeepSeek 固定参考夹具

## 范围和身份

[夹具读取器](../../development/model/deepseek_v41/fixtures.py) 用 SHA256 绑定固定产物.
普通检查无需 PyTorch, GPU 执行或访问检查点即可加载它们.
它们是参考输入和期望输出, 而不是服务性能结果.
不要用运行时输出替换它们.
模型和算子数值限制是[验证计划](validation.zh.md) 中的独立门槛.

| 夹具 | 内容 | SHA256 |
|---|---|---|
| [短文本](../../development/model/deepseek_v41/reference.npz) | 两个完整 FP64 词表向量 | `3bda8a9f03c797e65222a6eb85c82b758636514591f70f855e8f691a1f1fead0` |
| [长文本](../../development/model/deepseek_v41/long-context.npz) | 16385 和 65537 tokens 处的四个完整 FP64 向量 | `001cc434d3adb106fc490286d3a9625b18326b83d5fcca7deb83a553b11ea474` |
| [提示词编码](../../development/model/deepseek_v41/codec.json) | 二十个带完整 token IDs 的编码用例 | `30a3079291e9326961f3d79937fe25bac3235f49ad79c250edc6763619fe7a75` |
| [DSpark 续算](../../development/model/deepseek_v41/dspark.npz) | 十八个完整 FP64 词表行和两个置信度向量 | `c78d89d66d2ff38999af2aa44b845c6d2fea1702280c217a92e7a0d693ee6fa8` |
| [媒体端点](../../development/model/deepseek_v41/media.npz) | 静态图片和视频帧图片之后的两个完整 FP64 词表向量 | `a847254bb8a1b858d67eeb7405edb4a5435c24f2280585ed18f6b4b0eee86223` |
| [精确算子](../../development/model/deepseek_v41/exact-operators.npz) | 四个合成量化和 Engram 用例 | `a422c1f4cadeeb4ef918a18253834eae848c32b5173b0b88398e9bf0ede23e79` |
| [Engram lookup](../../development/model/deepseek_v41/engram-lookup.npz) | FP8 表, scale, 索引和 49152 个 BF16 输出位模式 | `930cee947ae1ee8bcdd3d15571732713305d849c18df8c4d98418dc309e40c2c` |
| [Engram 检查点](../../development/model/deepseek_v41/engram-checkpoint.npz) | 每个表所选 32 行的原始字节及精确 BF16 输出 | `967109d7d713191212dc37fc18335c926fa3e2420a79e9fe03f5f2bbe365efce` |
| [Hyper-connection 索引](../../development/model/deepseek_v41/hyperconnection/index.json) | 五个包含完整 FP64 输出和固定字段限值的用例 | `933323bf315a75806e9dc8f66b0da47777622cd36c8b98f262f29b3978afdb47` |
| [路由索引](../../development/model/deepseek_v41/routing/index.json) | 五个从 logits 开始, 包含精确专家 ID 和 FP64 系数的用例 | `bbb8538a912bede95a6d4364f52a63a026b04385de2e34cad5030109d305bf28` |
| [投影索引](../../development/model/deepseek_v41/projection/index.json) | 五个 A8W8 和五个 A8W4 用例, 包含完整 FP64 输出 | `db9c81bb0dd4a2f10aa3fdd289349dfb9162ede223a7ceb5cbe0e80444646fc6` |
| [启动转换](../../development/model/deepseek_v41/startup-promotion.npz) | 精确的 FP8 到 BF16 及 BF16 到 F32 转换 | `0235b26faaeb799ba13ae0388f0fb0396c76aca5a43684d6df738da381efca51` |
| [Ratio2 池化索引](../../development/model/deepseek_v41/pooling/index.json) | 四个完整成对用例, 含 FP64 输出和固定限值 | `878e5a00b671191bbd8081118217d0f0dfad790e9d345a9cd802afeafcf1fe08` |
| [稀疏注意力索引](../../development/model/deepseek_v41/sparse-attention/index.json) | 五个打包缓存用例, 含完整 FP64 输出和按请求设置的限值 | `65480d5f378455ce128018013121cdd4d784c49796e1cedfe066fde273295e4d` |

产物绑定检查点清单 SHA256 `0f42736d2943ec4725cb33d9e860a10ed994ba1d33e50d79c191847d3facea67`.
完整模型产物包含清单, 生成器身份, 固定输入, 数组形状和摘要.
精确算子包绑定清单摘要.
编码用例包含渲染文本, 完整 token IDs 和媒体记录.
读取器检查适用元数据及完整输出.
普通测试从不重新生成期望值.

## 精确算子

`load_exact_operators()` 提供四个现有算子 `Fixture` 对象.
用例覆盖 FP8/UE8M0 group32, FP4/UE8M0 group32, FP4/E4M3 group16, 以及含 dead tokens 的 Engram 分块续算.
期望值来自手工检查的字面量, 而不是候选实现输出.
按每个用例的精确限值, 比较编码字节, 尺度字节, BF16 往返位模式, 哈希和历史.
这些合成用例不设定复合数值或性能限值.

`load_engram_lookup()` 提供固定输入和独立计算的 BF16 输出位模式.
其 16 行表覆盖重复索引, 有符号零, subnormal 中点和不同 scale 组.

`load_engram_checkpoint()` 提供原始表样本, 行 ID, scale 和独立解码的输出.
全部所选值都能用 BF16 精确表示.
[检查点检查](../../development/operators/check_engram_lookup.py)使用这些固定输出验证完整共享存储.

可选 native 模块提供量化, 整数 Engram 哈希, 以及通过共享 pinned-host 映射执行的 lookup.
它们不提供服务集成.
量化调用者必须拒绝任何非零的分组状态.
Lookup 调用者必须拒绝任何非零的行状态.
Engram 返回私有续算历史, 不修改输入历史.

将 `EXACT_REPORT` 设为新的外部 JSON 路径.
构建模块.
运行共用的 Eager 和 CUDA Graph 检查:

```bash
scripts/with-env.sh python -m development.native_build \
  --module device --module quantization --module engram
OH_MY_VLLM_GPU_COUNT=1 scripts/with-env.sh scripts/with-gpu.sh \
  python -m development.operators.check_deepseek_exact --output "$EXACT_REPORT"
```

## Hyper-connections

`load_hyperconnection()` 提供 decode, prefill, verification, 零值和抵消用例.
它们使用原始 F32 权重, 四份残差, 宽度 5120, 以及 1, 128 或 384 行.
索引绑定全部数组和 `deepseek-hc-precision-v1` 策略.

| 输出 | 归一化 L2 限值 | 最大绝对误差 |
|---|---:|---|
| `collapsed`, `expanded` | `2^-8` | `max(1e-7, 2^-8 * P)` |
| `next_pre` | `2^-16` | `max(1e-7, 2^-16 * P)` |

`P` 是每个字段和请求的 FP64 峰值.
RMS 下限为 `1e-6`. 近零绝对限值为 `1e-6`.
这些工程限值不能证明完整模型精度.

[Native workspace](../../python/oh_my_vllm/operators/hyperconnection/workspace.py) 将子层前的准备与子层后的展开分开.
折叠使用传入系数. `next_pre` 属于下一个子层.
[检查器](../../development/operators/check_deepseek_hc.py)还置换通道, 以检测寻址错误.
它检查 Eager 和 Graph 执行, 非法参数, 空输入及输入不变性.
性能限值和模型集成仍待完成.

## 路由

`load_routing()` 提供目标 decode, prefill, verification, 草稿和合成决策用例.
边界从已存储的 FP32 logits 开始. 原始检查点偏置控制文本和图像的专家选择.
用例覆盖 384 个专家的 top-6 和 128 个专家的 top-3.
[模型契约](model.zh.md)规定不含偏置的系数和稳定的全局专家 ID.

系数的归一化 L2 限值为 `2^-19`.
其绝对误差限值为 `max(1e-7, 2^-19 * P)`, 其中 `P` 是每个请求的 FP64 峰值.
RMS 下限和近零绝对限值与 hyper-connections 相同.
专家 ID 必须精确匹配.

索引绑定独立的分数误差策略.
实现前, 非并列的选择和顺序必须通过该策略的区间检查.
合成并列值使用相同的已存储 logits 和偏置.
该门槛不包含投影舍入, 专家分发, 延迟和完整模型精度.

## 编码投影

`load_projection()` 提供编码 FP8 激活, UE8M0 scales 和完整 FP64 矩阵乘积.
原始检查点权重保存在外部. 合成 FP4 用例保存自身权重.
[检查器](../../development/operators/check_deepseek_projection.py)覆盖权重打包, K=576 和最大 15360 的宽度.

每个请求使用归一化 L2 `2^-7` 和最大绝对误差 `max(1e-7, 2^-6 * P)`.
`P` 是该请求的 FP64 峰值. RMS 下限和近零绝对限值为 `1e-6`.
构造的零值端点也有 `1e-6` 的绝对限值.
这些工程限值不包含激活量化误差, 提升精度后的权重格式, collectives, 延迟和完整模型精度.

`load_startup_promotion()` 覆盖全部 64770 个有限 FP8/UE8M0 输入组合, 其中包含 560 个被拒绝的溢出组合.
它提供其余 64210 个组合及全部 65280 个有限 BF16 位模式的精确输出位.
它还提供非法 BF16 位模式, 以及原始和合成 dense32 scale 用例.
转换使用 ties-to-even 舍入, 保留有符号零, 并拒绝非有限输出.
外部生成元数据的 SHA256 为 `b341dd5dcf26691c104757b29b7cd6d1ea1eecf0f5429f3f7638e5f29d7e0dcb`.

## 归一化, 旋转和激活

`load_norm_rope()` 提供四组 RMS 样例和三组 RoPE 样例.
RMS 使用无偏移的原始 BF16 权重, 文本 epsilon 为 `1e-20`, 视觉 epsilon 为 `1e-6`.
RoPE 使用传入的 FP32 余弦/正弦系数, 旋转最后 64 个通道中的相邻通道对, 并保持前缀位不变.

归一化 L2 限值为 `2^-8 + 64*2^-24`.
绝对误差限值为 `max(1e-7, (2^-8 + 64*2^-24)*P)`.
每个请求分别使用自身的峰值 `P`, RMS 下限 `1e-6` 和近零绝对误差限值 `1e-6`.
这些样例不覆盖相位生成, 二维视觉位置和完整模型验收.

`load_swiglu()` 提供四组带有 BF16 gate/up 输入和完整 FP64 输出的样例.
文本 gate 仅有上限 10. 文本 up 的下限和上限为 -10 和 10.
可选的 FP32 路由权重在最终 BF16 转换前乘到每一行.
视觉使用宽度 2816, 没有这些限值或路由权重.

SwiGLU 使用与 RMS 和 RoPE 相同的误差限值, 每个请求分别计算.
[检查器](../../development/operators/check_deepseek_activation.py)覆盖 [API](../../python/oh_my_vllm/operators/activation/api.py) 中规定范围内的输入.
投影, 分发, 量化和完整模型验收另行处理.

## Ratio2 池化

`load_pooling()` 提供宽度为 512 的完整 FP32 KV 和 gate 对.
Softmax 沿每对的两个元素计算, 各通道独立. 加权和在 RMS 之前仅转换一次为 BF16.
固定限值包含最终 BF16 舍入和考虑抵消的 FP32 误差余量.
[检查器](../../development/operators/check_deepseek_pooling.py) 覆盖 Eager 和 Graph 执行, 空输入, 输入不变性及无效参数.
此边界不覆盖投影, RMS, RoPE, 不完整组状态, 性能或完整模型精度.

## 打包稀疏注意力

`load_sparse_attention()` 提供 BF16 查询, 原始 F32 sinks, 编码缓存行和显式槽位 ID.
[API](../../python/oh_my_vllm/operators/sparse_attention/api.py) 规定格式, 掩码, 重复 IDs 和错误状态.
仅解码选中的缓存块, 不展开整个逻辑缓存.

归一化 L2 限值为 `2^-6`.
索引为每个请求的每个通道固定一个绝对误差限值.
RMS 下限和近零绝对误差限值均为 `1e-6`.
[检查器](../../development/operators/check_deepseek_sparse_attention.py) 使用全部输出, 精确零值保护, Eager/Graph 执行和非法 ID 检查.
投影, 索引选择, 状态恢复, 延迟和完整模型验收另行处理.

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

`load_dspark()` 提供十八个完整 FP64 词表行和两个各含五个值的置信度向量.
129-token 前缀将短提示词重复八次, 再添加其第一个 token.
数组覆盖 prefill, 五个提议, 六个验证输入, 恢复后续算和后续五个提议.

该用例选择两个验证输入, 相当于接受一个草稿.
它独立重算此前缀, 不使用被拒绝状态.
下一个草稿块仅使用已接受特征.
此强制选择不是采样拒绝事件.
[构建指南](reference.zh.md#dspark-参考) 提供生成器和源码检查.

## 媒体端点

`load_media()` 提供一张合成静态图片和两张已解码视频帧的完整 FP64 prefill 与续算 logits.
37-token 占位符提示词展开为 624 tokens.
prefill argmax 选择 token 9544, 从已接受状态续算.
元数据绑定完整提示词, 有序图片, patches, 视觉嵌入和 token 映射.

[参考指南](reference.zh.md#媒体参考) 规定生成流程.
此夹具不能证明解码器, 服务, 性能或容量通过验收.

## 提示词编码

[编码生成器](../../development/model/deepseek_v41/codec_reference.py) 执行经过验证的本地编码器和分词器字节.
它在排他发布前检查输入与源码身份, 不重新计算权重载荷哈希.
禁用自动特殊 token 插入, 填充和截断.
Python 编码器在适用位置提供 BOS.
用例保存输入, 渲染文本, 完整 IDs 和有序媒体记录.
生成器不获取或预处理图片.

用例覆盖聊天, 推理强度别名及数值, 推理移除, 推理保留, 工具, 系统消息, 助手续写和图像.
工具对话按本地规则保留此前推理.
显式提供 context 时, 编码器仅返回续写文本及其中的新图像.
调用方分别保留此前上下文和媒体.
这些夹具不证明 Jinja 等价性.
已标识的 Python 编码器仍是权威来源.

### 工具图像顺序修正

`tool-result-images` 记录一个源码缺陷: 编码器在按调用顺序排序工具结果前收集图片.
反向提供结果会导致占位符与图片错位.
这是诊断行为, 不是获准的服务行为.

`tool-result-images-canonical` 在编码前排序完整结果记录, 保持正确图片顺序.
服务适配器必须在提取媒体前应用此排序.
它必须拒绝缺失, 重复或未知的调用身份.
仅移动图片记录不够.

## 上游检查点边界

权重和参考使用本地检查点.
用于比较的[上游检查点](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)具有匹配的布局, 但全部 48 个分片的整文件哈希不同.
不要用其权重或数值结果进行替换.

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
