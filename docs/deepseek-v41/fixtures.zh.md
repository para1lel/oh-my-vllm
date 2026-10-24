# DeepSeek 固定参考夹具

## 范围和身份

[夹具读取器](../../development/model/deepseek_v41/fixtures.py) 检查 SHA256 身份, 元数据和完整数组.
链接的索引包含固定数值限值和源码身份.
普通检查无需 PyTorch, GPU 执行或访问检查点即可加载这些产物.
不要在验收期间重新生成期望值, 也不要用候选实现输出替换它们.

算子夹具覆盖独立边界, 不证明模型, 状态, 容量或性能通过验收.
其工程限值不是最坏情况误差证明.
这些独立门槛见[验证计划](validation.zh.md).

| 夹具 | 内容 |
|---|---|
| [短文本](../../development/model/deepseek_v41/reference.npz) | 两个完整 FP64 词表向量 |
| [长文本](../../development/model/deepseek_v41/long-context.npz) | 16385 和 65537 tokens 处的四个完整 FP64 向量 |
| [提示词编码](../../development/model/deepseek_v41/codec.json) | 二十个包含完整 token IDs 和媒体顺序的用例 |
| [DSpark 续算](../../development/model/deepseek_v41/dspark.npz) | 十八个词表行和两个置信度向量 |
| [媒体端点](../../development/model/deepseek_v41/media.npz) | 静态图片和帧图片之后的 FP64 prefill 与续算 |
| [精确算子](../../development/model/deepseek_v41/exact-operators.npz) | 量化字节, Engram 哈希和续算历史 |
| [Engram lookup](../../development/model/deepseek_v41/engram-lookup.npz) | 合成编码表和精确 BF16 输出 |
| [Engram 检查点](../../development/model/deepseek_v41/engram-checkpoint.npz) | 两个表的原始样本和精确输出 |
| [Engram 融合](../../development/model/deepseek_v41/engram-blend/index.json) | 四个包含完整 FP64 门控和残差的用例 |
| [Hyper-connections](../../development/model/deepseek_v41/hyperconnection/index.json) | 五个包含完整输出和字段限值的用例 |
| [路由](../../development/model/deepseek_v41/routing/index.json) | 五个包含精确专家 ID 和 FP64 系数的用例 |
| [编码投影](../../development/model/deepseek_v41/projection/index.json) | 五个 A8W8 和五个 A8W4 矩阵乘积 |
| [Rank 自有投影](../../development/model/deepseek_v41/rank-encoded-projection/index.json) | 1, 128 和 384 行的精确量化及完整 FP64 乘积 |
| [未量化投影](../../development/model/deepseek_v41/unquantized-projection/index.json) | 七个包含完整 FP64 乘积的 BF16/FP32 用例 |
| [启动转换](../../development/model/deepseek_v41/startup-promotion.npz) | 精确的 FP8 到 BF16 及 BF16 到 F32 转换 |
| [归一化和旋转](../../development/model/deepseek_v41/norm-rope/index.json) | 四个 RMS 和三个 RoPE 用例 |
| [激活](../../development/model/deepseek_v41/swiglu/index.json) | 四个 SwiGLU 用例 |
| [Ratio2 池化](../../development/model/deepseek_v41/pooling/index.json) | 四个完整成对用例 |
| [稀疏注意力](../../development/model/deepseek_v41/sparse-attention/index.json) | 五个包含完整 FP64 输出的打包缓存用例 |

读取器绑定本地检查点清单. 完整模型产物还包含固定输入, 生成器身份, 形状和数组摘要.
浮点比较使用独立的请求限值. 整数决策和编码字段在规定位置使用精确限值.

## 精确算子

`load_exact_operators()` 覆盖 FP8/UE8M0 group32, FP4/UE8M0 group32, FP4/E4M3 group16, 以及含 dead tokens 的 Engram 分块续算.
期望值来自手工检查的字面量.
比较编码值, scale, BF16 往返位模式, 哈希和历史.

`load_engram_lookup()` 在合成 16 行表中覆盖重复索引, 有符号零, subnormal 中点和不同 scale 组.
`load_engram_checkpoint()` 提供每个表的 32 个原始行及精确 BF16 输出.
[Lookup 检查器](../../development/operators/check_engram_lookup.py) 使用这些样本验证完整共享存储.

量化调用者必须拒绝非零分组状态. Lookup 调用者必须拒绝非零行状态.
Engram 哈希返回私有续算历史, 不修改输入历史.

将 `EXACT_REPORT` 设为新的外部 JSON 路径. 构建并检查模块:

```bash
scripts/with-env.sh python -m development.native_build \
  --module device --module quantization --module engram
OH_MY_VLLM_GPU_COUNT=1 scripts/with-env.sh scripts/with-gpu.sh \
  python -m development.operators.check_deepseek_exact --output "$EXACT_REPORT"
```

## Engram 融合

`load_engram_blend()` 覆盖 1, 128 和 384 行, 以及合成决策用例.
输入包含 BF16 残差, 四个投影 key, 一个共享 value, 两份归一化权重和 token mask.
[边界](../../python/oh_my_vllm/operators/engram/blend.py) 对每份残差分别归一化, 然后应用带符号平方根门控.
[检查器](../../development/operators/check_engram_blend.py) 比较全部门控和残差, 包括极小带符号点积和被屏蔽的行.
Lookup 和投影是独立边界.

## Hyper-connections

`load_hyperconnection()` 使用原始 F32 权重, 覆盖 decode, prefill, verification, 零值和抵消用例.
[Workspace](../../python/oh_my_vllm/operators/hyperconnection/workspace.py) 将子层前的准备与子层后的展开分开.
折叠使用传入系数. `next_pre` 属于下一个子层.
[检查器](../../development/operators/check_deepseek_hc.py) 还置换通道, 以检测寻址错误.
`Workspace.collapse()` 为最终 head 提供输入, 无需计算新的混合系数.

## 路由

`load_routing()` 从已存储的 FP32 logits 和原始文本/图像偏置开始.
用例覆盖 384 个专家的 top-6 和 128 个专家的 top-3.
[模型契约](model.zh.md) 规定不含偏置的系数和稳定的全局专家 ID.
专家 ID 必须精确匹配. 非并列选择必须通过索引中的分数区间检查.
合成并列值使用相同的 logits 和偏置.

## 编码投影

`load_projection()` 提供编码激活, UE8M0 scales 和完整 FP64 乘积.
原始权重保存在外部. 合成 FP4 用例包含自身权重.
[检查器](../../development/operators/check_deepseek_projection.py) 覆盖打包, K=576 和最大 15360 的宽度.
激活量化和 collective 归约是独立边界.

`load_rank_encoded_projection()` 使用原始 rank 自有 `wq_a` 权重检查[量化与投影](../../python/oh_my_vllm/models/deepseek_v41/linear.py).
其 FP64 乘积使用量化后的激活.
在状态提交前拒绝非零量化状态.

`load_startup_promotion()` 包含全部有限 FP8/UE8M0 组合和全部有限 BF16 位模式.
它还包含溢出输入, 非法 BF16 位模式, 以及原始和合成 dense32 scales.
转换使用 ties-to-even 舍入, 保留有符号零, 并拒绝非有限输出.
外部生成元数据的 SHA256 为 `b341dd5dcf26691c104757b29b7cd6d1ea1eecf0f5429f3f7638e5f29d7e0dcb`.

## 未量化投影

`load_unquantized_projection()` 提供七个包含完整 FP64 矩阵乘积的 BF16/FP32 用例.
[Workspace](../../python/oh_my_vllm/operators/linear/workspace.py) 支持 BF16 到 BF16, BF16 到 FP32, 以及 FP32 到 FP32 乘积.
分组 `wo_a` 使用完整输入和输出行步长.
路由 logits 保持 FP32. 累加使用 FP32, 不使用 TF32 或中间 BF16 转换.
[检查器](../../development/operators/check_deepseek_linear.py) 覆盖分组, 完整词表输出, 抵消和精确舍入决策.

## 归一化, 旋转和激活

`load_norm_rope()` 使用无偏移的原始 BF16 RMS 权重.
文本 epsilon 为 `1e-20`. 视觉 epsilon 为 `1e-6`.
RoPE 使用传入的 FP32 余弦/正弦系数处理最后 64 个通道中的相邻通道对. 前缀位保持不变.
相位生成和二维视觉位置是独立边界.

`load_swiglu()` 提供 BF16 gate/up 输入和完整 FP64 输出.
[API](../../python/oh_my_vllm/operators/activation/api.py) 规定文本截断范围和可选 FP32 路由权重.
[Plan](../../python/oh_my_vllm/operators/activation/plan.py) 规定视觉宽度.
[检查器](../../development/operators/check_deepseek_activation.py) 对每个请求分别应用固定限值.

## Ratio2 池化

`load_pooling()` 提供宽度为 512 的完整 FP32 KV 和 gate 对.
Softmax 沿每对的两个元素计算, 各通道独立. 加权和在 RMS 之前仅转换一次为 BF16.
固定限值包含舍入和抵消余量.
[检查器](../../development/operators/check_deepseek_pooling.py) 不覆盖不完整组状态, 投影, RMS 和 RoPE.

## 打包稀疏注意力

`load_sparse_attention()` 提供 BF16 查询, 原始 F32 sinks, 编码缓存行和显式槽位 ID.
[API](../../python/oh_my_vllm/operators/sparse_attention/api.py) 规定格式, 掩码, 重复 IDs 和错误状态.
仅解码选中的缓存块, 不展开整个逻辑缓存.
提交状态前拒绝非零状态码.
[检查器](../../development/operators/check_deepseek_sparse_attention.py) 比较全部输出, 精确零值保护和非法 ID 行为.

## 短文本

[参考指南](reference.zh.md) 规定方程和固定的 16-token 提示词.
产物包含全部 129280 个 prefill logits 和全部 129280 个续算 logits.
prefill argmax 选择 token 16, 从相同状态续算.
权重解码和模型算术使用 FP64, 不进行激活量化.

## DSpark 续算

`load_dspark()` 提供十八个完整 FP64 词表行和两个各含五个值的置信度向量.
129-token 前缀将短提示词重复八次, 再添加其第一个 token.
数组覆盖 prefill, 五个提议, 六个验证输入, 恢复后续算和后续五个提议.
该用例选择两个验证输入, 相当于接受一个草稿, 并独立重算此前缀, 不使用被拒绝状态.

后续提议仅使用已接受特征. 此强制选择不是采样拒绝事件.
参见[构建流程](reference.zh.md#dspark-参考).

## 媒体端点

`load_media()` 提供一张合成静态图片和两张已解码视频帧的完整 FP64 prefill 与续算 logits.
37-token 占位符提示词展开为 624 tokens. Token 9544 从已接受状态续算.
元数据绑定提示词, 有序图片, patches, 视觉嵌入和 token 映射.
参见[参考流程](reference.zh.md#媒体参考).

## 提示词编码

[编码生成器](../../development/model/deepseek_v41/codec_reference.py) 执行经过验证的本地编码器和分词器字节.
它禁用自动特殊 token 插入, 填充和截断. Python 编码器在适用位置提供 BOS.
用例保留渲染文本, 完整 IDs 和有序媒体记录. 生成器不获取或预处理图片.

用例覆盖聊天, 推理强度, 推理移除和保留, 工具, 系统消息, 助手续写和图像.
显式提供 context 时, 编码器仅返回新增文本和图像. 调用方分别保留此前上下文和媒体.
已标识的 Python 编码器是权威来源. 这些夹具不证明 Jinja 等价性.

### 工具图像顺序修正

`tool-result-images` 记录一个源码缺陷: 编码器在按调用顺序排序工具结果前收集图片.
反向提供结果会导致占位符与图片错位. 这是诊断行为.

`tool-result-images-canonical` 在编码前排序完整结果记录.
服务适配器必须在提取媒体前应用此顺序, 并拒绝缺失, 重复或未知的调用身份.
仅移动图片记录不够.

## 上游检查点边界

权重和参考使用本地检查点.
[用于比较的上游检查点](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) 具有匹配的布局, 但全部 48 个分片的整文件哈希不同.
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
