# DeepSeek V4.1 源码契约

## 权威来源和身份

这是 Phase 0 数学源码契约, 版本为 1.
它不表示运行时支持或数值验收通过.
[状态契约](state.zh.md) 规定稀疏选择, Engram, DSpark 和恢复行为.
[验证计划](validation.zh.md) 负责数值参考和验收限值.

`LOCAL-V41` 标识所提供的 `DeepSeek-V4.1-Flash` 检查点, 而不是主机目录或上游修订.
`LOCAL-V41:` 后面的名称是相对源码身份.
以下 SHA256 值绑定数学源码.
它们不验证权重载荷.

| 源码身份 | SHA256 |
|---|---|
| `LOCAL-V41:config.json` | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| `LOCAL-V41:model.safetensors.index.json` | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |
| `LOCAL-V41:inference/model.py` | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| `LOCAL-V41:inference/kernel.py` | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |
| `LOCAL-V41:inference/convert.py` | `035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76` |
| `LOCAL-V41:inference/engram.py` | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| `LOCAL-V41:inference/vision.py` | `5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c` |
| `LOCAL-V41:inference/image_processor.py` | `482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272` |
| `LOCAL-V41:encoding/encoding.py` | `502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1` |
| `LOCAL-V41:tokenizer.json` | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |
| `LOCAL-V41:tokenizer_config.json` | `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547` |
| `LOCAL-V41:chat_template.jinja` | `d959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3` |
| `LOCAL-V41:README.md` | `347c9db4e5506acb531cbc3b724407ab88e9af8781679152f0823d7bac16d251` |
| `LOCAL-V41:LICENSE` | `f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040` |

本地许可证为 MIT.
[DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9) 是只读算法参考, 固定修订为 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`.
其[许可证](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/LICENSE) 为 Apache-2.0.
其[默认检查点](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/_config.py) 为 `deepseek-ai/DeepSeek-V4.1-Flash-1017`.
这个名称不能证明它与本地检查点的权重身份相同.

## 目标模型计算

配置选择 `DeepseekV41ForCausalLM`, 宽度为 5120, 目标层数为 40, 词表大小为 129280.
总上下文限制为 1048576 个 token.
注意力有 64 个查询头, 一个共享的 512 通道 KV 向量, 以及 64 个旋转通道.
查询秩为 1280.
输出投影有八个组, 每组秩为 1024.
嵌入与词表头使用不同权重.

`LOCAL-V41:inference/model.py` 中的 `Transformer.forward`, `Block` 和 `Attention` 定义本地计算图.
四个超连接副本承载每个残差.
初始合并选择副本 0.
注意力使用前一个 FFN 的 pre-mix.
FFN 使用当前注意力的 pre-mix.

每个子层为自己的输出生成 post-mix 和残差混合系数.
Pre-mix 是 sigmoid 加 `1e-6`.
Post-mix 是 sigmoid 的两倍.
残差混合使用 20 次 Sinkhorn 迭代, 稳定项为 `1e-6`.
这一有限次计算不是精确的双随机投影.
文本 RMS 归一化使用 FP32 统计量, epsilon 为 `1e-20`.

每个目标层有 384 个路由专家, 选中六个专家, 并有一个共享专家.
专家中间宽度为 2304.
路由分数为 `sqrt(softplus(xW))`, 门控温度为 1.

文本或图像校正偏置选择专家, 但不缩放专家输出.
选中的无偏分数用分母 epsilon `1e-20` 归一化, 然后乘以 1.5.
SwiGLU 将门控投影的上限设为 10, 并将 up 投影截断到 `[-10,10]`.
路由权重在 down 投影之前乘到激活后的中间值上.

## CED 边界

`LOCAL-V41:README.md` 将层 0-19 称为因果编码器, 将层 20-39 称为解码器.
压缩率为 1 的全局 KV 源是第 20 层归一化后的注意力输入.
后续解码器层读取这一全局 KV, 而不是生成各自的全局 KV.
Decode 执行全部 40 个目标层.

本地 `Transformer.forward` 也对完整 prefill 执行全部 40 层.
它没有实现仅编码器 prefill 或有界解码器重放.
其解码器滑动窗口 KV 来自每层自身的注意力输入.
其目标隐藏特征也依赖这些解码器计算.

DeepInfer 的 [CED 辅助模块](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/yoco.py) 默认使用共享的解码器 SWA 输入.
其[部署说明](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md) 允许截断的近似重放.
这两种行为都不等同于本地数学计算.
没有获批的等价契约时, 不得省略解码器 prefill 行.

## 权重和缩放格式

`Linear`, `ParallelEngramEmbedding` 和 `LOCAL-V41:inference/convert.py` 定义存储解释方式.
本地索引中的张量名称已使用 `layers`, `attn`, `ffn`, `mtp` 和 `scale`.
别名转换不得改变张量身份.

FP8 矩阵权重使用 E4M3, 每个 32-by-32 输出/输入块有一个 UE8M0 乘数.
重建公式为 `W[o,k]=E4M3[o,k]*S[o//32,k//32]`.
当 `0<=b<=254` 时, UE8M0 字节 `b` 表示 `2^(b-127)`.
字节 255 是 NaN, 不是有限缩放值或零点.
缩放名称不表示要乘以倒数.

路由专家权重用有符号字节存储两个 E2M1 值.
逻辑形状 `[O,K]` 的存储形状为 `[O,K/2]`.
提取半字节之前, 按无符号值读取字节.
低半字节提供通道 `2j`.

高半字节提供通道 `2j+1`.
编码 0-7 表示 `0,0.5,1,1.5,2,3,4,6`.
编码 8-15 提供对应负值, 编码 8 为有符号零.
每个输出行的每 32 个输入通道有一个 UE8M0 乘数.
这不是 32-by-32 专家缩放布局.

Engram FP8 表的每行每 32 个通道有一个 UE8M0 缩放值.
其存储形状为 `[384006168,256]` 和 `[384016682,256]`.
每个缩放表有八列.
权威表和缩放值保存在页锁定主机存储中.

本地转换器将分组 `wo_a` FP8 权重展开为 BF16.
模型将压缩率为 2 的压缩器 BF16 投影加载到 FP32 参数中.
词表头和置信度投影也从 BF16 权重得到 FP32 参数.
这些是源码的计算选择, 不是原始存储格式.
不得静默替换为专家 FP8 转换或不同的投影舍入边界.

## 中间精度

`LOCAL-V41:inference/kernel.py` 定义量化与累加.
稠密投影和专家投影按 32 个一组将激活量化为 E4M3.
其缩放值为 `2^ceil(log2(max(amax,1e-4)/448))`.
转换之前, 量化值截断到 `[-448,448]`.
本地默认配置中的 GEMM 使用 FP32 累加器和 BF16 输出.

滑动窗口 KV 在 RoPE 之后对全部 512 个通道使用这一 FP8 量化/反量化操作.
索引器查询和键使用 32 个一组的 FP4, 缩放格式为 UE8M0.
其缩放值将 `max(amax,6*2^-126)/6` 向上舍入到二的幂.
压缩主 KV 使用 16 个一组的 FP4, 缩放格式为 E4M3.

其缩放值为 `E4M3(max(amax,6*2^-9)/6)`, 而不是 UE8M0.
FP4 转换截断到 `[-6,6]`.
参考实现将反量化后的缓存值存储在 BF16 张量中.
打包的服务缓存必须复现这些量化边界.

RoPE 旋转最后 64 个通道中的相邻元素对.
注意力输出在分组输出投影之前接受逆查询旋转.
仅窗口层使用 theta 10000, 不使用 YaRN.
压缩层使用 theta 160000, 原始长度为 65536, YaRN 因子为 16.
YaRN 使用 beta 值 32 和 1.
本地注意力分数缩放为 `1/sqrt(512)`, 没有额外的 YaRN 幅度乘数.

## 提示词编码

`LOCAL-V41:encoding/encoding.py` 负责提示词文本, 工具顺序, 推理标记和有序图像记录.
它在渲染后的提示词中提供 BOS.
它使用 V4.1 DSML 标签, 在 `calls`, `invoke` 和 `parameter` 前有一个前导空格.
工具结果遵循其调用顺序.
工具对话保留此前推理, 不应用通常的推理删除规则.

数值推理力度为 1-100.
别名 `low`, `high` 和 `max` 映射到 50, 75 和 100.
默认值为 `high`, 即 75.
力度前缀仅在思考模式下的对话索引 0 处出现.
这些别名不定义其他公共 API 力度名称的映射.
冻结编码后的 token ID 和图像顺序, 不得仅依靠渲染文本相似性.

## 图像和视频边界

`LOCAL-V41:inference/image_processor.py` 负责缩放, 填充, 归一化, patch 顺序和图像片段展开.
RGB 像素使用 `(pixel/255-0.5)/0.5`, 然后转为 BF16.
最小像素面积为 295936.

最终图像片段最多有 1024 个 token.
规划器按 `H*(W+1)+2` 计数, 每个对齐后的行有一个换行 token.
Patch 大小为 14.
参考实现使用 PIL 的填充和缩放行为, 而不是通用中心裁剪.

`LOCAL-V41:inference/vision.py` 定义 32 个双向层, 宽度为 1024, 头数为 16, 中间宽度为 2816.
视觉使用 2D RoPE, RMS epsilon 为 `1e-6`.
对齐器将 patch 网格补零到 3 的倍数.
它收集不重叠的 3-by-3 单元, 然后用两层 GELU 投影器映射到宽度 5120.

图像特征在超连接展开之前替换文本嵌入.
开始, 结束和换行位置使用学习得到的嵌入.
所有图像片段位置都携带 token ID 129264, 并使用单独的 token 类型.
语言模型位置是顺序 token 位置, 不是 Qwen 三轴位置.
本地 forward 仅允许在第一个 prefill 分块中接收完整图像片段.

本地编码源码支持有序图像, 不提供视频帧采样器或时间戳协议.
视频必须先通过单独规定的编解码器生成有序帧图像和时间戳文本.
不存在时序视觉编码器.
帧选择, 时间戳写法, token 位置和数值参考输出是尚未解决的视频前置条件.

## 兼容性关卡

DeepInfer 的[分词器策略](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/hugging_face/tokenizer.py) 添加多模态 ID 129280-129300, 并改变 BOS/PAD 行为.
本地分词器禁用自动 BOS/EOS 插入, 并将分词器 PAD 设为 EOS 别名.
模型配置另行选择 PAD 2.
不得在生成 Engram 映射之前应用这些上游变更.

本地 README 声称没有 Jinja 模板, 但 `LOCAL-V41:chat_template.jinja` 存在.
在 token-ID 等价性审查之前, 使用已标识的 Python 编码源码作为权威来源.
权重载荷身份, 独立端点向量和获批限值是单独的 Phase 0 关卡.
本次源码阅读不提供这些数值产物.
