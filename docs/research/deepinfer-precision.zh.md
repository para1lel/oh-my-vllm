# DeepInfer 缓存与计算精度调研

调研日期: 2026-11-04. 本文只检查第一方公开源码和模型配置, 范围为 NVIDIA SM100 路径. 本文没有执行上游程序, 构建或模型, 也没有进行 GPU 测量.

## 结论与版本

用户所指的第一方实现是 `deepseek-ai/DeepInfer`. DeepSeek 的[发布问答](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md#L17-L21)明确给出这个仓库, 并说明它用于 V4.1 Flash. [模型配置读取器](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/hugging_face/config.py#L245-L253)实际检查 `deepseek_v41`, 默认检查点是官方的 [DeepSeek-V4.1-Flash-1017](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md#L65-L69). 本文用该检查点的固定配置交叉核对维度和共享层, 没有按项目名称猜测支持情况.

本次检查的 DeepInfer 主分支提交为 `a679bb86ce36e883a98a97733492b686e5b9a08e`. 它的[固定依赖树](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e/vendor)指定 FlashMLA `2e5429fc5653bab6e081f09477126f731882a6a9` 和 DeepGEMM `1fc9c66975ee322d628b3288534ca0e9ce12a64d`. 下文所有源码链接都固定在这些提交. 官方模型配置固定在 `1bae0e1088993b9b82de08f59de5be5a34e42170`.

核心结论是: 默认全局 KV 用 FP4 持久存储, 主 attention 用 BF16 操作数和 FP32 累加. 索引器另用 MXFP4 QK 指令. "全程 FP4" 和 "全程 BF16" 这两种说法都不充分. 证据见[默认配置](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/hugging_face/config.py#L299-L306), [主 attention 指令类型](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/config.h#L251-L264)和[索引器指令类型](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh#L371-L380).

## 持久缓存的格式

下表中的 "每条" 指一个缓存向量, 不是一个原始 token 在全部层中的总量. 量化精度表示解码规则; `uint8` 或 `int8` 表示装载编码的物理容器.

| 缓存 | 数学格式与分组 | 每条净字节 | 物理布局与源码 |
|---|---|---:|---|
| 全局压缩 KV, 默认 | 512 个 E2M1 FP4 值, 每 16 个值配一个 E4M3 浮点 scale | 256 + 32 = 288 | `uint8` 缓存, 每条 `[data \| scale]`; [格式定义](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L119-L128) |
| 索引器的键 | 128 个 E2M1 FP4 值, 每 32 个值配一个 UE8M0 指数 scale | 64 + 4 = 68 | 块内全部条目的数据位于全部 scale 之前; [格式定义](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L129-L137) |
| SWA 窗口 | 512 个 E4M3 FP8 值, 每 32 个值配一个 UE8M0 指数 scale | 512 + 16 = 528 | `uint8` 缓存, 每条 `[data \| scale]`; [格式定义](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L109-L118) |
| 全局 KV, 关闭 FP4 时 | 与 SWA 相同的 FP8 格式 | 528 | 由 `use_fp4_sparse_kv` 选择; [布局选择](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L46-L56) |

FP4 是 E2M1 浮点编码, 两个值装入一字节, 不是四位整数. 全局 KV 的 scale 是一字节 E4M3, 索引键的 scale 是一字节 UE8M0; UE8M0 在部分接口中把四字节打包成一个 `int32`. 它们不能互换. 全局 KV 的全部 512 维都量化, 包括 64 维 RoPE, 没有额外 BF16 尾部. 证据见[物理视图](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L275-L328)和 [FlashMLA 格式常量](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/cuda_kernels/kv_cache_format.h#L5-L19).

全局 FP4 写入 kernel 先计算每组绝对值最大值, 把 `amax / 6` 限幅并舍入成 E4M3 scale, 再用这个已经舍入的 scale 的倒数生成 E2M1 编码. 这个格式不额外保存一个 FP32 总 scale. 索引器采用 UE8M0 规则. 证据见[实际写入代码](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cuda/store_kv_cache.py#L141-L184).

官方配置的 KV 源层是 `2/8/14/20`, 对应压缩率 `2/2/2/1`. 索引计算模块的源层有八个, 但配置读取器在缺少独立的 `index_k_source_layer_ids` 时, 让索引键跟随四个 KV 源层. 因而不能按八个索引模块重复计算八份键缓存. 证据见[官方层配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json#L60-L123)和[三种共享映射](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/hugging_face/config.py#L62-L85).

分配器让消费者直接引用源层的 `ica_cache` 或 `ca_cache`, 不再增加该项偏移. 全局缓存形状为 `[num_pages, block_size / ratio, 1, bytes_per_entry]`. 索引区按 32 字节对齐; 整页还按 `lcm(128, main_bytes_per_entry)` 对齐. SWA 在每个序列的状态槽中单独分配窗口环. 证据见[页面与共享分配](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L406-L484)和 [128 字节约束](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/model_metadata_utils.py#L137-L140).

这些持久缓存没有一份与完整历史等大的 BF16 展开副本. 压缩率大于 1 的源层另有 FP32 压缩状态环, 形状为 `[state_slots, ring_size, 2, dim]`; 它保存压缩与回退状态, 不随完整上下文线性增长. 证据见[缓存字段](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L205-L218), [环长](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L119-L133)和 [FP32 状态视图](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L367-L374).

## 预填充, 解码和投机验证怎样读取

| 路径 | 实际读取和展开位置 | attention 的计算类型 |
|---|---|---|
| 主 attention 预填充 | 从分页 FP4 全局 KV 和 FP8 历史窗口装载到连续 BF16 缓冲; kernel 先量化当前窗口, 再展开 | Q/K/V 为 BF16; QK 与 PV 的 tensor core 累加为 FP32 |
| 主 attention 解码 | 把 FP8 SWA 和 FP4 全局 KV 作为两个缓存直接传给 FlashMLA; kernel 按所选条目在寄存器和共享内存中展开为 BF16 | Q/K/V 为 BF16; QK 与 PV 累加为 FP32 |
| 目标模型投机验证 | decode 元数据容纳 `nextn + 1` 个查询行, 仍走非预填充分支 | 与主 attention 解码相同的缓存格式和指令类型 |
| 索引器预填充 | 从页中提取编码和 scale, 保持 FP4, 不先展开成 BF16 全量键 | MXFP4 QK 指令, FP32 点积累加; 随后的按头加权与归约为 BF16 |
| 索引器解码及验证 | DeepGEMM 从分页编码直接评分, 支持候选块限制 | 同上; 输出 BF16 分数用于选择 |

主路径的调用依据是[两个缓存的解码调用](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/attn.py#L628-L669), [预填充装载与分支](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/attn.py#L812-L877), [当前窗口量化展开](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/attn.py#L402-L449), [装载 kernel 的 BF16 输出](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cuda/load_paged_kv_cache.py#L100-L125)和[投机查询长度](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/input_metadata/prepare_tensor_input_zos.py#L307-L358).

kernel 的类型证据是融合路径的 [BF16 展开](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.cuh#L256-L294)与 [BF16/BF16/FP32 矩阵指令](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/config.h#L251-L278). 未融合的 64 头[预填充](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/cuda_kernels/sm100/prefill/sparse/fwd/head64/config.h#L125-L131)和[解码](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/cuda_kernels/sm100/decode/sparse/head64/config.h#L174-L181)也定义相同操作数与累加类型. `is_fp8_kvcache=True` 这个接口参数名不能证明全局 KV 是 FP8; [格式定义](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/cuda_kernels/kv_cache_format.h#L37-L54)允许 FP8 主缓存加 FP4 额外缓存.

主 attention 的 softmax 中间分数, 最大值与分母为 FP32; 指数结果写成 BF16 后参加 PV. 默认的融合路径在被选用时还把输出量化成 E4M3 加每 32 维 UE8M0 scale, 供后续投影使用. 因此 FP32 累加不表示所有中间值都保留 FP32. 证据见 [softmax 与 BF16 概率](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.cuh#L929-L990), [输出格式](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/config.h#L28-L36)和[融合选择条件](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/attn.py#L85-L92).

索引器 Q 先转 FP4; 预填充装载仍返回编码与 scale, 解码直接传分页缓存. 证据见 [Q 量化与 K 装载](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/index_attn.py#L164-L202), [预填充调用](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/index_attn.py#L284-L325)和[分页调用](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/index_attn.py#L421-L454). DeepGEMM 明确使用 E2M1/UE8M0 的[块缩放指令与 FP32 累加类型](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh#L371-L419), 对应 [MXFP4 每 32 维指令](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/ptx/tcgen05.cuh#L124-L145). kernel 对点积结果应用 ReLU 并转换成 BF16, 然后用 BF16 头权重做乘加和归约; [普通评分](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh#L440-L516)与[候选块评分](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_sparse_mqa_logits.cuh#L568-L625)都有直接证据.

## 对 64 条乘 800000 token 的含义

按上述官方源层和压缩率, 每条序列的全局缓存条目数为 `800000 / 2 * 3 + 800000 = 2000000`. 每条同时需要 `288 + 68 = 356` 字节. 因而净数据为:

| 项目 | 全部 64 条的净字节 | GiB |
|---|---:|---:|
| 全局 KV | 36864000000 | 34.332275 |
| 索引键 | 8704000000 | 8.106232 |
| 两者合计 | 45568000000 | 42.438507 |
| 再加 40 层, 128 槽 SWA | 45741015040 | 42.599640 |
| 再加 43 层, 128 槽 SWA | 45753991168 | 42.611725 |

这里 `1 GiB = 2^30` 字节. 这些值由[官方 40 层与窗口配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json#L27-L60), [三个预测层配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json#L145-L155)和前述缓存格式直接计算. 43 层这一行计入三个草稿层的窗口, 不是新增三层全局历史 KV; [运行层构造](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/hugging_face/config.py#L136-L155)把它们设为窗口 attention.

这些数值只是逻辑净数据, 不是 DeepInfer 的实际显存分配或完整运行容量. 实际 SWA 环还包括回退槽; 状态槽默认预留比例为 `1.2`, 并多分配一个槽; 此外还有页面对齐, FP32 压缩状态, 模型权重和工作区. 证据见[窗口与回退长度](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L88-L110), [状态槽数量](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L201-L218)和[默认环境配置](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L135-L171).

预填充另有较大的临时展开成本. `_prepare_prefill_kv` 分配 `[total_kv, 1, 512]` 的 BF16 缓冲, 同一共享段复用, 下一源层开始前释放上一段. 如果一个没有上下文并行拆分的调用同时装入 64 条各 800000 token 的比例 1 源层历史, 单这一段全局 KV 的展开量就是 `64 * 800000 * 512 * 2 = 48.828125 GiB`, 窗口部分还要占用更多存储. 这是条件计算, 不是实测峰值; 也不应把四个源层的展开量全部相加成同时常驻量. 证据见[分段分配与释放](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/attn.py#L723-L750)和[缓冲长度与装载](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/attn.py#L812-L842). 分块查询本身也不保证历史键展开量受到同样限制, 需要同时约束装载的总键数.

## 其他精度, 仅列相关边界

| 部分 | 可由本次源码确认的精度 |
|---|---|
| 量化线性层 | 激活转 E4M3, 每 32 维用 UE8M0 scale; 量化线性层通常输出 BF16. BF16 权重分支单独存在. [转换参数](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/linear.py#L13-L24), [线性调用](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/linear.py#L140-L178) |
| 路由专家 | 官方检查点声明 FP4 专家权重; runtime 保留编码权重, 输入支持每 32 维 FP8, 调用 FP8/FP4 分组或融合接口. 本文未逐个审计所有专家矩阵指令. [检查点量化配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json#L12-L20), [编码权重与接口选择](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/mlp.py#L207-L255), [激活转换](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/mlp.py#L274-L288) |
| 路由器 | GPU 调用 `bf16_mega_gate`; BF16 输入和权重, FP32 tensor core 累加. [runtime 调用](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/mlp.py#L397-L400), [矩阵指令](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_gate.cuh#L149-L171) |
| HC/mHC | 混合参数和系数为 FP32, 残差与输出可为 BF16, 也可融合输出 FP8. 融合投影使用 TF32 操作数和 FP32 累加, 不能因参数为 FP32 就称其为完整 FP32 乘法. [参数类型](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/hc_blocks.py#L38-L41), [输出缓冲](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/hc.py#L252-L298), [TF32 指令](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh#L233-L238) |

## 可借鉴点与未证实事项

当前实现可借鉴三个边界: 让持久 KV 保持编码状态并只由四个源层拥有; 把编码格式, scale 格式和页面偏移作为明确契约; 让 decode 在读取少量条目时展开, 同时对 prefill 的 BF16 展开量单独设预算. 这些思路使用前述共享分配, 格式定义和两条读取路径. 这些是待移植和验证的设计方向, 本文没有证明它们与本仓库当前计算次序逐位等价, 也没有实现产品代码.

公开版本有可检查的物理缓存与读取 kernel, 但第一方明确说明完整生产服务的 KV 缓存系统和负载均衡没有完全开源. 因而不能从本文推导生产系统的全部复制策略, 多级缓存占用或每卡可承载序列数. 证据见[公开边界](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md#L19-L21)和[缺少的服务组件](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md#L38-L42). 本文没有审计 Ascend kernel, 没有验证构建产物选择的 kernel, 也没有用数值或容量实测证明任何部署配置.
