# 工作状态

Rust 提供 HTTP API, CLI, 调度器, 逻辑 KV cache 和 worker 生命周期控制.
服务模式为普通解码和 MTP4.
HTTP 输入可包含文本, 图片和视频.

Python worker 加载 checkpoint, 执行语言模型, MTP 和 vision 模型路径.
`stub` 模型选择用于服务测试的占位实现.
Rust 通过推理 worker 的进程和协议边界发送媒体引用.

[算子契约](operators/model.zh.md)包含 18 个内部阶段及 7 个 serving 边界.
负载包含六类请求模板, 每类使用一个 decode context.

仓库内的 CUDA C++ provider 使用固定版本的 CUTLASS 头文件, 通过[算子开发工具](operators/development.zh.md)提供七个边界.
构建工具与推理加载器分开.
每个 provider 提供数值检查, 并在测量前恢复可变状态.
报告绑定 checkpoint shards, 源文件, 二进制文件及构建依赖.
Vision MLP 激活使用投影输出存储.

Vision 使用一个 [kernel](../python/oh_my_vllm/kernels/cuda_backend/vision_residual_norm.cuh#L3)<!-- source: __global__ void vision_residual_norm_kernel( --> 完成第一次残差更新和第二次归一化.
它保留原有 BF16 舍入点, 中心化方差及归约树.
舍入后的残差保留在寄存器中供归一化使用, 并保留在全局存储中供后续残差更新使用.

Vision [MLP 下投影](../python/oh_my_vllm/kernels/cuda_backend/vision_classic.cuh#L255)<!-- source: inline void vision_classic_down( --> 在行数大于 32 时使用经典 cuBLAS.
BF16 输入生成 FP32 投影存储, 原有 BF16 偏置与残差舍入不变.
自有 handle 和 cuBLAS 缓冲区在捕获前准备.
构建与加载器绑定供应商头文件, 映像和链接标志, 并核对实际加载的映像.

每个 context 在捕获前准备一份 [GELU 查找表](../python/oh_my_vllm/kernels/cuda_backend/vision_lookup_init.cuh#L3)<!-- source: __global__ void vision_lookup_initialize_kernel(cutlass::bfloat16_t *table) { -->.
表中每个 BF16 输入位模式对应一个 BF16 输出.
各项使用原有 GPU functor 和 BF16 输出转换.

一个[原生引用](../python/oh_my_vllm/kernels/cuda_backend/vision_classic.cuh#L95)<!-- source: class VisionClassicGraphLeaseObj : public tvm::ffi::Object { -->持有此 context, 直到图清理结束.
Context 在释放其查找表和 cuBLAS 缓冲区之前等待 GPU 工作结束.

Vision [Q/K/V 打包](../python/oh_my_vllm/kernels/cuda_backend/vision.cu#L49)<!-- source: __shared__ float cosines[72], sines[72]; --> 为每个 token 行计算一次位置系数.
一个 block 在 attention head 之间共享经 BF16 舍入的 sine 和 cosine 值.
Bias 舍入, 配对维度, 补齐零值及帧顺序保持不变.

Vision [attention](../python/oh_my_vllm/kernels/cuda_backend/vision_attention.cuh#L18)<!-- source: static_assert(Dimension == 80); --> 对具有 72 个维度的 head 计算 80 个维度.
Q/K/V 存储保留每行 128 个元素的 stride.
[打包 epilogue](../python/oh_my_vllm/kernels/cuda_backend/vision_attention.cuh#L30)<!-- source: using Epilogue = cute::conditional_t<Packed, VisionPackedEpilogue<BaseEpilogue>, --> 将每个 head 的 72 维直接写入输出 tensor.
分组 TMA 拷贝使用固定 CUTLASS collective 的共享内存布局及原始 softmax scale.

采样归一化每块使用 640 个线程.
MTP 输入行数不超过 32 时, 输入归一化每块使用 640 个线程.
对于这些输入, MTP 使用分为四个 K 分区的 [BF16 投影](../python/oh_my_vllm/kernels/cuda_backend/mtp_projection.cuh#L117)<!-- source: inline void mtp_projection_launch(TensorView input, TensorView weight, TensorView partials, -->.
它在存储 BF16 输出前累加 FP32 部分结果.

MTP [端点复制](../python/oh_my_vllm/kernels/cuda_backend/mtp.cu#L151)<!-- source: for (int vector = threadIdx.x; vector < 640; vector += blockDim.x) { --> 使用 16 字节加载和存储.
原有张量对齐契约继续适用.
残差归一化使用 [16 字节输入加载](../python/oh_my_vllm/kernels/cuda_backend/normalization.cuh#L30)<!-- source: uint4 input = *reinterpret_cast<const uint4 *>(x + row * 5120 + column); -->.

FFN 下投影, FA 输出和 GDN 输出对行数不超过 8 的输入使用共用的 [projection 实现](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L194)<!-- source: inline void decode_projection_launch(TensorView activation, TensorView weight, -->.
单行输入使用[固定行数](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L199)<!-- source: decode_projection_rows_launch<1>(activation, weight, activation_scale, weight_scale, partials, -->, 单列 TMEM 加载及固定数组.
4 或 5 行输入使用[固定 kernel](../python/oh_my_vllm/kernels/cuda_backend/projection_decode_16dp.cuh#L195)<!-- source: inline void launch(TensorView activation, TensorView weight, TensorView activation_scale, -->.
其 TMEM 加载对每个 warp 使用 16 条 data path.
成对 FP32 指令保留原有 K-group 顺序.
其他输入使用运行时行数.

原生算子使用普通 CUDA stream 启动, 不使用 PDL.
当输入有 1, 4 或 5 行时, FFN 使用一次 [CUDA 启动](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L176)<!-- source: inline void paired_up_launch(TensorView input, TensorView input_scale, TensorView weight, --> 完成上投影, 激活和量化.

FA 和 GDN 对 33 到 128 行的输入使用[原生投影流水线](../python/oh_my_vllm/kernels/cuda_backend/projection_prefill.cuh#L31)<!-- source: __global__ __launch_bounds__(640, 1) void projection_kernel( -->.
更多行的回退输出投影也使用此流水线.
FFN 对 65 到 128 行的输入使用相同流水线.

它保留原有 K-group 顺序, FP32 缩放因子及 BF16 舍入点.
每个 block 在重复矩阵乘法前加载全部缩放因子. 独立槽位保护输入存储和 TMEM 读取者.

对于 65 到 128 行, FFN [上投影](../python/oh_my_vllm/kernels/cuda_backend/ffn_prefill_fused.cuh#L37)<!-- source: __global__ __launch_bounds__(640, 1) void upper_kernel<false>( --> 对共享内存中的 BF16 结果执行 SiLU 和 FP8 量化.
中间 gate 和 up 值保留在共享内存中.

更多行的 FFN 输入使用[自有 MXFP8 context](../python/oh_my_vllm/kernels/cuda_backend/mx_fp8_plan.cuh#L203)<!-- source: class MXContextObj : public tvm::ffi::Object { -->.
Context 将预处理后的权重和 scratch 存储保留到 CUDA Graph 清理完成.
每个 FFN owner 为全部 65536 种 BF16 gate 位模式准备一张 FP32 SiLU 表.
此表保留原有乘法顺序及 BF16 乘积存储.
[精度契约](operators/model.zh.md#中间精度)规定其操作数类型和具名容差.

超过 128 行的完整 FA 输入使用[自有 MXFP8 context](../python/oh_my_vllm/kernels/cuda_backend/fa_mx_plan.cuh#L44)<!-- source: class MXContextObj : public tvm::ffi::Object { -->.
它提供输入投影, 并在 attention 只有一个分区时提供输出投影.
Attention 有其他分区数时, 保留原有输出投影.

超过 128 行的 GDN 输入使用[独立的自有 MXFP8 context](../python/oh_my_vllm/kernels/cuda_backend/gdn_mx_plan.cuh#L44)<!-- source: class MXContextObj : public tvm::ffi::Object { -->.
它将每层预处理后的权重及所有调用张量保留到 graph 清理完成.
对于 prefill 及 `max_queries > 1` 的验证, 它还提供输出投影.
单查询验证保留先前的输出投影.

验证输入只有 1 行时, GDN 用一次 [CUDA launch](../python/oh_my_vllm/kernels/cuda_backend/gdn_input_pipeline.cuh#L160)<!-- source: if constexpr (FuseConvolution) { --> 计算输入投影和卷积.
当 `max_queries > 1` 时, 验证使用[独立的递推实现](../python/oh_my_vllm/kernels/cuda_backend/gdn_recurrent_core.cuh#L6)<!-- source: namespace gdn_recurrent { -->.
它直接读取交错存放的 Q/K/V 和 gate 投影结果.
它在 GPU 上检查 metadata 和私有状态槽位.

当最长序列超过 128 个 token 时, GDN [选择分块 prefill](../python/oh_my_vllm/kernels/gdn.py#L19)<!-- source: def chunked_prefill(rows, batch, max_queries, verify): -->.
可以使用不同的序列长度. 它在 784-token checkpoint 和序列末尾处划分 64-token block.
[三角求解](../python/oh_my_vllm/kernels/cuda_backend/gdn_wy_bf16_inverse.cuh#L68)<!-- source: __device__ inline void inverse_blocks(Storage &s) { -->使用 FP32 对角块求解及 BF16 分块乘积, 并使用 FP32 累加.
矩阵乘法使用 BF16 操作数和 FP32 累加.

[Carry kernel](../python/oh_my_vllm/kernels/cuda_backend/gdn_wy_register.cuh#L87)<!-- source: __global__ __launch_bounds__(Threads, 2) void register_carry_kernel( --> 将状态保存在 FP32 寄存器中.
它直接将寄存器片段转换为 BF16 矩阵操作数.
BF16 私有 checkpoint 存储不会改变 FP32 寄存器状态.
[模型契约](operators/model.zh.md#中间精度)给出新值和状态更新各自的舍入点.

输入超过 32 行且每个序列最多 128 个 token 时, prefill 的逐 token 扫描使用 [16-token 预取槽位](../python/oh_my_vllm/kernels/cuda_backend/gdn_kernels.cuh#L145)<!-- source: constexpr int GDNPrefetchRows = 16; -->.
这些槽位将缩放后的 Q 值保留为 FP32, K 值保留为 BF16.
独立 kernel 执行输出归一化和量化.

FA 输入投影为[每个 block 使用 64 个输出列](../python/oh_my_vllm/kernels/cuda_backend/fa_small_input.cuh#L225)<!-- source: fa_small_input_launch(TensorView input, TensorView input_scale, TensorView weight, -->.
full 和 query-only 路径对 1, 4 或 5 个输入行使用此 kernel.
FP32 缩放更新保留 K-group 顺序及原有 BF16 输出转换.

当 attention scratch 缓冲区空间足够时, [KV-only 路径](../python/oh_my_vllm/kernels/cuda_backend/fa.cu#L151)<!-- source: bool split_kv = variant == 1 && (rows == 4 || rows == 16) && -->对 4 或 16 行使用八个 K 分区.
Head kernel 在 BF16 转换, K 归一化, RoPE 和私有 KV 写入前合并 FP32 partials.
这会改变 FP32 结合顺序. 使用命名容差.
其他 KV-only 输入使用 groupwise 实现.

FA prefill 对拷贝, 矩阵乘法, softmax 和输出修正使用[独立角色](../python/oh_my_vllm/kernels/cuda_backend/fa_prefill_pipeline.cuh#L40)<!-- source: __global__ void fa_prefill_kernel( -->.
较大的输入使用[两个 SM 组成的 cluster](../python/oh_my_vllm/kernels/cuda_backend/fa_prefill_cluster.cuh#L84)<!-- source: template <bool ExportBFloat16, bool Decouple> -->.
选择 `Decouple` 时, 独立的 score 和 probability 缓冲区让拷贝, score 矩阵乘法和输出矩阵乘法同时执行.

对于 query 数为 512 到 1023 的单个请求, FA 使用 [ring 调度](../python/oh_my_vllm/kernels/cuda_backend/fa_core.cuh#L56)<!-- source: if (max_queries >= 512 && max_queries < 1024 && batch == 1) { -->.
Worker 从 CPU metadata 提供最大上下文长度.
FA 缓冲区缓存键为 12544-token 调度阈值设置独立的值.

Head kernel 读取 packed projection 存储后, 单分片 prefill 将 BF16 注意力写入该存储.
后续 kernel 直接执行门控和 FP8 转换, 不读取 FP32 partials.

[分组 TMA 描述符](../python/oh_my_vllm/kernels/cuda_backend/fa_prefill.cuh#L60)<!-- source: uint32_t key_box[5] = {64, 64, 4, 1, 1}; --> 提供完整且对齐的 KV tile.
每个描述符 box 都位于一个 784-token 页内.
额外的 16-token TMA 拷贝提供跨页边界的完整对齐 tile.
向量拷贝处理未对齐的起点和最后一个部分 tile.

[Softmax 输出转换](../python/oh_my_vllm/kernels/cuda_backend/fa_prefill_pipeline.cuh#L337)<!-- source: asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(word) : "f"(p0), "f"(p1)); -->用一条 RN 指令将两个 FP32 概率转换为 BF16.
每次向量存储写入八个 BF16 概率. FP32 求和保留原有顺序.

FA 的 [decode 加载](../python/oh_my_vllm/kernels/cuda_backend/fa_attention.cuh#L82)<!-- source: Position base_offset = base % 784; -->在向量循环之前计算 cache page 指针.
每个请求有 1 个 query 且输入行数不超过 32 时, FA 使用[转置的 warp 矩阵乘法](../python/oh_my_vllm/kernels/cuda_backend/fa_q1_warp_transpose.cuh#L17)<!-- source: __global__ void fa_q1_warp_partial_kernel( -->.
请求数不小于 4 且最大上下文长度不小于 32768 tokens 时, 可以使用 128 个 KV 分区.
FA 缓冲区缓存键将小于 32768 tokens 的上下文长度与不少于 32768 tokens 的长度分开.
此路径保留 BF16 概率和原始标量 softmax reduction tree.

两个非空请求可以使用[独立的 warp kernel](../python/oh_my_vllm/kernels/cuda_backend/fa_core.cuh#L16)<!-- source: if (batch == 2 && rows > max_queries && first_token == 0 && min_context >= 32768 && -->.
仅当 `2 <= max_queries <= 5`, `rows > max_queries`, `first_token == 0`, 且 `min_context >= 32768` 时使用此路径.
提供至少 64 个 partition 的容量.
缓冲区 cache key 包含此选择条件.

其他小输入使用 [16 行或 32 行 query tile](../python/oh_my_vllm/kernels/cuda_backend/fa_core.cuh#L198)<!-- source: if (max_queries * 6 <= 16) -->.
FA 每块使用 256 个线程合并两份或更多注意力的部分结果.
只有一份部分结果时, 每块使用 32 个线程.

[数值契约](operators/model.zh.md#中间精度)使用固定版本的 vLLM 参考, 并允许所选的 FP8 或 BF16 中间操作数.
项目不引入 vLLM 运行时依赖.
具名容差适用于特定输出. Cache 检查和状态选择保持各自的独立要求.

性能测量使用[评估流程](performance.zh.md).
算子测试不设最低效率要求.
每次交付必须完成全部算子测量, 并更新两种语言的 README 表格.
Roofline 计算要求使用选定 GPU UUID 的[实测峰值](calibration.zh.md).
HTTP 比较保留 50% 的效率要求.

Worker 提供约束解码, MTP4, 以及仅含 decode 请求的 batch 的 CUDA Graph 执行.
模型, MTP context 和 MTP draft 使用[固定 metadata 缓冲区](../python/oh_my_vllm/worker/metadata.py#L5)<!-- source: class Metadata: -->.
数据类型相同的数组共用一个缓冲区. 每次更新对每个缓冲区执行一次复制.
每个视图的起始位置都位于 16 字节边界.

采样对 mask, 参数, 随机计数器和 token 历史使用[固定 CPU 缓冲区](../python/oh_my_vllm/worker/sampling.py#L9)<!-- source: class SamplingInputs: -->.
仅当惩罚项使用 token 历史时, 才将其复制到 GPU.
MTP 验证不下载未使用的采样 token.

worker 使用 CPU 前缀筛选与完整 token 区间比较来[检查 token 历史](../python/oh_my_vllm/worker/repetition.py#L9)<!-- source: def cycle_continuations(history): -->.
历史窗口, 循环长度与最小重复次数保持不变.

对于所有行温度均为零的 batch, 采样使用[两阶段最大值归约](../python/oh_my_vllm/kernels/cuda_backend/sample_greedy.cuh#L6)<!-- source: __global__ void greedy_scores_kernel( -->.
它保留 MTP 接受所需的 one-hot 概率. 随机和混合 batch 使用排序路径.
Sampler 为各路径保留[独立的 CUDA Graph](../python/oh_my_vllm/worker/sampling.py#L113)<!-- source: self.replays[self.greedy] = self.resources.enter_context( -->.
它[在 CPU 上计算采样随机计数器](../python/oh_my_vllm/worker/sampling.py#L122)<!-- source: counters = self.inputs.random[:, 1] + (self.inputs.parameters[:, 0] > 0) -->.

验收审计使用[内核目录中的原生源码](../development/kernels/acceptance.py#L128)<!-- source: source = SOURCES / f"{name}.cu" -->.
它必须包含[贪心数值证据](../development/kernels/acceptance.py#L336)<!-- source: def verify_greedy(check): -->.

`copy_bytes` 现在只在目标在 CPU 上时才同步.
`gather_rows_kernel` 在行和基地址满足 16 字节对齐时使用 16 字节向量化加载.

FFN 的 [`paired_up_kernel`](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L59)<!-- source: uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST), --> 和 [`decode_projection_kernel`](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L76)<!-- source: uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST), --> 对权重加载使用 `EVICT_FIRST` 和 `CU_TENSOR_MAP_L2_PROMOTION_NONE`.

[行拷贝](../python/oh_my_vllm/kernels/cuda_backend/device.cu#L217)<!-- source: int column_blocks = std::clamp<int64_t>(row_bytes / 4096 + (row_bytes % 4096 != 0), 1, -->可为每行使用多个 block.
行数限制每行的 block 数量. 每行至少使用一个 block.

MTP [接受操作](../python/oh_my_vllm/worker/speculation.py#L158)<!-- source: self.gather = device.empty((batch * 5,), "int32") -->保留固定的行索引缓冲区.
MTP context [直接读取模型和边界的 hidden 缓冲区](../python/oh_my_vllm/worker/speculation.py#L53)<!-- source: self.boundary_hidden = cache.hidden -->.
Context 持有这些缓冲区. 它的 graph 缓存键包含源缓冲区身份.
Draft 计算为下一阶段使用[采样输出的视图](../python/oh_my_vllm/worker/speculation.py#L479)<!-- source: previous[index] = d.view( -->.
每个阶段先复制所有必要的行, 然后采样才再次使用源缓冲区.

greedy MTP 请求可以使用[一个 CUDA Graph 执行四个 draft 阶段](../python/oh_my_vllm/worker/speculation.py#L409)<!-- source: if can_chain(batch, active, graph, self.cache.blocks): -->.
此路径适用于没有 penalties, grammar, thinking, stop strings 或 repetition guard 的请求.
最小输出长度必须在全部四个阶段禁止选择 EOS.
四个 draft 位置必须具有私有 KV pages.

各阶段使用相同的 sampling 输入缓冲区. 每个阶段具有不同的输出缓冲区.
worker [一次下载四个 draft token](../python/oh_my_vllm/worker/draft_chain.py#L122)<!-- source: tokens = d.download(self.tokens, np.int64) -->, 并将概率复制到请求缓冲区.
其他请求在每个 draft 阶段后下载 token.

两项性能审查使用同一份源码.

进一步修改 kernel 前, 使用[性能计划](operators/performance-plan.zh.md)及固定版本的公开 attention 比较.
使用 producer 与 consumer 融合. 推理 worker 不使用 PDL.
模型检查, 可复用经验及临时产物清理遵循[优化流程](operators/development.zh.md#优化流程).

按[端到端要求](performance.zh.md)测量完整 HTTP 请求时间.
性能门槛使用记录的 trace, 并累加其中 serving 算子的 roofline 时间.
每个 provider 都要有数值检查和完整边界测量.

调度器优先选择 decode, 并在 prefill 请求之间分配剩余 token 预算.
默认 token 预算为 32768. 默认运行中请求的最大数量为 4.
使用[服务选项](serving.zh.md#启动)修改这些设置.

工作使用 `main`.
主机路径和产物位置保存在被忽略的本地配置中.
保持[运行时契约](requirements.zh.md)及[进程边界](architecture.zh.md).

所有英文文档继续遵循 ASD-STE100. 不在本文档中记录测试结果.
