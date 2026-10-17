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

Vision [MLP 下投影](../python/oh_my_vllm/kernels/cuda_backend/vision_classic.cuh#L147)<!-- source: inline void vision_classic_down( --> 在行数大于 32 时使用经典 cuBLAS.
BF16 输入生成 FP32 投影存储, 原有 BF16 偏置与残差舍入不变.
自有 handle 和工作区在首次捕获前准备, 在相关图销毁后关闭.
构建与加载器绑定供应商头文件, 映像和链接标志, 并核对实际加载的映像.

Vision [Q/K/V 打包](../python/oh_my_vllm/kernels/cuda_backend/vision.cu#L49)<!-- source: __shared__ float cosines[72], sines[72]; --> 为每个 token 行计算一次位置系数.
一个 block 在 attention head 之间共享经 BF16 舍入的 sine 和 cosine 值.
Bias 舍入, 配对维度, 补齐零值及帧顺序保持不变.

Vision [attention](../python/oh_my_vllm/kernels/cuda_backend/vision_attention.cuh#L18)<!-- source: static_assert(Dimension == 80); --> 对具有 72 个维度的 head 计算 80 个维度.
全局存储保留每行 128 个元素的 stride. Unpack 操作读取第 0 至 71 维.
分组 TMA 拷贝使用固定 CUTLASS collective 的共享内存布局及原始 softmax scale.

采样归一化每块使用 640 个线程.
MTP 输入行数不超过 32 时, 输入归一化每块使用 640 个线程.
对于这些输入, MTP 使用分为四个 K 分区的 [BF16 投影](../python/oh_my_vllm/kernels/cuda_backend/mtp_projection.cuh#L114)<!-- source: inline void mtp_projection_launch(TensorView input, TensorView weight, TensorView partials, -->.
它在存储 BF16 输出前累加 FP32 部分结果.

MTP [端点复制](../python/oh_my_vllm/kernels/cuda_backend/mtp.cu#L151)<!-- source: for (int vector = threadIdx.x; vector < 640; vector += blockDim.x) { --> 使用 16 字节加载和存储.
原有张量对齐契约继续适用.
残差归一化使用 [16 字节输入加载](../python/oh_my_vllm/kernels/cuda_backend/normalization.cuh#L19)<!-- source: uint4 input = *reinterpret_cast<const uint4 *>(x + row * 5120 + column); -->.

FFN 下投影, FA 输出和 GDN 输出对行数不超过 8 的输入使用共用的 [projection 实现](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L191)<!-- source: inline void decode_projection_launch(TensorView activation, TensorView weight, -->.
单行输入使用[固定行数](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L196)<!-- source: decode_projection_rows_launch<1>(activation, weight, activation_scale, weight_scale, partials, -->, 单列 TMEM 加载及固定数组.
其他输入使用运行时行数.
原生算子使用普通 CUDA stream 启动, 不使用 PDL.
当输入有 1, 4 或 5 行时, FFN 使用一次 [CUDA 启动](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L173)<!-- source: inline void paired_up_launch(TensorView input, TensorView input_scale, TensorView weight, --> 完成上投影, 激活和量化.

对于超过 64 行的输入, FFN prefill 使用[成对 FP32 累加](../python/oh_my_vllm/kernels/cuda_backend/ffn_pair_accum.cuh#L41)<!-- source: template <class Base> struct FFNPairedAccum : Base { -->.
它保留原有 K-group 顺序, 缩放因子和 BF16 存储.
上投影使用 [TMA multicast](../python/oh_my_vllm/kernels/cuda_backend/ffn_prefill.cuh#L43)<!-- source: using Cluster = std::conditional_t<MulticastA, Shape<_1, _2, _1>, Shape<_1, _1, _1>>; -->, 将输入分发给输出列不同的两个 block.
这些 block 共用相同的输入 tile. 下投影使用独立的 TMA 加载.

输入为 2 至 32 行时, GDN 通过一次 [CUDA launch 计算门控投影和卷积](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L208)<!-- source: if (rows > 1 && rows <= 32) -->.
验证输入只有 1 行时, GDN 用一次 [CUDA launch](../python/oh_my_vllm/kernels/cuda_backend/gdn_input.cuh#L142)<!-- source: if constexpr (FuseConvolution) { --> 计算输入投影和卷积.

对于单请求的一行验证输入, GDN 使用 [64 位状态加载和存储](../python/oh_my_vllm/kernels/cuda_backend/gdn_kernels.cuh#L234)<!-- source: uint2 packed = *reinterpret_cast<const uint2 *>(address); -->.
成对 BF16 转换保留每个元素的 RN 转换. Live carry 和递推算术保持 FP32.

对于超过 32 行的 prefill 输入, GDN 在 BF16 投影后[预先计算门控](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L81)<!-- source: bool precomputed_gates = !verify && rows > 32; -->.
独立 block 处理每组 16 个 state value 行.
每个 block 向两个 shared-memory 槽位[预取 32 个 token](../python/oh_my_vllm/kernels/cuda_backend/gdn_kernels.cuh#L145)<!-- source: constexpr int GDNPrefetchRows = 32; -->.
跨槽位和 BF16 checkpoint 存储时, FP32 carry 保留在寄存器中.
独立 kernel 执行输出归一化和量化.

在此 prefill 路径中, 每个 [16-lane 组](../python/oh_my_vllm/kernels/cuda_backend/gdn_kernels.cuh#L189)<!-- source: __device__ inline float gdn_half_warp_sum(float value) { -->处理一个 value 行.
两组各含四个 key 的和保持原有归约顺序.

Value 分组也适用于[单个五行验证请求](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L79)<!-- source: bool split_values = (!verify && rows > 32) || (verify && batch == 1 && rows == 5); -->.
对于其他超过 1 行的验证输入, GDN 递推计算每块使用 [512 个线程](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L259)<!-- source: run(std::integral_constant<int, 512>{}); -->.

FA 输入投影为[每个 block 使用 64 个输出列](../python/oh_my_vllm/kernels/cuda_backend/fa_small_input.cuh#L139)<!-- source: inline void fa_small_input_launch(TensorView input, TensorView input_scale, TensorView weight, -->.
full 和 query-only 路径对 1, 4 或 5 个输入行使用此 kernel.
FP32 缩放更新保留 K-group 顺序及原有 BF16 输出转换.

当 attention scratch 缓冲区空间足够时, [KV-only 路径](../python/oh_my_vllm/kernels/cuda_backend/fa.cu#L103)<!-- source: bool split_kv = variant == 1 && (rows == 4 || rows == 16) && -->对 4 或 16 行使用八个 K 分区.
Head kernel 在 BF16 转换, K 归一化, RoPE 和私有 KV 写入前合并 FP32 partials.
这会改变 FP32 结合顺序. 使用命名容差.
其他 KV-only 输入使用 groupwise 实现.

FA prefill 对完整且对齐的 KV tile 使用 [TMA 拷贝](../python/oh_my_vllm/kernels/cuda_backend/fa_prefill.cuh#L136)<!-- source: if (first % 16 == 0 && base + 64 <= last) { -->.
每个描述符 box 都位于一个 784-token 页内.
向量拷贝处理未对齐的起点和最后一个部分 tile.

FA 的 [decode 加载](../python/oh_my_vllm/kernels/cuda_backend/fa_attention.cuh#L75)<!-- source: Position base_offset = base % 784; -->在向量循环之前计算 cache page 指针.
每个请求有 1 个 query, 输入行数不超过 32 且 KV 分区数为 64 时, FA 使用[转置的 warp 矩阵乘法](../python/oh_my_vllm/kernels/cuda_backend/fa_q1_warp_transpose.cuh#L17)<!-- source: __global__ void fa_q1_warp_partial_kernel( -->.
此路径保留 BF16 概率和原始标量 softmax reduction tree.

其他小输入使用 [16 行或 32 行 query tile](../python/oh_my_vllm/kernels/cuda_backend/fa.cu#L206)<!-- source: if (max_queries * 6 <= 16) -->.
FA 每块使用 256 个线程合并两份或更多注意力的部分结果.
只有一份部分结果时, 每块使用 32 个线程.

[数值契约](operators/model.zh.md#中间精度)遵循固定 vLLM 源码版本的类型及舍入点.
项目不引入 vLLM 运行时依赖.
具名容差适用于特定输出. Cache 检查和状态选择保持各自的独立要求.

60% 性能要求仍有工作待完成.

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

验收审计使用[内核目录中的原生源码](../development/kernels/acceptance.py#L117)<!-- source: source = SOURCES / f"{name}.cu" -->.
它必须包含[贪心数值证据](../development/kernels/acceptance.py#L236)<!-- source: def verify_greedy(check): -->.

已选定的 kernel 和 worker 改动已集成. 进一步优化暂缓.
根目录 [README](../README.zh.md#性能测量) 包含算子摘要, HTTP 测量及预热范围.
`copy_bytes` 现在只在目标在 CPU 上时才同步.
`gather_rows_kernel` 在行和基地址满足 16 字节对齐时使用 16 字节向量化加载.

FFN 的 [`paired_up_kernel`](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L59)<!-- source: uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST), --> 和 [`decode_projection_kernel`](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L76)<!-- source: uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST), --> 对权重加载使用 `EVICT_FIRST` 和 `CU_TENSOR_MAP_L2_PROMOTION_NONE`.

[行拷贝](../python/oh_my_vllm/kernels/cuda_backend/device.cu#L182)<!-- source: int column_blocks = std::clamp<int64_t>(row_bytes / 4096 + (row_bytes % 4096 != 0), 1, -->可为每行使用多个 block.
行数限制每行的 block 数量. 每行至少使用一个 block.

MTP [接受操作](../python/oh_my_vllm/worker/speculation.py#L148)<!-- source: self.gather = device.empty((batch * 5,), "int32") -->保留固定的行索引缓冲区.
MTP context [直接读取模型和边界的 hidden 缓冲区](../python/oh_my_vllm/worker/speculation.py#L53)<!-- source: self.boundary_hidden = cache.hidden -->.
Context 持有这些缓冲区. 它的 graph 缓存键包含源缓冲区身份.
Draft 计算为下一阶段使用[采样输出的视图](../python/oh_my_vllm/worker/speculation.py#L469)<!-- source: previous[index] = d.view( -->.
每个阶段先复制所有必要的行, 然后采样才再次使用源缓冲区.

greedy MTP 请求可以使用[一个 CUDA Graph 执行四个 draft 阶段](../python/oh_my_vllm/worker/speculation.py#L399)<!-- source: if can_chain(batch, active, graph, self.cache.blocks): -->.
此路径适用于没有 penalties, grammar, thinking, stop strings 或 repetition guard 的请求.
最小输出长度必须在全部四个阶段禁止选择 EOS.
四个 draft 位置必须具有私有 KV pages.

各阶段使用相同的 sampling 输入缓冲区. 每个阶段具有不同的输出缓冲区.
worker [一次下载四个 draft token](../python/oh_my_vllm/worker/draft_chain.py#L122)<!-- source: tokens = d.download(self.tokens, np.int64) -->, 并将概率复制到请求缓冲区.
其他请求在每个 draft 阶段后下载 token.

若重新启动优化, 缩短执行时间. 用同一份源码完成两项性能审查.

进一步修改 kernel 前, 比较 TMA 权重布局与原生 BF16 矩阵库.
增加打包权重存储前, 比较 score 缓冲区调度与直接 TMA 描述符.
将权重补充位置与累加器读取者的完成时机比较.
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
