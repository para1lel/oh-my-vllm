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

采样归一化每块使用 640 个线程.
MTP 输入行数不超过 32 时, 输入归一化每块使用 640 个线程.
对于这些输入, MTP 使用分为四个 K 分区的 [BF16 投影](../python/oh_my_vllm/kernels/cuda_backend/mtp_projection.cuh#L114)<!-- source: inline void mtp_projection_launch(TensorView input, TensorView weight, TensorView partials, -->.
它在存储 BF16 输出前累加 FP32 部分结果.
残差归一化使用 [16 字节输入加载](../python/oh_my_vllm/kernels/cuda_backend/normalization.cuh#L19)<!-- source: uint4 input = *reinterpret_cast<const uint4 *>(x + row * 5120 + column); -->.

FFN 和 FA 对行数不超过 8 的输入使用共用的 [projection 实现](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L140)<!-- source: inline void decode_projection_launch(TensorView activation, TensorView weight, -->.
原生算子使用普通 CUDA stream 启动, 不使用 PDL.
当输入有 1, 4 或 5 行时, FFN 使用一次 [CUDA 启动](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L173)<!-- source: inline void paired_up_launch(TensorView input, TensorView input_scale, TensorView weight, --> 完成上投影, 激活和量化.

输入为 2 至 32 行时, GDN 通过一次 [CUDA launch 计算门控投影和卷积](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L152)<!-- source: if (rows > 1 && rows <= 32) -->.
验证输入只有 1 行时, GDN 用一次 [CUDA launch](../python/oh_my_vllm/kernels/cuda_backend/gdn_input.cuh#L142)<!-- source: if constexpr (FuseConvolution) { --> 计算输入投影和卷积.
对于超过 32 行的 prefill 输入, GDN 用[独立 block](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L178)<!-- source: scan(std::false_type{}, std::integral_constant<int, 16>{}, std::integral_constant<int, 256>{}); --> 处理每组 16 个 state value 行.
独立 kernel 执行输出归一化和量化.
对于超过 1 行的验证输入, GDN 递推计算每块使用 [512 个线程](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L187)<!-- source: run(std::integral_constant<int, 512>{}); -->.

FA 具有独立的 prefill 实现.
输入行数不超过 32 且每个请求有 1 或 2 个 query 时, FA 使用 [16 行 query tile](../python/oh_my_vllm/kernels/cuda_backend/fa.cu#L156)<!-- source: if (max_queries * 6 <= 16) -->.
FA 每块使用 256 个线程合并两份或更多注意力的部分结果.
只有一份部分结果时, 每块使用 32 个线程.

[数值契约](operators/model.zh.md#中间精度)遵循固定 vLLM 源码版本的类型及舍入点.
项目不引入 vLLM 运行时依赖.
具名容差适用于特定输出. Cache 检查和状态选择保持各自的独立要求.

60% 性能要求仍有工作待完成.

Worker 提供约束解码, MTP4, 以及仅含 decode 请求的 batch 的 CUDA Graph 执行.
采样对 mask, 参数, 随机计数器和 token 历史使用[固定 CPU 缓冲区](../python/oh_my_vllm/worker/sampling.py#L9)<!-- source: class SamplingInputs: -->.
仅当惩罚项使用 token 历史时, 才将其复制到 GPU.
MTP 验证不下载未使用的采样 token.

算子和端到端优化正在进行.
`copy_bytes` 现在只在目标在 CPU 上时才同步.
`gather_rows_kernel` 在行和基地址满足 16 字节对齐时使用 16 字节向量化加载.

FFN 的 [`paired_up_kernel`](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L59)<!-- source: uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST), --> 和 [`decode_projection_kernel`](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L68)<!-- source: uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST), --> 对权重加载使用 `EVICT_FIRST` 和 `CU_TENSOR_MAP_L2_PROMOTION_NONE`.

[行拷贝](../python/oh_my_vllm/kernels/cuda_backend/device.cu#L182)<!-- source: int column_blocks = std::clamp<int64_t>(row_bytes / 4096 + (row_bytes % 4096 != 0), 1, -->可为每行使用多个 block.
行数限制每行的 block 数量. 每行至少使用一个 block.

MTP [接受操作](../python/oh_my_vllm/worker/speculation.py#L142)<!-- source: self.gather = device.empty((batch * 5,), "int32") -->保留固定的行索引缓冲区.
MTP context [直接读取模型和边界的 hidden 缓冲区](../python/oh_my_vllm/worker/speculation.py#L50)<!-- source: self.boundary_hidden = cache.hidden -->.
Context 持有这些缓冲区. 它的 graph 缓存键包含源缓冲区身份.
Draft 计算为下一阶段使用[采样输出的视图](../python/oh_my_vllm/worker/speculation.py#L445)<!-- source: previous[index] = d.view( -->.
每个阶段先复制所有必要的行, 然后采样才再次使用源缓冲区.

下一步: 缩短执行时间. 用同一份源码完成两项性能审查.
使用 producer 与 consumer 融合. 推理 worker 不使用 PDL.

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
