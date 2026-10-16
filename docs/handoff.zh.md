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
残差归一化使用 [16 字节输入加载](../python/oh_my_vllm/kernels/cuda_backend/normalization.cuh#L19)<!-- source: uint4 input = *reinterpret_cast<const uint4 *>(x + row * 5120 + column); -->.

FFN 和 FA 对行数不超过 8 的输入使用共用的 [projection 实现](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L140)<!-- source: inline void decode_projection_launch(TensorView activation, TensorView weight, -->.
原生算子使用普通 CUDA stream 启动, 不使用 PDL.
当输入有 1, 4 或 5 行时, FFN 使用一次 [CUDA 启动](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L173)<!-- source: inline void paired_up_launch(TensorView input, TensorView input_scale, TensorView weight, --> 完成上投影, 激活和量化.

输入为 2 至 32 行时, GDN 通过一次 [CUDA launch 计算门控投影和卷积](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L150)<!-- source: if (rows > 1 && rows <= 32) -->.
验证输入只有 1 行时, GDN 用一次 [CUDA launch](../python/oh_my_vllm/kernels/cuda_backend/gdn_input.cuh#L142)<!-- source: if constexpr (FuseConvolution) { --> 计算输入投影和卷积.

FA 具有独立的 prefill 实现.
输入行数不超过 32 且每个请求有 1 或 2 个 query 时, FA 使用 [16 行 query tile](../python/oh_my_vllm/kernels/cuda_backend/fa.cu#L156)<!-- source: if (max_queries * 6 <= 16) -->.
FA 每块使用 256 个线程合并两份或更多注意力的部分结果.
只有一份部分结果时, 每块使用 32 个线程.

[数值契约](operators/model.zh.md#中间精度)遵循固定 vLLM 源码版本的类型及舍入点.
项目不引入 vLLM 运行时依赖.
具名容差适用于特定输出. Cache 检查和状态选择保持各自的独立要求.

60% 性能要求仍有工作待完成.

算子优化已停止.
Worker 提供约束解码, MTP4, 以及仅含 decode 请求的 batch 的 CUDA Graph 执行.
工作已按维护者要求暂停.
下一步: 获得维护者指示后, 继续完成端到端性能审查.
使用 producer 与 consumer 融合. 推理 worker 不使用 PDL.

按[端到端要求](performance.zh.md)测量完整 HTTP 请求时间.
每个 provider 都要有数值检查和完整边界测量.

调度器优先选择 decode, 并在 prefill 请求之间分配剩余 token 预算.
默认 token 预算为 32768. 默认运行中请求的最大数量为 4.
使用[服务选项](serving.zh.md#启动)修改这些设置.

工作使用 `main`.
主机路径和产物位置保存在被忽略的本地配置中.
保持[运行时契约](requirements.zh.md)及[进程边界](architecture.zh.md).

所有英文文档继续遵循 ASD-STE100. 不在本文档中记录测试结果.
