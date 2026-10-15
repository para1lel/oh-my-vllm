# 工作状态

Rust 提供 HTTP API, CLI, 调度器, 逻辑 KV cache 和 worker 生命周期控制.
服务模式为普通解码和 MTP4.
HTTP 输入可包含文本, 图片和视频.

Python worker 是占位实现, 不加载模型权重, 不使用 GPU.
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
残差归一化使用 [16 字节输入加载](../python/oh_my_vllm/kernels/cuda_backend/normalization.cuh#L24)<!-- source: uint4 input = *reinterpret_cast<const uint4 *>(x + row * 5120 + column); -->.

FFN 和 FA 对行数不超过 8 的输入使用共用的 [projection 实现](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L147)<!-- source: inline void decode_projection_launch(TensorView activation, TensorView weight, -->.
FFN 使用 [CUDA 启动控制](../python/oh_my_vllm/kernels/cuda_backend/programmatic.cuh#L9)<!-- source: attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization; -->.

FA 具有独立的 prefill 实现.
FA 每块使用 256 个线程合并两份或更多注意力的部分结果.
只有一份部分结果时, 每块使用 32 个线程.

[数值契约](operators/model.zh.md#中间精度)遵循固定 vLLM 源码版本的类型及舍入点.
项目不引入 vLLM 运行时依赖.
具名容差适用于特定输出. Cache 检查和状态选择保持各自的独立要求.

完整算子实现尚未完成. 60% 性能要求仍有工作待完成.

下一步: 优化已测量的瓶颈.
完成每个算子的负载验收.
然后将模型计算接入 worker.
使用[实现计划](operators/implementation.zh.md).
每个 provider 都要有数值检查和完整边界测量.

工作使用 `main`.
主机路径和产物位置保存在被忽略的本地配置中.
保持[运行时契约](requirements.zh.md)及[进程边界](architecture.zh.md).

所有英文文档继续遵循 ASD-STE100. 不在本文档中记录测试结果.
