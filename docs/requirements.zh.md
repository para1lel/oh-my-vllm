# 运行时契约

构建在单张 NVIDIA B200 上运行的 Qwen3.8-27B-FP8 推理服务.
Rust 负责服务, 调度器和逻辑 KV cache.

Python 负责 GPU 计算. 进程间使用 ZMQ DEALER 和 msgpack.

保持块大小 784, 16 层 FA, 48 层 GDN 和 `mamba_cache_mode="align"`.
输入与输出合计支持到 262144 tokens.

保留离线及 HTTP 功能和数值容差.
只使用普通解码和 MTP4.

支持文本, 图片和视频输入.
所有模型算子使用 CUDA C++ 或 CuTe DSL.
原生 CUDA 库可以提供基础设施和 tile task 的 device 组件.
独立库 tensor launch 只能作为临时迁移路径.
不使用 DeepGEMM, FlashInfer 等其他外部算子库.
不引入 vLLM 依赖, 环境或缓存.

开发工具和推理包使用不同的依赖文件.

数学和状态行为使用[模型规格](operators/model.zh.md).
GPU 实现使用 [tile 执行设计](tile-execution.zh.md)和[迁移计划](tile-migration.zh.md).
最终验收时, 全部 GPU tensor 计算必须使用依赖 tile 的 MegaKernel 执行.
迁移期间允许临时保留 native 算子路径.

迁移期间不设性能门槛.
最终实现必须使全部 24 种 [HTTP 配置](performance.zh.md#配置)达到规定的标称资源门槛.
算子计时, 公开算子比较和实测峰值校准不属于验收要求.
保留适用的数值和状态检查.
每次交付前更新两种语言的 README 表格.

本文所述功能和模式的详细标准见以下历史契约:

- [核心](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/core.zh.md)
- [性能](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/performance.zh.md)
- [服务](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/service.zh.md)
- [算子](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/kernels.zh.md)
- [教程](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/learning.zh.md)

工作流程, 文档范围和测试结果记录以[贡献流程](../CONTRIBUTING.zh.md)为准.
历史记录不构成新代码的验收证据.
