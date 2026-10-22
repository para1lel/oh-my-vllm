# 运行时契约

下列现有模型专属契约适用于 Qwen.
[Nightly 提案](deepseek-v41/design.zh.md)定义规划的 DeepSeek 扩展, 不表示当前可执行程序已支持.
扩展期间保持共享的语言, 传输, 算子和执行约束.

构建在单张 NVIDIA B200 上运行的 Qwen3.8-27B-FP8 推理服务.
Rust 负责服务, 调度器和逻辑 KV cache.

Python 负责 GPU 计算. 进程间使用 ZMQ DEALER 和 msgpack.

保持块大小 784, 16 层 FA, 48 层 GDN 和 `mamba_cache_mode="align"`.
输入与输出合计支持到 262144 tokens.

保留离线及 HTTP 功能和全模型数值检查.
只使用普通解码和 MTP4.

支持文本, 图片和视频输入.
所有模型算子使用 CUDA C++ 或 CuTe DSL.
原生 CUDA 库可以提供基础设施和算子计算.
不使用 DeepGEMM, FlashInfer 等其他外部算子库.
不引入 vLLM 依赖, 环境或缓存.

开发工具和推理包使用不同的依赖文件.

数学和状态行为使用[模型规格](operators/model.zh.md).
算子调度和 CUDA Graph 资源使用 [GPU 执行规则](execution.zh.md).
包含 prefill 的 Execute 对每个分片使用 Eager 执行.
仅含 decode 的 Execute 消息使用 CUDA Graph 录制和重放.

使用 24 种 [HTTP 配置](performance.zh.md#配置)测量性能.
使用[评估流程](operators/evaluation.zh.md)中的短输入, 单次迭代 logits 对照.
保留适用的服务, 状态所有权和 graph 资源检查.
交付范围和结果位置使用 [nightly 验证与记录策略](deepseek-v41/validation.zh.md).

本文所述功能和模式的详细标准见以下历史契约:

- [核心](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/core.zh.md)
- [服务](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/service.zh.md)
- [算子](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/kernels.zh.md)
- [教程](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/learning.zh.md)

工作流程, 文档范围和测试结果记录以[贡献流程](../CONTRIBUTING.zh.md)为准.
历史记录不构成新代码的验收证据.
