# 运行时契约

构建在单张 NVIDIA B200 上运行的 Qwen3.8-27B-FP8 推理服务.
Rust 负责服务, 调度器和逻辑 KV cache.

Python 负责 GPU 计算. 进程间使用 ZMQ DEALER 和 msgpack.

保持块大小 784, 16 层 FA, 48 层 GDN 和 `mamba_cache_mode="align"`.
输入与输出合计支持到 262144 tokens.

保留离线及 HTTP 功能, 普通解码, MTP4, DSpark 和数值容差.
不改变 DSpark checkpoint 的字节内容.

使用 Torch 操作和仓库 CUDA. 允许使用原生 CUDA 库.
不引入 vLLM 依赖, 环境或缓存.

开发工具和推理包使用不同的依赖文件.

详细功能和验收标准见以下历史契约:

- [核心](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/core.zh.md)
- [性能](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/performance.zh.md)
- [服务](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/service.zh.md)
- [算子](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/kernels.zh.md)
- [教程](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/learning.zh.md)

工作流程, 文档范围和测试结果记录以[贡献流程](../CONTRIBUTING.zh.md)为准.
历史记录不构成新代码的验收证据.
