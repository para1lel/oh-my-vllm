# 窄 FP8 投影契约

[投影实现](../../python/oh_my_vllm/kernels/fp8.py)保留检查点 FP8 权重, FP32 分块缩放和 BF16 输出.
[原生绑定](../../python/oh_my_vllm/kernels/cuda_backend/groupwise.py)验证形状, 步长, 对齐, 容量和调用流.
每次调用和改变输入后的图重放都读取缩放值. 部分结果缓冲和输出哨兵必须位于分配的存储范围内.

使用 [数值要求](../requirements/kernels.zh.md), [候选安全流程](cuda-safety.zh.md) 和 [性能检查](../testing/performance.zh.md).
[验收索引](../acceptance.zh.md)标明各源码范围的验证记录.
