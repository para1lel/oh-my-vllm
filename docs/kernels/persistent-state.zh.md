# GDN 状态契约

[Python GDN](../../python/oh_my_vllm/kernels/gdn.py) 负责张量验证和 GPU 计算.
Rust 选择已提交的读取槽和独立的推测写入槽. 被拒绝的候选不能改变已提交的前缀状态.
保留 FP32 持久状态和可观察的 BF16 边界.

[原生扫描](../../python/oh_my_vllm/kernels/cuda_backend/operators/gdn_chunk.cuh)必须保留存储, 流和重放契约.
[状态测试](../../tests/numerical/test_gdn_prefill_persistent.py)检查抵消, 空序列, 填充和错位行, 输入字节和输出哨兵.
验证范围见 [数值要求](../requirements/kernels.zh.md) 和 [验收索引](../acceptance.zh.md).
