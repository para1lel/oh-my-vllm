# CUDA 算子

使用 CUDA C++ 和 CUTLASS, 或[运行时契约](../../../docs/requirements.zh.md)允许的 CuTe DSL.
使用 [GPU 执行规则](../../../docs/execution.zh.md)和[算子设计](../../../docs/operators/design.zh.md).
模型维度和 checkpoint 格式放在所选适配器的契约中.
保持原始权重值和 scale.

完整操作有收益时融合 producer 和 consumer 计算.
接口必须允许已准备的设备布局, 共享量化和有收益的 PDL.
测量完整算子边界和完整推理时间.

使用调用者的 stream, Eager 和 CUDA Graph 执行使用相同算子.
启动前检查 shape, 类型, stride, 设备和存储重叠.
模型专属接受与停止处理完成前, 推测写入保持私有.
遵循适配器的 checkpoint, 精度和恢复规则.

使用[测试范围](../../../docs/testing.zh.md#模型验收)中所选模型的冻结数值参考, fixtures 和限值.
不将推理输出作为预期值.
正确性, 内存安全和性能证据分别保存.
