# CUDA 算子

此处用 CUDA C++ 和 CUTLASS 实现[内部阶段](specification.py).
[运行时契约](../../../docs/requirements.zh.md)也允许 CuTe DSL.
使用 [GPU 执行规则](../../../docs/execution.zh.md).
使用 checkpoint 的维度. 保持原始权重值和 scale.

当完整操作变快时, 将小型 producer 操作融合到 consumer 中.
性能测量使用算子 microbench 检查和完整 HTTP 计时.

使用调用者的 stream. Eager 和 CUDA Graph 执行使用相同的算子.
launch 前检查 shape, 类型, stride, device 和存储重叠.

target 接受对应 token 前, 持久写入保留在私有 slots 中.
通过状态 slot 选择接受 token. 不为接受操作复制完整状态.
保持 784-token checkpoint 和 BF16 循环状态存储.

使用[评测流程](../../../docs/operators/evaluation.zh.md).
使用 PyTorch FP64 全模型 logits 参考.
固定算子用例, 仓库 fixture 和验收门槛使用 [microbench 契约](../../../docs/operators/microbench.zh.md).
