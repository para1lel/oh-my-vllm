# CUDA 候选安全

新的 CUDA 候选运行前, 先检查存储和形状契约.
编译及 `can_implement` 成功不能证明所有形状的内存安全.
实验通过完整数值与 replay 检查前, 保存在仓库外部.

## 窄矩阵 scale 拷贝失败

外部 transposed split-K 实验测试了 8, 16 和 32 列 tile.
1 行和 4 行下的 18 个组合完成了抽样数值, replay, canary 和计时检查.
实验在 20 行, 8 列 tile, 2 个 slices 的案例中出现非法内存访问.
该候选已被撤销. 实验源码没有进入生产实现.

独立源码审查发现, 8 列和 16 列变体存在共享内存边界缺陷.
固定版本的 [CUTLASS builder](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/include/cutlass/gemm/collective/builders/sm100_blockwise_umma_builder.inl#L178) 使用 32 个 scale 拷贝线程.
它的[拷贝谓词](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/include/cutlass/gemm/collective/sm100_mma_warpspecialized_blockwise_scaling.hpp#L963) 检查全局 scale 边界.
它没有把这些写入限制在较小局部 tile 的 scale 存储内.
更多有效全局行可能造成该存储范围之外的写入.

这一缺陷与观测到的形状边界一致. 它不能定位具体 fault 指令, 也不能证明唯一原因.
实验缺少 Compute Sanitizer 的 fault 定位证据.
部分数值和 canary 结果不能证明完整内存安全.

[生产窄矩阵入口](../../python/oh_my_vllm/kernels/cuda_backend/groupwise_transposed.cuh) 使用 32 列 tile, 并限制最多 32 行.
这一观测不增加生产验收结果.

## 候选流程

1. 检查每个操作数的 dtype, device, shape, stride, alignment, capacity 和 alias 契约.
2. 除法, 乘法或指针运算前, 验证标量选项.
3. 分别检查全局, 局部 tile, stage, batch 和共享内存边界.
4. 新形状或新布局运行前, 取得独立源码审查.
5. 计时前, 对每个边界形状运行 Compute Sanitizer. 首次失败后停止.
6. 检查输入变化, 非默认 stream, replay, 数值输出和最终状态.
7. 这些检查通过后, 再采集成对计时.

初次运行每个形状时, 使用新进程. 不与失败候选共享 CUDA context.
不要将未经检查的失败候选移到其他 GPU.
停止准确归属任务的进程, 并在失败后记录设备状态.
将失败源码, 构建身份, 日志和审查保存在外部证据目录.

这些检查降低已知风险. 它们不能证明候选不会失败.
继续遵守[数值契约](../requirements/kernels.zh.md)和[测量方法](../profiling/measurement.zh.md).
