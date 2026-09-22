# CUDA kernel 开发

已验收的对照源码逐字节复制到 `python/oh_my_vllm/kernels/tilelang_reference`，源码 hash、原提交、依赖锁 hash 和 TileFoundry 固定版本记录在 `development/kernels/tilelang-reference.json`。不得调优对照实现。

在 Python 启动前设置 `OH_MY_VLLM_KERNEL_BACKEND=cuda` 选择新后端，未实现的入口直接报错，不回退 TileLang。迁移期间默认仍为 TileLang，全部 CUDA 验收完成后再切换。原生代码使用独立 TVM FFI、调用方当前 CUDA stream 和 SM100a；TileFoundry 不成为原生算子的运行/构建依赖。必须在图捕获和正式测量前完成编译。

当前里程碑：所有自有 kernel 入口均有原生 CUDA 实现，包括 FP64 相位的融合 RMS/RoPE、FP32 循环 GDN、卷积和分页注意力。注意力合并优化后，现有完整 CUDA 测试通过 153 项和 24 项子测试。独立生产入口的 packed Q/K、接近 262144 的 int32/int64 位置、非默认 stream 和修改输入后的图重放通过 FP64 检查。这些是正确性结果，不代表实际模型或性能验收。默认仍为 TileLang。

正式静态矩阵位于 `development/kernels/cases.py`，完整算子 fixture 保留必需的快照及合并操作，包括三条序列的独立分派、提案/追赶元数据和图缓存用满后的普通执行路径。有效 FA 页排除空页 0，不使用动态 shape 提取。运行：

```bash
scripts/with-env.sh python benchmarks/kernels.py --list
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_KERNEL_BACKEND=cuda CUDA_HOME=/usr/local/cuda-13.1 TVM_FFI_CUDA_ARCH_LIST=10.0a python benchmarks/kernels.py --output /tmp/cuda-kernels.json
```

`--operations` 选择诊断子集，部分覆盖不能通过完整矩阵验收。有未提交源码改动的运行仅作诊断。采集程序记录源码 hash、GPU 身份、全部成对样本、置信判定和单独的 TileFoundry 估计；GPU 争用或源码变化使采集失效。首轮诊断因外部 GPU 进程进入而作废，不构成验收结果。注意力及其他小 shape 路径仍需调优。TileFoundry 的 `TileLang` twin 始终绑定冻结实现，不随生产后端选择改变。

`development/kernels/comparison.py` 要求至少三轮、每轮 20 对样本。采用可复现的分层 bootstrap：先重采样轮次，再采样轮内配对，要求平均节省耗时的单侧 95% 下界大于零，同时每轮 CUDA 中位耗时更低。正式程序需交替后端顺序、预热两条路径、恢复可变状态，并保留全部原始时间对；profiler 运行的耗时不能用于该判定。

静态配置选择和完整验收遵循 REQ-KERNEL-002。硬件指标、编译器资源报告和 TileFoundry HIR 估计分开记录。当前 HIR 缺少 FP64，不能证明长位置 RoPE 相位精度；该边界继续使用未改动的独立参考和完整模型测试。

本里程碑已验证：Nsight Compute 可读取 B200 硬件计数器，过滤 quantize_kernel 的采样成功；TileFoundry analyze 和 Nsight CSV 导入均实际运行。非默认 CUDA stream 和改变输入后的 CUDA Graph 重放通过精确 FP8 比较，TileLang 的 14 项回归亦通过。完整有效的最大配置性能和最终框架验收仍待完成。每次运行后停止本任务 GPU 进程。
