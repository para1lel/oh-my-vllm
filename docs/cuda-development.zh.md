# CUDA kernel 开发

已验收的对照源码逐字节复制到 `python/oh_my_vllm/kernels/tilelang_reference`，源码 hash、原提交、依赖锁 hash 和 TileFoundry 固定版本记录在 `development/kernels/tilelang-reference.json`。不得调优对照实现。

在 Python 启动前设置 `OH_MY_VLLM_KERNEL_BACKEND=cuda` 选择新后端，未实现的入口直接报错，不回退 TileLang。迁移期间默认仍为 TileLang，全部 CUDA 验收完成后再切换。原生代码使用独立 TVM FFI、调用方当前 CUDA stream 和 SM100a；TileFoundry 不成为原生算子的运行/构建依赖。必须在图捕获和正式测量前完成编译。

当前里程碑：FP8 量化和 SiLU 已有 CUDA 实现，现有 14 项 FP8/融合 SiLU 正确性测试通过，包括精确量化。其余原生算子、静态最大 shape 配置、计时/指标集成及完整最终验收仍在进行，目前不宣称原生实现提速。

`development/kernels/comparison.py` 要求至少三轮、每轮 20 对样本。采用可复现的分层 bootstrap：先重采样轮次，再采样轮内配对，要求平均节省耗时的单侧 95% 下界大于零，同时每轮 CUDA 中位耗时更低。正式程序需交替后端顺序、预热两条路径、恢复可变状态，并保留全部原始时间对；profiler 运行的耗时不能用于该判定。

静态配置选择和完整验收遵循 REQ-KERNEL-002。硬件指标、编译器资源报告和 TileFoundry HIR 估计分开记录。当前 HIR 缺少 FP64，不能证明长位置 RoPE 相位精度；该边界继续使用未改动的独立参考和完整模型测试。

本里程碑已验证：Nsight Compute 可读取 B200 硬件计数器，过滤 quantize_kernel 的采样成功；TileFoundry analyze 和 Nsight CSV 导入均实际运行。非默认 CUDA stream 和改变输入后的 CUDA Graph 重放通过精确 FP8 比较，TileLang 的 14 项回归亦通过。正式最大配置性能尚未采集，阶段 GPU 进程均已退出。
