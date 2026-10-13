# 原生 CUDA backend

使用仓库 CUDA C++, 可为 B200 使用 PTX. 允许原生 CUTLASS, CuTe, CUB 和 CUDA 库.
Python 绑定可以加载仓库 kernel. 不得以 Python 算子库作为 provider.
每个 backend 的 `.cu`, `.cuh`, `.py` 文件最多 800 个物理行.

通过 CUDA hook 使用固定的 clang-format.

## 契约

保持调用者 stream, 对齐, stride, graph replay, 状态写入和数值容差.
融合必须保持可观察的 BF16 舍入和状态边界.
证据身份包含源码, 头文件, 编译器, 配置和已加载模块.

依赖来源不依附于 Python 算子 wheel 或开发工具更新.
使用独立 Torch reference 验证数学契约. 选用实现前测量完整算子和框架.
正式案例和证据摘要放在仓库中. 临时调优与原始 trace 放在外部存储.
