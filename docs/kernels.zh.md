# 内核开发

使用 [算子契约](requirements/kernels.zh.md), [harness 指令](../development/kernels/AGENTS.zh.md) 和 [性能协议](requirements/performance.zh.md).
当前 provider 迁移状态见 [交接](handoff.zh.md). 不把旧测量当作新验收.
选择新的数值或调度路径前, 阅读 [算法调研和实测经验](kernels/optimization.zh.md).

## 实现来源

[CUDA backend](../python/oh_my_vllm/kernels/cuda_backend/) 负责设备代码和原生绑定.
[Semantic IR](../python/oh_my_vllm/ir/) 负责算子契约, reference, mutation schema 和 provider 选择.
[目标模型](../python/oh_my_vllm/models/qwen.py) 提供调用位置的上下文.

调度, 布局 guard, launch 参数和特化阈值放在这些源码中.
Torch 不提供的算子使用仓库 CUDA. 允许原生 CUDA 库.
Python 绑定是进入自有 kernel 的桥接, 不是外部算子 provider.

每个 backend 的 `.cu`, `.cuh`, `.py` 文件最多 800 行.
使用固定的 clang-format 和定义空行 hook. 必要的数值与寿命说明保持简短.

```bash
scripts/with-env.sh python scripts/format_cuda.py
scripts/with-env.sh python scripts/format_cuda.py --check
scripts/with-env.sh python scripts/check_definition_spacing.py
```

## 数值与状态契约

每个算子有输入/输出 shape, dtype, stride, 对齐和 mutation 契约.
使用独立 Torch reference, 遵守规定精度与舍入边界.
更高精度不能替代规定的 FP32 除法或可观察的 BF16 cast.
除返回值比较外, 保留已写状态 witness 和未修改存储检查.

融合必须保持数学语义, 已接受状态边界和数值容差.
缓存和状态存储不能作为临时 donation 目标.
在 graph replay 中检查变化的输入和 metadata. launch 前拒绝不支持布局或无效绑定.

## 正式案例

从全部十五个框架工作负载推导每条实现路径的最大合法调用.
路径不同时区分阶段. 相同配置去重.
不单独最大化不能同时出现的维度.
正式案例, 独立 reference, harness 和证据摘要放在仓库中.

算子计时用于诊断瓶颈和比较 candidate. 完整框架提供最终性能门槛.
比较完整算子, 包含必需复制, reduction 和转换.
使用交错 candidate/baseline 测量. profiling 与验收计时分开.
保留原始失败尝试和不可用计数器的状态.

## 工具与构建身份

TileFoundry 是开发工具. 更新不得悄悄改变生产构建依赖.
原生头文件单独固定, 将其来源身份与编译器选项, 已加载模块哈希一起记录.
计时前完成编译和 graph capture.
临时调优和原始 trace 放在外部存储. 每次运行后停止自有 GPU 进程.

主机/设备归因见 [profiling](profiling.zh.md), 采集见 [测试命令](testing/performance.zh.md).
