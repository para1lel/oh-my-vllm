# 算子与 IR 契约

## REQ-KERNEL-001: 原生依赖边界

GPU 计算使用 Torch 标准算子和仓库 CUDA.
允许 CUTLASS, CuTe 和 CUB 等原生 CUDA 库.
仓库 Python 绑定可以启动自有 CUDA. 不允许外部 Python 算子库.

从当前工作树删除旧比较实现, 依赖, 文档, 测试和证据. 旧结果保留在 Git 历史.
第三方未使用的示例不代表项目可以导入或安装其依赖.

固定的 TileFoundry fork 用于开发. Rebase 到兼容的当前上游, 并验证 CUDA harness.
大幅修改 fork 设计前先讨论. 临时调优保留在仓库外.

## REQ-KERNEL-002: CUDA 与 PTX

B200 使用自有 CUDA C++ 内核, 可包含 inline PTX.
独立执行精度, 框架, 上下文和真实服务门槛.

按功能划分自有 CUDA 后端. 每个 `.cu`, `.cuh` 和 `.py` 文件最多 800 行.
使用固定 clang-format 23.1.3 及 pre-commit hook. 保留 include 依赖顺序.
源码和头文件进入构建, 加载及正式证据身份.

在 15 个工作负载内静态推导每个实现路径的最大合法调用.
相同配置去重. 实现路径不同的阶段分别保留.
不独立最大化无法同时出现的维度, 不以动态 tracing 选择正式用例.
所有合法路径保持独立的数值, 舍入和状态验证, 并保留补充 DSpark 用例.

融合候选与仓库 CUDA 基线至少做 3 轮, 每轮 20 个交错配对.
记录中位数和置信界. 等价链包括必要复制和 reduction.
算子计时用于诊断. 根据实测框架收益选择融合.
框架 2 倍下界门槛是性能验收标准.

正式用例, harness 和汇总证据保留在仓库内.
原始 trace 和临时调优保留在仓库外. Profiling 与验收计时分开.
明确记录缺少的计数器. 门槛保持有效. 使用 [内核流程](../kernels.zh.md).

## REQ-IR-001: Semantic IR

拥有 Qwen target, MTP 和 DSpark 路径中项目 CUDA 算子的语义和 provider 选择.
普通 Torch 算子, embedding, reshape 和 `F.linear` 不在此算子清单内.
每个算子具有 Torch reference, shape / dtype 契约, mutation schema 和静态选择的 provider.
Provider 选择使用 phase, shape, dtype, layout 和 device metadata.
Provider 失败即报错. Reference 只作为显式 debug 选项.

使用 `torch.compile(fullgraph=True)` 编译 prefill, target decode, MTP draft 和四步 proposal.
DSpark target feature, context injection, backbone, Markov step 和 greedy proposal 使用相同 fullgraph 契约.
Provider lowering 前保留语义节点. 主机规划, 分配, 调度和 ZMQ 与这些单元分离.
手动 CUDA Graph 包含编译单元. 禁用编译器管理的 graph.

保留持久写入顺序, capture 恢复和显存保护.
Activation donation 需要对指定临时值的证明. 持久缓存不能被 donation.
Graph rewrite 需要等价测试和注明例外原因的调用点清单.

验收包括现有精度测试, 6 个普通 / MTP4 上下文用例, 正式算子用例和 15 个性能行.
DSpark 服务及补充算子用例保持有效.
真实模型测试必须覆盖全部 4 个编译单元, 以及 metadata 变化后的 graph replay.
不增加编译加速百分比门槛.
