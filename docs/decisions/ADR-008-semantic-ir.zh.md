# ADR-008: Semantic IR 与编译 forward

日期: 2026-09-29. 状态: 在 `619c9d9` 的实测工作集上 accepted.

## 决策

以 `torch.library` 算子作为项目 CUDA 和关键 FlashInfer 算子的语义节点.
每个节点具有 executable reference, fake shape/dtype 合同, mutation schema 和有序 provider.
Provider 从静态 phase, shape, dtype, layout 和 device metadata 选择, 不读取 tensor value.
选择时冻结 registry configuration.
Reference 是显式 debug provider; provider 失败报错.

使用 fullgraph `torch.compile` 编译 prefill, target decode, MTP draft 和四步 proposal.
在 Inductor 前通过小型 FX backend lower 语义节点.
Rust 调度, 主机规划和持久分配保留在编译单元外.
手动 graph capture 已预热的编译单元; 禁用 Inductor graph.
保持持久写入有序, 禁止 persistent-cache donation.

## 影响

模型使用语义名称代替低层入口.
Eager 和编译路径共享 provider 选择.
普通 PyTorch 算子, 包括 `F.linear`, 保留在算子清单外.
经过检查的调用点清单记录每个低层例外.
有限的 FakeTensor, FX metadata 和 backend lookup 使用固定 PyTorch2.14 内部接口, 配有回归测试.

Planned attention 的 CPU handle 保留在动态 device metadata 外.
Graph capture 恢复保持事务性.
已测试的 late rewrite 在生产 provider 兼容时融合 single-use BF16 `silu_mul -> fp8_linear` 链.
Dynamo 限制为 4096, 每 256 个 variant 警告.
这些限制不保证 compiler-memory 容量.
当前未启用 activation donation.

## 验证与替代方案

真实模型集成覆盖 4 个编译单元和 metadata 变化后的 graph replay.
对应源码通过完整正确性 / 上下文套件, 147 个算子用例和 12 个框架行.
3 个完整 prefix-batch1 尝试因 spread 失败; 第 4 个完整采集通过.
编译成功本身不代表性能验收.
参见 [验收](../acceptance.zh.md) 和 [开放限制](../audit.zh.md).

项目拥有此有限 DSL, 不导入 vLLM 运行时, 不构建通用编译器.
[vLLM IR 设计](https://docs.vllm.ai/en/latest/design/vllm_ir/) 是设计参考.
