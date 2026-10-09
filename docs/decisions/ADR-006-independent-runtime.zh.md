# ADR-006: 独立 GPU 运行时

日期: 2026-09-21. 状态: accepted, 已实现.
本决策替代 [ADR-002](ADR-002-gpuworker-adapter.zh.md) 和 [ADR-005](ADR-005-v2-model-runner.zh.md).

## 决策

保持 Rust 服务, 调度, 逻辑 KV 和 Python GPU 执行.
使用小型项目实现和独立库, 替代 vLLM 运行时依赖.
只在保留来源和许可证时移植选定代码; 不整体复制框架.
构建, 测试和推理必须不依赖 vLLM package, checkout, 旧环境或编译缓存.

按依赖顺序选择兼容稳定版本, 固定已验证组合.
使用主机驱动和可用 CUDA / 编译器工具, 高层库安装在项目环境中.
实际主机位置记入 `LOCAL.md`.

## 影响

项目拥有 FP8 weight/scale, GQA/GDN, MTP rollback, loading, cache state, sampling, graph 和服务 adapter.
保留实际路径独立 FP64 参考和全部现有功能.
每个里程碑独立审查, 测量受影响热路径.
[需求](../requirements.zh.md) 和 [验收](../acceptance.zh.md) 定义有效范围和实测身份.

## 后续边界

新架构和 GPU 后端在声明支持前需要具体合同和测试.
多 GPU 需要 rank-local state 和 collective, Rust 保持调度所有权.
DSpark 实现必须具备独立 draft state, target hidden feature 和可变 verification/commit/rollback.
其 5 层 BF16 设计, feature index 和 block 区别见 [架构](../architecture.zh.md#扩展边界).
本轮不添加空接口或多节点支持.
