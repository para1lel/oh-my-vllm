# ADR-006：拥有 GPU 执行运行时

日期：2026-09-21。状态：用户已批准，已实现。验收证据见 ../acceptance.md。

## 决策

保留 Rust 服务、调度和逻辑 KV 所有权，以及 Python GPU 执行。用小型项目自有实现和独立库替代 vLLM 集成。可改编选定上游代码并保留来源和许可证，但不能整套复制框架。最终构建、测试和服务不依赖 vLLM 包、源码检出、已编译算子、旧 conda 环境或构建缓存。过渡阶段可继续使用旧适配器。

从 GPU 工具链和 tensor 运行时开始，按依赖顺序选择较新、兼容的稳定版本，验证后固定组合。允许使用系统驱动和正常工作的 CUDA/编译工具；高层库放在 conda oh-my-vllm。除非显式修订本决策，不使用 nightly 依赖。

优先调查风险最高的边界：FP8 权重/scale 语义、GQA、GDN 递归和 MTP 状态回滚。随后集成模型加载、请求/缓存状态、采样、计算图执行和服务工具。FP64 观测点与实际执行路径同步迁移，保持参考实现独立。

每个里程碑由相关正确性测试和独立审查把关。热路径修改附带针对性性能检查；执行链完成和最终验收时需要完整九组验收。记录中间性能缺口。仅对比 2026-09-19 原始冻结 EngineCore 基线，门槛 >=95%；不重跑原生 vLLM，不增加相对 V2 的门槛。

## 设计参考

- https://docs.vllm.ai/en/latest/design/model_runner_v2/ ：持久请求状态、增量更新、GPU 元数据和明确的计算图生命周期。
- https://docs.vllm.ai/en/latest/design/vllm_ir/ ：将算子语义与实现分离；不要为单模型任务构建通用编译器 IR。
- https://docs.vllm.ai/en/latest/design/cuda_graphs/ ：计算图捕获与编译分离，使用稳定 buffer 和明确执行 shape。

## 未来扩展（非当前实现要求）

让模型/权重映射、设备 kernel、缓存状态及执行流程保持可读且集中。需要不同模型和 NVIDIA GPU 时，再添加具体实现。单机多 GPU 可围绕相同 Rust 调度契约添加 rank 本地权重/状态和 collective。PD 分离及多机执行不在范围内。不要添加推测性的空接口。

DSpark 主要指 /data0/shared/Qwen3.8-27B-DSpark。配置声明五层 BF16 GQA draft 模型，target 特征层 [5,19,33,47,61]、confidence 和 Markov head、七个 proposal，以及含 bonus 的八个验证 token。draft 的 block_size=7、training_block_size=16 不是 784 的 KV 页大小。未来集成需要独立 draft 状态、target hidden feature，以及可变验证/提交/回滚处理。确切特征提取和 token 映射届时验证；checkpoint README 不能代替本地验收。
