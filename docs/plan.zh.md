# 已完成计划：TileLang、TTFT 和长上下文

1. 更新并冻结隔离的官方 vLLM main 基线及其环境。
2. 实现逐请求 TTFT 和独立 FA/GDN 缓存容量。
3. 将七个项目自有 Triton kernel 模块替换为 TileLang，集成仅开发使用的个人 TileFoundry fork 和离线自动调优流程。
4. 调优所需 shape，全部 12 组达到吞吐 >=95%、TTFT <=110% 和用户修订后的 10% 稳定性门槛，保留完整原始测量。
5. 通过未改动的正确性检查、六组 262144-token 普通/MTP4 边界、长前缀、约束、生命周期及真实 oh-my-pi 两种接口。
6. 添加 25 份中文配套文档并同步最终证据。83d77ff 上的最终文档读取复核通过，审查和 GPU 释放核查已完成。最终证据在 main 提交并推送。

当前证据和回答局限见 acceptance.zh.md、handoff.zh.md。

# 实现状态

独立运行时迁移（ADR-006）已实现：稳定依赖固定在 conda oh-my-vllm 中；模型加载、GPU 状态和所需 kernel 由项目或独立库拥有。旧适配器及全部 vLLM 运行/构建/测试依赖已移除。Rust 继续拥有服务、调度和逻辑 KV；ZMQ 协议及 block784 不变。

FP64 算子与实际模型探针、普通/MTP 文本、前缀/抢占、约束、生命周期及 CUDA Graph 回归通过。相对原始基线的九组性能全部通过。文档对齐后，两种最终 oh-my-pi 工作流均已完成；结果和生成回答的局限记录在 acceptance.md 和 handoff.md。

此前 GPUWorker/V2 阶段是历史里程碑，不是现行实现。其证据和决策保留在 bench/baseline 和 docs/decisions 下。未来架构、NVIDIA 后端、单机多 GPU 及本地 DSpark 扩展见 ADR-006，不提供占位接口，也不声称已经支持。CPU KV swap、PD/多机、LoRA 和多模态执行仍不在范围内。继续独立审查、本地 pre-commit 检查、仅 main 提交，以及及时清理任务自有 GPU 程序。
