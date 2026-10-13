# 文档索引

英文文件为权威原文; 每份均有同目录完整中文译文.
先阅读 [需求](requirements.zh.md), [架构](architecture.zh.md) 和 [当前工作](handoff.zh.md).

## 指南

| 文档 | 用途 |
|---|---|
| [开发](development.zh.md) | 环境, 配置, 依赖和构建. |
| [测试](testing.zh.md) | CPU/GPU 检查及正式验收流程. |
| [服务](serving.zh.md) | HTTP 子集, mask, 思考, 状态和限制. |
| [内核](kernels.zh.md) | CUDA, 冻结 TileLang, 正式用例和开发工具. |
| [Profiling](profiling.zh.md) | 诊断, counter, 来源和计时限制. |
| [验收](acceptance.zh.md) | 实测源码和证据范围的统一索引. |
| [审计](audit.zh.md) | 剩余问题和简短已关闭修复索引. |
| [写作](writing.zh.md) | ASD-STE100 规则, 翻译和自动检查. |
| [术语表](glossary.zh.md) | 项目技术名词, 动词和定义. |
| [教程](../code-journey/README.zh.md) | 13 个交互中文章节和实际源码覆盖. |
| [贡献](../CONTRIBUTING.zh.md) | 审查, 检查和提交流程. |

## 决策记录

| 记录 | 状态和用途 |
|---|---|
| [ADR-001](decisions/ADR-001-zmq-socket-type.zh.md) | 有效: DEALER transport. |
| [ADR-002](decisions/ADR-002-gpuworker-adapter.zh.md) | 已替代: 原始 GPUWorker adapter. |
| [ADR-003](decisions/ADR-003-mtp-state-slots.zh.md) | 有效: BF16 MTP 状态和 block784. |
| [ADR-004](decisions/ADR-004-openai-serving.zh.md) | 有效: Rust HTTP 和真实 OMP 验收. |
| [ADR-005](decisions/ADR-005-v2-model-runner.zh.md) | 已替代: 过渡 V2 runner. |
| [ADR-006](decisions/ADR-006-independent-runtime.zh.md) | 有效: 独立运行时和扩展边界. |
| [ADR-007](decisions/ADR-007-ttft-and-cache-capacities.zh.md) | 有效: TTFT 协议和独立 FA/GDN 容量. |
| [ADR-008](decisions/ADR-008-semantic-ir.zh.md) | 有效: 语义算子和编译 GPU 单元. |
| [ADR-009](decisions/ADR-009-dspark-optional-mode.zh.md) | 有效: 可选 DSpark 模式和绑定源码的独立验收. |
| [ADR-010](decisions/ADR-010-phase-latency-and-execution.zh.md) | 有效: 理论阶段下界和 GPU 执行合同. |
| [ADR-011](decisions/ADR-011-bounded-serving-and-worker-recovery.zh.md) | 有效: 资源上限, 已提交前缀及 worker 恢复. |
| [ADR-012](decisions/ADR-012-reasoning-token-budget.zh.md) | 有效: target 思考预算与 OMP 服务默认值. |
| [ADR-013](decisions/ADR-013-cuda-graph-workspace-ownership.zh.md) | 实现进入验收: cuBLAS workspace 所有权. |
| [ADR-014](decisions/ADR-014-long-prefill-budget-sharing.zh.md) | 实现进入验收: 长 prefill 预算分配. |
| [ADR-015](decisions/ADR-015-shared-serving-sampling-defaults.zh.md) | 实现进入验收: 生产和测试的公共采样默认值. |
| [ADR-016](decisions/ADR-016-canonical-serving-control-tokens.zh.md) | 实现进入验收: 统一思考和工具控制边界. |
| [ADR-017](decisions/ADR-017-disposable-native-goal-soaks.zh.md) | 实现进入验收: 可删除沙箱中的原生 goal. |
| [ADR-018](decisions/ADR-018-client-owned-tool-descriptions.zh.md) | 实现进入验收: 原始工具描述和客户端参数检查. |

架构包含原设计文档.
当前工作包含原计划.
内核开发合并原 CUDA 和 TileLang 指南.
审计保留剩余问题和有用修复身份; 详细任务历史保留在 Git 中.
主机专属维护记录放在被忽略的 `LOCAL.md`.
