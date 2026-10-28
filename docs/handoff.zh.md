# 工作状态

继续在 `nightly` 上工作.

## 当前服务路径

Qwen 保留现有的单 worker 服务路径.
共享 Rust service, scheduler, logical cache, DEALER transport 和 msgpack protocol 保持启用.

DeepSeek-V4.1-Flash 已有 model adapter 和四 rank Python bridge.
adapter 使用原始 48 shard checkpoint 和 rank-local placement.
它支持 text target execution 和 DSpark5 greedy proposal verification.
bridge 支持 `tp4`, `tep4` 以及已明确记录的 DEP4 executor fallback.

adapter 使用 DeepSeek descriptor, cache groups, tokenizer identity, inference-config identity, native projection, routing, activation 和 transformer support.
prefill row 会按 native operator row limit 切分.
DSpark verification 保持 rejected draft 为 private state, 并输出 target correction.
Engram table 使用 file-backed host mapping.
Lookup 只将选中的 row 分阶段传到 GPU.

DeepSeek execution 当前使用 Eager.
共享 Qwen CUDA Graph path 已有 stream handoff, bounded graph workspace 和 cleanup checks.
DeepSeek multi-rank Graph capture 尚未接入.

## 当前证据

已在 GPU `4,5,6,7` 上使用 TEP4 完成短真实 checkpoint execution.
它覆盖 readiness, ordinary decode, DSpark proposal output 和 draft rejection.
CPU 和 Rust checks 覆盖 descriptor, model, operators, launcher cleanup, matrix scope, speculation 和 shared runtime.

2026 年 10 月 28 日, 完整 Rust HTTP service 使用 TEP4 和短 context 成功 ready.
Smoke run 检查 model listing, diagnostics, ordinary text, DSpark5, SSE usage, validation error 和 shutdown.
DSpark5 HTTP log 记录了 2 个 proposed draft token 和 2 个 verified draft token.
这次 smoke run 是功能证据, 不替代 approved matrix.

[DeepSeek matrix](../development/serving/deepseek_matrix.py) 中有批准的 workload definition.
HTTP runner 定义 36 个 DSpark5 case 和 2 个 ordinary TP4 case.
AgentX runner 定义 TP4/4, TEP4/16 和 DEP4/64 case.
当前还没有记录完整的 approved HTTP 或 AgentX result set.

## 下一步

- 完成真实 TP4 readiness 和 ordinary decode run.
- 为 window, compressed KV, index keys, accepted state, prefix reuse 和 restoration 实现 physical DeepSeek cache contract.
- 用稳定的 request ownership 和 expert dispatch 替换 DEP4 fallback.
- 接入 native sparse attention, hyper-connections 和 transform workspace.
- 添加 coordinated DeepSeek CUDA Graph capture 和 replay path.
- 在 GPU 0 以外的显卡上运行 38 个 HTTP case 和 3 个 AgentX case.
- 交付前取得 independent code review.

原始报告, host settings, model paths 和临时 process records 保留在 repository 外.
host settings 放在 `LOCAL.md` 中.
