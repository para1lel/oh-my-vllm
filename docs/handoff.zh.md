# 工作状态

继续在 `nightly` 上工作.

## 当前服务路径

Qwen 保留现有的单 worker 服务路径.
共享 Rust service, scheduler, logical cache, DEALER transport 和 msgpack protocol 保持启用.

DeepSeek-V4.1-Flash 已有 model adapter 和四 rank Python bridge.
adapter 使用原始 48 shard checkpoint 和 rank-local placement.
它支持 text target execution 和 DSpark5 greedy proposal verification.
bridge 支持 `tp4`, `tep4` 以及已明确记录的 DEP4 executor fallback.

adapter 使用 DeepSeek descriptor, cache groups, tokenizer identity, inference-config identity, native projection, routing, activation, mHC, RMS, sparse attention 和 transformer support.
prefill row 会按 native operator row limit 切分.
DSpark verification 保持 rejected draft 为 private state, 并输出 target correction.
Engram table 使用 file-backed host mapping.
Lookup 只将选中的 row 分阶段传到 GPU.

released DeepSeek shape 支持 native mHC, RMS 和 sparse attention path.
mHC path 为每个 block 保留有界 workspace, 并在超过 native row limit 时使用 reference path.
sparse attention status failure 会禁用 native path, 并使用 reference path.
native sparse packing 通过 `OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE=1` opt-in.
因为每次 call packing 较慢, reference sparse path 保持为默认.

DeepSeek physical page 现在会在 request slot 之间保存 window KV, compressed KV,
index key, Engram hash state 和 compressor tail.
cache operation 支持 logical block store, load, drop, partial prefix copy 和
page-backed host snapshot.
accepted DSpark target state 会 seed MTP window, speculative rollback 会保持
rejected draft state 为 private state.
nonzero multi-token prefill 当前使用 sequential correctness path, fused chunk path
仍是后续 performance work.

DeepSeek execution 当前使用 Eager.
共享 Qwen CUDA Graph path 已有 stream handoff, bounded graph workspace, graph upload 和 cleanup checks.
DeepSeek multi-rank Graph capture 尚未接入.

DeepSeek startup 只读取 `config.json` 和 `model.safetensors.index.json` 生成 descriptor identity.
Ready 之前不会 hash weight shard.
rank-local checkpoint tile 使用 2048 行, 以减少小块 host-to-device copy.
外部 startup trace 使用 `OH_MY_VLLM_STARTUP_TRACE`, 并使用与 repository 分开的 location.

## 当前证据

已在 GPU `4,5,6,7` 上使用 TEP4 完成短真实 checkpoint execution.
它覆盖 readiness, ordinary decode, DSpark proposal output 和 draft rejection.
CPU 和 Rust checks 覆盖 descriptor, model, operators, launcher cleanup, matrix scope, speculation 和 shared runtime.

2026 年 10 月 28 日, 完整 Rust HTTP service 使用 TEP4 和短 context 成功 ready.
Smoke run 检查 model listing, diagnostics, ordinary text, DSpark5, SSE usage, validation error 和 shutdown.
DSpark5 HTTP log 记录了 2 个 proposed draft token 和 2 个 verified draft token.
这次 smoke run 是功能证据, 不替代 approved matrix.

physical cache change 之后, HTTP service 使用 GPU `1,2,3,4`, TEP4, 8 个 block 和 1024-token physical context 成功 ready.
两个 ordinary request 都成功返回.
第二个 request 从 physical prefix cache 复用了 13 个 prompt token.
一个 DSpark5 target request 也成功返回, 且没有 worker restart.
这个 2-token DSpark5 request 没有 proposal budget, 所以 proposal counter 为零.
cache behavior test 覆盖 host snapshot store, load, drop, prefix copy 和 restoration.

最新的 native-operator smoke 使用 GPU `1,2,3,4` 成功 ready.
它覆盖 ordinary decode, DSpark5 verification, prefix reuse 和 graceful shutdown.
ordinary 和 DSpark5 request 都完成, 且没有 worker replacement.

2026 年 10 月 28 日, GPU `1,2,3,4` 上的 TEP4 smoke 在 28.013 秒内达到 Ready.
它使用 8 个 physical block 和 1024-token context, 并返回 4 个 ordinary output token.
四个 startup trace 的 metadata identity 都低于 0.1 秒, 且没有 shard hash phase.
rank-local weight load 约用时 19-20 秒, service 没有 worker replacement.

[DeepSeek matrix](../development/serving/deepseek_matrix.py) 中有批准的 workload definition.
HTTP runner 定义 36 个 DSpark5 case 和 2 个 ordinary TP4 case.
AgentX runner 定义 TP4/4, TEP4/16 和 DEP4/64 case.
当前还没有记录完整的 approved HTTP 或 AgentX result set.

## 下一步

- 完成真实 TP4 readiness 和 ordinary decode run.
- 用稳定的 request ownership 和 expert dispatch 替换 DEP4 fallback.
- 将 native DeepSeek path 与 reference path 和公开 baseline 做性能比较.
- 将 native mHC 和 sparse-attention temporary storage 移入 adapter-owned graph workspace.
- 添加 coordinated DeepSeek CUDA Graph capture 和 replay path.
- 在 GPU 0 以外的显卡上运行 38 个 HTTP case 和 3 个 AgentX case.
- 交付前取得 independent code review.

原始报告, host settings, model paths 和临时 process records 保留在 repository 外.
host settings 放在 `LOCAL.md` 中.
