# Work state

Work continues on `nightly`.

## Current service paths

Qwen keeps its existing single-worker service path.
The shared Rust service, scheduler, logical cache, DEALER transport, and msgpack protocol stay active.

DeepSeek-V4.1-Flash has a model adapter and a four-rank Python bridge.
The adapter loads the original 48-shard checkpoint with rank-local placement.
It supports text target execution and DSpark5 greedy proposal verification.
The bridge supports `tp4`, `tep4`, and a documented DEP4 executor fallback.

The adapter uses the DeepSeek descriptor, cache groups, tokenizer identity, inference-config identity, native projection, routing, activation, and transformer support.
Prefill rows are split at the native operator row limit.
DSpark verification keeps rejected drafts private and emits the target correction.

DeepSeek execution is currently Eager.
The shared Qwen CUDA Graph path has stream handoff, bounded graph workspaces, and cleanup tests.
DeepSeek multi-rank Graph capture is not connected.

## Current evidence

Short checkpoint execution has completed on GPUs `4,5,6,7` with TEP4.
It covers readiness, ordinary decode, DSpark proposal output, and draft rejection.
CPU and Rust checks cover the descriptor, model, operators, launcher cleanup, matrix scope, speculation, and shared runtime.

The approved DeepSeek workload definitions are available in [deepseek matrix](../development/serving/deepseek_matrix.py).
The HTTP runner defines 36 DSpark5 cases and 2 ordinary TP4 cases.
The AgentX runner defines TP4/4, TEP4/16, and DEP4/64 cases.
No complete approved HTTP or AgentX result set is recorded yet.

## Next work

- Complete a TP4 readiness and ordinary decode run.
- Implement a physical DeepSeek cache contract for window, compressed KV, index keys, accepted state, prefix reuse, and restoration.
- Replace the DEP4 fallback with stable request ownership and expert dispatch.
- Connect native sparse attention, hyper-connections, and transform workspaces.
- Add a coordinated DeepSeek CUDA Graph capture and replay path.
- Run the 38 HTTP cases and the three AgentX cases on GPUs apart from GPU 0.
- Get an independent code review before delivery.

Keep raw reports, host settings, model paths, and temporary process records in a separate location.
Use `LOCAL.md` for host settings.
