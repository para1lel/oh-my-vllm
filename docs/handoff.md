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
Engram tables use file-backed host mappings.
Lookup stages only the selected rows on the GPU.

DeepSeek physical pages now keep window KV, compressed KV, index keys, Engram
hash state, and compressor tails across request slots.
Cache operations support logical block store, load, drop, partial prefix copy, and
page-backed host snapshots.
Accepted DSpark target states seed the MTP window and speculative rollback keeps
rejected draft state private.
Nonzero multi-token prefill uses a sequential correctness path while a fused
chunk path stays a future performance task.

DeepSeek execution is currently Eager.
The shared Qwen CUDA Graph path has stream handoff, bounded graph workspaces, and cleanup tests.
DeepSeek multi-rank Graph capture is not connected.

## Current evidence

Short checkpoint execution has completed on GPUs `4,5,6,7` with TEP4.
It covers readiness, ordinary decode, DSpark proposal output, and draft rejection.
CPU and Rust checks cover the descriptor, model, operators, launcher cleanup, matrix scope, speculation, and shared runtime.

On October 28, 2026, the full Rust HTTP service reached ready with TEP4 and a short context.
The smoke run checked model listing, diagnostics, ordinary text, DSpark5, SSE usage, validation errors, and shutdown.
The DSpark5 HTTP log reported 2 proposed and 2 verified draft tokens.
This smoke run is functional evidence and does not replace the approved matrix.

After the physical cache change, an HTTP service reached ready on GPUs `1,2,3,4`
with TEP4, 8 blocks, and a 1024-token physical context.
Two ordinary requests returned successfully.
The second request reused 13 prompt tokens from the physical prefix cache.
A DSpark5 target request also returned successfully without worker restart.
The 2-token DSpark5 request had no proposal budget and therefore reported zero proposal counters.
The cache behavior test covers host snapshot store, load, drop, prefix copy, and restoration.

The approved DeepSeek workload definitions are available in [deepseek matrix](../development/serving/deepseek_matrix.py).
The HTTP runner defines 36 DSpark5 cases and 2 ordinary TP4 cases.
The AgentX runner defines TP4/4, TEP4/16, and DEP4/64 cases.
No complete approved HTTP or AgentX result set is recorded yet.

## Next work

- Complete a TP4 readiness and ordinary decode run.
- Replace the DEP4 fallback with stable request ownership and expert dispatch.
- Connect native sparse attention, hyper-connections, and transform workspaces.
- Add a coordinated DeepSeek CUDA Graph capture and replay path.
- Run the 38 HTTP cases and the three AgentX cases on GPUs apart from GPU 0.
- Get an independent code review before delivery.

Keep raw reports, host settings, model paths, and temporary process records in a separate location.
Use `LOCAL.md` for host settings.
