# Work state

Work continues on `nightly`.

## Current service paths

Qwen keeps its existing single-worker service path.
The shared Rust service, scheduler, logical cache, DEALER transport, and msgpack protocol stay active.

DeepSeek-V4.1-Flash has a model adapter and a four-rank Python bridge.
The adapter loads the original 48-shard checkpoint with rank-local placement.
It supports text target execution and DSpark5 greedy proposal verification.
The bridge supports `tp4`, `tep4`, and a documented DEP4 executor fallback.

The adapter uses the DeepSeek descriptor, cache groups, tokenizer identity, inference-config identity, native projection, routing, activation, mHC, RMS, sparse attention, and transformer support.
Prefill uses working chunks of at most 8192 total rows.
Set `OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS` to use a smaller model-local chunk.
Completed chunks release their hidden tail before the next native call.

DSpark verification keeps rejected drafts private and emits the target correction.
Engram tables use file-backed host mappings.
Lookup stages only the selected rows on the GPU.

Released DeepSeek shapes support native projection, routing, activation, mHC, RMS, and sparse attention paths.
The native contracts accept long row ranges. Small rows use bounded workspaces.
Large prefill calls use bounded 8192-row launches to keep temporary storage bounded.

Sparse attention status failures disable the native path and use the reference path.
Short-query native sparse packing is opt-in through `OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE=1`.
Long queries use native sparse packing because the reference path scales poorly.

DeepSeek physical pages now keep window KV, compressed KV, index keys, Engram
hash state, and compressor tails across request slots.
Cache operations support logical block store, load, drop, partial prefix copy, and
page-backed host snapshots.

Accepted DSpark target states seed the MTP window and speculative rollback keeps
rejected draft state private.
Nonzero multi-token prefill uses the fused chunk path.
Target and MTP seed chunks keep the total native rows below the operator limit.
Aligned prefill batches use subbatches of at most four requests.

MoE dispatch gathers sorted routed rows once and reuses slices for local experts.
TP4 and TEP4 FP4 routed experts use a CUTLASS pointer-array grouped GEMM for
the two up projections and the down projection.
The grouped path keeps the local token order and falls back to the reference
expert loop for non-FP4 weights and test experts.
DEP4 still uses the TEP4 executor fallback and has no rank-owned dispatch.

Non-final prefill skips the vocabulary projection.
DSpark keeps only the target hidden tail needed by its MTP window.
Ordinary execution does not collect target hidden states.

Complete initial prompts can run as one aligned request batch.
Shared query attention state resets at each target call.
Index sources always publish the cache view for the current request slots.

The serving engine enforces the configured `max_num_seqs` resident limit.
It keeps one additional request in preparation to overlap CPU admission with GPU work.

DeepSeek execution is currently Eager.
The shared Qwen CUDA Graph path has stream handoff, bounded graph workspaces, graph upload, and cleanup tests.
DeepSeek multi-rank Graph capture is not connected.
An audit against current vLLM and SGLang graph patterns keeps DeepSeek Eager until
static multi-rank workspaces and capture buckets are defined.
The shared Qwen graph path already owns stream handoff, graph resources, and cleanup.

DeepSeek startup reads only `config.json` and `model.safetensors.index.json` for its descriptor identity.
It does not hash weight shards before Ready.
Rank-local checkpoint tiles use 65536 rows to reduce small host-to-device copies.
Set `OH_MY_VLLM_DEEPSEEK_LOAD_TILE_ROWS` to override this startup setting.
External startup traces use `OH_MY_VLLM_STARTUP_TRACE` and use a separate location from the repository.

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

The latest native-operator smoke reached ready on GPUs `1,2,3,4`.
It covered ordinary decode, DSpark5 verification, prefix reuse, and graceful shutdown.
The ordinary and DSpark5 requests completed without a worker replacement.

On October 28, 2026, a TEP4 smoke on GPUs `1,2,3,4` reached Ready in 31.012 seconds.
It used 8 physical blocks, a 1024-token context, and returned 4 ordinary output tokens.
All four startup traces had metadata identity below 0.1 seconds and no shard hash phase.
The rank-local weight load took about 19-20 seconds, and the service had zero replacements.

The approved DeepSeek workload definitions are available in [deepseek matrix](../development/serving/deepseek_matrix.py).
The HTTP runner defines 36 DSpark5 cases and 2 ordinary TP4 cases.
The AgentX runner defines TP4/4, TEP4/16, and DEP4/64 cases.
No complete approved HTTP or AgentX result set is recorded yet.
CodeGraph 1.6.2 is installed in Codex.
This repository has a local CodeGraph index and ignores `.codegraph/`.

## Next work

- Tune AgentX physical block and temporary workspace capacity for 100K to 200K token traces at concurrency 16.
- Replace the DEP4 fallback with stable request ownership and expert dispatch.
- Measure the native DeepSeek paths against the reference path and public baselines.
- Move large-row native temporary storage into adapter-owned graph workspaces.
- Extend grouped expert GEMM to the DEP4 ownership and communication path.
- Define static multi-rank workspaces before adding DeepSeek CUDA Graph capture and replay.
- Run the 38 HTTP cases and the three AgentX cases on GPUs apart from GPU 0.
- Get an independent code review before delivery.

Keep raw reports, host settings, model paths, and temporary process records in a separate location.
Use `LOCAL.md` for host settings.
