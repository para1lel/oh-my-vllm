# Architecture

Rust controls request lifetime and logical memory.
Python controls GPU tensors and computation.
The [requirements](requirements.md) define accepted behavior. [decisions](README.md#decision-records) record key alternatives.

## Request flow

1. Rust validates the HTTP request and reserves admission capacity.
2. Python applies the tokenizer/template and prepares sampling and grammar state in a background thread.
3. The Python main thread installs the prepared state.
4. Rust admits the prompt, finds a consistent prefix, and schedules token work.
5. Rust allocates logical FA/GDN slots and sends execution metadata.
6. Python builds the batch plan and starts the target or MTP GPU units.
7. Python returns kept tokens, next drafts, and optional service output.
8. Rust validates the full reply before state changes.
9. Rust commits accepted history, completes or reschedules the request, and supplies output.

Cancellation takes effect between completed engine operations.
Finished-only execution flushes pending Python cleanup, even when no work stays.
Request-local failures stop their request. Worker/device failures stop the engine.
The Rust child owner and Linux parent-death signal prevent abandoned Python workers.

## Rust modules

| Module | Responsibility |
|---|---|
| `crates/kv-cache/src/block.rs`, `pool.rs`, `free_queue.rs` | Block references, cache metadata, and LRU free queue. |
| `crates/kv-cache/src/hash.rs`, `group.rs`, `coordinator.rs` | Prefix hashes, group layouts, and coordinated allocation. |
| `crates/scheduler/src/request.rs`, `output.rs`, `lib.rs` | Accepted history, draft scheduling, validation, and queue transitions. |
| `crates/zmq-worker/src/client.rs`, `protocol.rs`, `error.rs` | Child lifetime, correlated RPCs, typed messages, and error boundaries. |
| `crates/zmq-worker/src/main.rs` | CLI, offline driver, benchmark timing, and runtime configuration. |
| `crates/zmq-worker/src/serving/` | Axum HTTP, typed events, tool parsing, response storage, and online admission. |

## Python modules

| Module | Responsibility |
|---|---|
| `worker/zmq_bridge.py`, `protocol.py` | Message handling, background preparation, main-thread installation, and cleanup. |
| `worker/model_runner.py`, `runtime.py`, `logging_utils.py` | Independent initialization, source/library identity, execution, and diagnostics. |
| `worker/batch_plan.py`, `tensors.py` | Validated CPU plans, slot selection, batched metadata transfer, and tensor views. |
| `worker/mtp.py` | Proposals, verification, accepted-state retention, and checkpoint writes. |
| `worker/decode_graph.py`, `graph_cache.py` | Persistent inputs, transactional capture, bounded graph entries, and replay. |
| `worker/serving.py`, `sampling.py`, `sampler.py` | Tokenization, XGrammar, sampling, detokenization, and output accounting. |
| `models/qwen.py` | Checkpoint loading, 64-layer target, MTP layer, FP8/ordinary projections, and forward units. |
| `ir/` | Semantic operations, references, mutation schemas, provider selection, lowering, and coverage. |
| `kernels/` | Attention, GDN, convolution, normalization, elementwise operations, and backend choice. |
| `kernels/cuda_backend/` | B200 CUDA implementation and loaded-module provenance. |
| `kernels/tilelang_reference/` | Pinned comparison implementation. |

All Python module paths are relative to `python/oh_my_vllm/`.

## Transport contract

Rust binds `ipc://<socket_path>`. The single Python worker connects.
Messages are msgpack dictionaries with a `type` field.
Prepare and execute replies echo `rpc_id`. Late cancelled prepare replies are discarded.

`register` and `abort` have no reply.

| Message | Fields and effect |
|---|---|
| `init` | `model_path`, `num_gpu_blocks`, `mamba_blocks`, `block_size`, `tensor_parallel_size`, `max_model_len`, `num_speculative_tokens`. Allocate worker state. |
| `ready` | `logical_num_blocks`, `mamba_blocks`. Report device capacities. |
| `register` | `request_id`, `prompt_token_ids`. Register offline input. |
| `prepare` / `prepared` | `rpc_id`, `request_id`, `request` / prepared `prompt_token_ids`. Create service state. |
| `execute` | `rpc_id`, `step_id`, `scheduled`, `finished_request_ids`, `preempted_request_ids`, `num_batched_tokens`. Compute selected work. |
| Scheduled row | `request_id`, `token_ids`, `num_computed_tokens`, `prefill_token_ids`, `fa_block_table`, `mamba_block_table`. |
| `execute_result` | `rpc_id`, `outputs`. Return the output rows. |
| Output row | `request_id`, `token_ids`, `num_accepted_draft_tokens`, `new_draft_token_ids`. Optional `error`, `text`, `finish_reason`, `reasoning_tokens`. |
| `error` | `message`, optional `rpc_id` and `kind`. Distinguish validation and internal failure. |
| `abort` / `shutdown` | `request_id` for abort. Release request state / stop the worker. |

`prefill_token_ids` contains full accepted history on admission or recompute.
Ordinary decode stays incremental. Drafts have different storage.
Initialization errors omit correlation and kind. Execution errors default to internal kind.

Missing registration becomes a request-local execution error.
See the Rust and Python protocol definitions for wire names and defaults.

## Scheduler and allocation

The scheduler handles running requests before waiting requests with sequence and token budgets.
Aligned prefill materializes reusable GDN checkpoints at 784-token boundaries.
A 32768-token prompt splits into 32144 and 624 tokens.
The scheduler rejects requests that cannot fit their required logical capacity.

Allocation first removes obsolete aligned state and counts required capacity with read-only pool access.
It includes the watermark in the capacity test.
It touches all reused blocks before any group allocates new blocks.
It then allocates new blocks and registers full cache blocks.
This order prevents double-counting reused blocks in the free queue.

On capacity failure, Rust preempts later-admitted running requests from the queue tail and retries earlier requests.
If no later victim stays, it preempts the current request.
Victims return to the waiting queue in admission order.
Recompute resets the computed cursor, keeps accepted history, and clears drafts.
Output validation checks IDs, scheduled counts, token bounds, accepted drafts, and errors before a transactional commit.

## Cache groups and prefix reuse

The FA group holds sixteen layers of 784-token pages.
GDN alignment holds running/checkpoint state, a protected previous state, and speculative state slots.
Null placeholders keep logical positions.
Unused cached checkpoints can stay until eviction.

By default, groups share one logical pool.
`--mamba-blocks N` gives GDN a different pool with group-local IDs.
FA capacity is `floor(num_gpu_blocks/3)`. Block IDs directly index tensor slots.
The former physical-stride mapping is retired.

GDN slot zero is read-only initial zero state and never a write destination.
FA page zero is the null page.

The conceptual prefix hash is `SHA-256(parent_hash || token_ids || extra_keys)[0..8]`.
The byte encoding includes a parent-presence marker and token/extra-key lengths.
Integers and the first eight digest bytes use little-endian encoding.
Each hash depends on preceding content. Resident blocks still must have a contiguous FA prefix.

The coordinator reconciles that prefix with an available GDN checkpoint.
The result is the minimum compatible FA/GDN hit length.
References protect live blocks. The free queue evicts least-recently-used cache entries.

Ordinary GDN state is FP32. MTP GDN state is BF16.
The state layout is `[slot,48,128,128]`. Convolution state is BF16 `[slot,10240,3]`.
See [ADR-003](decisions/ADR-003-mtp-state-slots.md) and [ADR-007](decisions/ADR-007-ttft-and-cache-capacities.md).

## Target computation and attention

The loader validates head dimension 256, rotary dimension 64, theta 10000000, and RMS epsilon `1e-6`.
FP8 activations use per-row 128-value scales.
At most 32 FP8 projection rows use TRT-LLM GEMM from a third-party library. Larger FP8 batches use CUTLASS.
Ordinary projections without scales use `F.linear`.

This avoids the observed FlashInfer CUTLASS instability at 17 through 32 rows.
Small BF16 vocabulary projections use FlashInfer CuTe-DSL GEMM.

Residual/RMS and SiLU/FP8 fusions keep the existing BF16 rounding points.
Convolution, GDN, and RMS accept packed projection strides and return dense outputs.
GDN prefill normalizes Q/K explicitly with FP32 norms and BF16 output.
Integer metadata transfers use one buffer per target/draft group.
Device views keep their backing storage.

The CUDA attention preparation combines Q/K RMS, partial NeoX RoPE, V conversion, and KV writes.
It accepts packed 14336-value projections and returns contiguous Q.
Each 512-wide Q head keeps its gate half unchanged.
FP64 phase calculation precedes FP32 trigonometry.
Negative slots skip KV writes and keep Q. Caches cannot alias inputs.

Target query spans of at least 1024 gather active KV for ragged TRT-LLM attention.
Spans from 128 through 1023 with at least 4096 KV tokens use native paged context attention.
Other small spans use paged FA 2.
The routing thresholds use the batch maximum query and KV lengths.
Mixed batches share the selected route.

The longest-context memory checks include temporary gathered KV.

Target `DecodeGraph` uses native decode and views each 784-token page as 49 subpages of 16 tokens.
Short target decode without a selected graph uses the paged FA 2 route.
K/V offset views and `page * 98 + subpage` tables avoid KV copies.
Table expansion and lengths use current execution metadata inside graph capture.
Draft prefill starts at position one. Spans of at least 128 use ragged attention.

Smaller draft spans use FA 2. Draft decode uses the owned kernel.

## MTP

Draft row `p` combines target hidden row `p-1` and token row `p`.
Row zero is absent.
The uniform RoPE offset keeps relative positions.
Rust allocates draft/state capacity. Python never invents logical blocks.

A 784-token boundary hidden state waits if its required page is unavailable.

MTP4 proposes four drafts and verifies them against target sampling.
Grouped verification has at most five queries with per-query causal lengths.
The single-token decode path has its own implementation.
The worker keeps accepted target output and the correct GDN/convolution snapshot.
Rejected scheduled drafts roll back computed progress. Unscheduled drafts have no computed progress to roll back.

Grammar simulation uses each speculative prefix and rolls back before it commits kept output.

## Semantic IR and graphs

`torch.library` operations supply references, fake implementations, mutation schemas, and provider registrations.
Selection uses static metadata. It converts symbolic dimensions only when necessary for a selection predicate.
The pinned PyTorch lowering keeps semantic nodes until provider replacement.
Production provider failure raises an error. The worker uses debug reference/eager modes only after explicit selection.

Four fullgraph units include prefill, target decode, MTP draft, and four-step proposal.
Host planning, logical allocation, ZMQ, sampling, and ordinary PyTorch operations are not in the operator inventory.
Manual CUDA Graphs contain compiled units. Inductor graphs are disabled.
Persistent cache writes keep ordering. Activation donation is currently absent.

The single-use BF16 SiLU-to-FP8 rewrite has equivalence tests.

Target graphs have a 32-entry budget.
Draft/proposal graphs share 32 entries with floors of 16 and 4.
At a full cache budget, replacement admission must have four observations.
Its decayed count must be more than twice the coldest evictable entry.
An available budget permits capture on the first miss.

Counts decay each 512 observations.
The cache skips new captures with less than 4 GiB of free GPU memory.
The worker still uses the compiled unit.
Capture failures back off for 64 observations.
A repeated MTP shape capture in a 4096-cache-decision window starts a shared 32768-decision capture cooldown.

Resident graphs continue to replay.

Pools are shared in each graph family, with different target/draft/proposal families and no overlapping replay.
Capture restores persistent state and FA writes. Output copies precede pool reuse.
Prefill uses compiled units without manual graph capture.
Dynamo limits are 4096 with warnings at 256.

These limits do not show bounded compiler memory with indefinite shape changes.
See [open audit work](audit.md).

## Extension boundaries

New models must have validated loading, semantic contracts, and cache layouts.
Multiple GPUs must have new process ownership and collective contracts.
Other NVIDIA backends must have device-specific providers and measured acceptance.
Current interfaces make no such support claim.

The DSpark design uses five BF16 GQA layers and target features `[5,19,33,47,61]`.
It has confidence/Markov heads, seven drafts, and eight-token verification.
Draft block 7 and training block 16 are different from target block 784.
This is extension documentation. No DSpark runtime is supplied.
