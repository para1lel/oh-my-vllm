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
6. Python builds the batch plan and starts the target and selected draft GPU units.
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
| `worker/dspark.py`, `worker/dspark_graph.py` | DSpark context, conditional proposals, confidence limits, and different proposal graphs. |
| `worker/decode_graph.py`, `graph_cache.py` | Persistent inputs, transactional capture, bounded graph entries, and replay. |
| `worker/serving.py`, `sampling.py`, `sampler.py` | Tokenization, XGrammar, sampling, detokenization, and output accounting. |
| `models/qwen.py` | Checkpoint loading, 64-layer target, MTP layer, FP8/ordinary projections, and forward units. |
| `models/dspark.py` | Local BF16 draft checkpoint, five GQA layers, target-feature projection, and Markov/confidence heads. |
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
| `init` | Model paths, capacities, `block_size`, `tensor_parallel_size`, `max_model_len`, speculative mode/count, and DSpark confidence threshold. Allocate worker state. |
| `ready` | `logical_num_blocks`, `mamba_blocks`. Report device capacities. |
| `register` | `request_id`, `prompt_token_ids`. Register offline input. |
| `prepare` / `prepared` | `rpc_id`, `request_id`, `request` / prepared `prompt_token_ids`. Create service state. |
| `execute` | `rpc_id`, `step_id`, `scheduled`, `finished_request_ids`, `preempted_request_ids`, `num_batched_tokens`. Compute selected work. |
| Scheduled row | `request_id`, `token_ids`, `num_computed_tokens`, `prefill_token_ids`, `fa_block_table`, `mamba_block_table`. |
| `execute_result` | `rpc_id`, `outputs`. Return the output rows. |
| Output row | `request_id`, `token_ids`, `num_accepted_draft_tokens`, `new_draft_token_ids`. Optional `error`, `text`, `finish_reason`, `reasoning_tokens`. |
| `error` | `message`, optional `rpc_id` and `kind`. Distinguish validation and internal failure. |
| `abort` / `shutdown` | `request_id` for abort. Release request state / stop the worker. |
| `set_speculative_mode` / `mode_changed` | Correlated comparison RPC. Change MTP/DSpark mode only in an idle comparison worker. |

`prefill_token_ids` contains full accepted history on admission or recompute.
Ordinary decode stays incremental. Drafts have different storage.
Initialization errors omit correlation and kind. Execution errors default to internal kind.

Missing registration becomes a request-local execution error.
See the Rust and Python protocol definitions for wire names and defaults.

## Scheduler and allocation

The scheduler handles running requests before waiting requests with sequence and token budgets.
Aligned prefill materializes reusable GDN checkpoints at 784-token boundaries.
A 32768-token prompt splits into 32144 and 624 tokens.

After a successful prefill allocation of at least one block, a later rounded one-token prefill can wait for the next step.
This applies only when the available budget cannot reach the next checkpoint or the prefill end.
Decode-only steps, speculative verification, reachable checkpoints, and final tails keep forward progress.
Configured token budgets less than one block also keep progress.

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

Ordinary GDN state is FP32. MTP and DSpark GDN state is BF16.
The state layout is `[slot,48,128,128]`. Convolution state is BF16 `[slot,10240,3]`.
See [ADR-003](decisions/ADR-003-mtp-state-slots.md) and [ADR-007](decisions/ADR-007-ttft-and-cache-capacities.md).

## Target computation and attention

The loader validates head dimension 256, rotary dimension 64, theta 10000000, and RMS epsilon `1e-6`.
FP8 activations use per-row 128-value scales.
At most 32 FP8 projection rows use TRT-LLM GEMM from a third-party library.

Larger FP8 batches use the owned SM100 CUTLASS launcher.
It keeps arbitrary FP32 block scales, FP32 accumulation, and BF16 output.
The launcher selects PDL on the caller stream.

Wide gate/up matrices with at least 2048 rows use an eight-tile swizzle for cache locality.
At least 32144 rows use swizzle 16 for the two-SM, 34816 by 5120 gate/up matrix.
This path uses K tiles of 256 elements, three pipeline stages, and column-major activation and weight scales.
Scale groups keep 128 elements. The K-major layout reference keeps K tiles of 128 elements and five stages.
Other large projections keep K-major scales and automatic stage selection.

At 624 to 2496 rows, five one-SM projection shapes use a CTA cluster with two blocks along the output-column dimension.
The N/K pairs are 34816/5120, 16384/5120, 14336/5120, 5120/17408, and 5120/6144.
TMA multicast sends the same activation tile to the two blocks.
Each block computes different output columns. Scales, accumulation, rounding, PDL, and the caller stream keep their previous contracts.
Theoretical work keeps the same logical rows and parameter ranges.

Ordinary projections without scales use `F.linear`.


This avoids the observed FlashInfer CUTLASS instability at 17 through 32 rows.
Small BF16 vocabulary projections use FlashInfer CuTe-DSL GEMM.

Residual/RMS and SiLU/FP8 fusions keep the existing BF16 rounding points.
For more than 32 rows, the MLP gate/up path uses one CUDA kernel for residual RMS and activation quantization.
It writes the BF16 residual sum and keeps the normalized BF16 value in registers before FP8 conversion.
The projection keeps its previous GEMM provider and mathematical scale values.
At most 32 rows use the previous CUDA residual RMS and quantization chain.

Scaled GDN output uses fused gated RMS and FP8 quantization before its previous GEMM.
Each token has 48 heads of width 128. The kernel keeps the gated RMS BF16 rounding before FP8 conversion.

It accepts packed token strides and removes the intermediate BF16 array.
Column-major activation scales use one head row per warp.
K-major scales use two head rows per warp with less than 32144 token rows, and four with at least 32144.
PDL waits before input reads. The caller stream owns the launch and output lifetime.
Unscaled GDN output keeps BF16 normalization and its ordinary projection.

The large gate/up path rounds physical row capacity to a multiple of four for scale alignment.
Its fused kernel writes logical rows directly into that storage.
Each call clears at most three tail rows and packs 43520 bytes of checkpoint scales.

Graph replay reads changed checkpoint scales. The output view keeps only logical rows.
Padding and scale packing are implementation overhead. They do not increase the theoretical bound.

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

## DSpark

The optional mode loads the local DSpark checkpoint through project-owned code.
It validates configuration, tensor names, shapes, BF16 dtype, and stable source-file identities before execution.
The loaded configuration and weight hashes identify the bytes that supplied the tensors.
Checkpoint Python code does not execute.

The draft uses five GQA layers and zero-based target layer outputs `(5,19,33,47,61)`.
Each feature is the BF16 layer output before the subsequent normalization.
MTP4 and ordinary computation do not calculate these features.
The shared target embedding and vocabulary head supply DSpark token representations and base logits.
The draft projects concatenated target features into context KV for its five layers.

DSpark context pages use the same Rust-owned FA page IDs and 784-token layout, in a different physical cache.
Only committed target input rows enter this context.
Rejected input rows can stay in target FA after the committed cursor and are overwritten before subsequent use.
DSpark context receives only kept input rows.
Seven proposal rows use bidirectional draft attention and temporary KV.
They do not enter persistent context.

The Markov head uses the previous draft token to adjust each row's base logits.
The confidence head limits the candidate prefix through cumulative confidence.
The initial `--dspark-confidence-threshold` setting is `0.2`.
A first confidence product less than the threshold returns zero candidates and uses one target token for that step.
Set it to `0.0` to keep all valid rows up to seven for fixed-count experiments.
Output and context budgets can make that prefix shorter.

Draft block 7 and training block 16 are different from target block 784.

Verification with greedy sampling compares draft tokens with the target tokens that have maximum scores.
For stochastic output, the worker keeps each candidate's full conditional proposal distribution `q`.
Target distribution `p` and proposal distribution `q` use the configured temperature, penalties, top-k/top-p, and grammar masks.

Accept candidate `x` with probability `min(1,p(x)/q(x))`.
After rejection, sample from normalized `max(p-q,0)`. After full acceptance, sample the bonus token from `p`.
Draft sampling has a different random generator. Target sampling keeps its own generator.

The existing GDN snapshot/rollback logic keeps the state after committed input, with at most eight verification rows.
Speculative state is BF16. The target and draft keep specified BF16 rounding points.
Grammar simulation rolls back before committed output changes persistent grammar state.
Cancellation and preemption release target and draft request state.

Comparison workers load MTP and DSpark before timing, with one target and the same physical target caches.
The idle-only mode RPC changes the active draft count between four and seven.
The scheduler resets prefix metadata between attempts.
HTTP workers select one mode at initialization and reject this RPC.
See [ADR-009](decisions/ADR-009-dspark-optional-mode.md).

## Semantic IR and graphs

`torch.library` operations supply references, fake implementations, mutation schemas, and provider registrations.
Selection uses static metadata. It converts symbolic dimensions only when necessary for a selection predicate.
The pinned PyTorch lowering keeps semantic nodes until provider replacement.
Production provider failure raises an error. The worker uses debug reference/eager modes only after explicit selection.

Ordinary/MTP4 fullgraph units include prefill, target decode, MTP draft, and four-step proposal.
DSpark adds target features, context injection, backbone, Markov step, and greedy proposal units.
Host planning, logical allocation, ZMQ, sampling, and ordinary PyTorch operations are not in the operator inventory.
Manual CUDA Graphs contain compiled units. Inductor graphs are disabled.

Persistent cache writes keep ordering. Activation donation is currently absent.

The single-use BF16 SiLU-to-FP8 rewrite has equivalence tests.

Ordinary/MTP4 target graphs have 32 entries. DSpark target-feature graphs have 64 entries.
A comparison worker has 96 entries, with family floors of 16 for target and 32 for target-feature graphs.
The modes have different family keys and memory pools. The 4 GiB capture-headroom check stays active.

DSpark proposal graphs have a different 16-entry cache and pool.
DSpark context injection with at most 32 rows uses a different pool and a 32-entry cache.

Warmup and capture save destination KV slots and restore them in `finally`. Replay commits the current inputs.
Larger context injection uses the compiled unit without manual capture.
The default MTP draft/proposal budget is 64 entries, with floors of 32 and 8.
This budget applies to standalone workers and comparison workers.

Long persistent MTP context uses a different eight-entry cache and pool, with 12 GiB capture headroom.
Its selected endpoints use owned decode attention with first position 1.
Capture saves written FA slots and restores them after warmup and capture.
Replay validates all tensor shapes, types, and devices before copies.
Cache-only graphs reuse the same query shape across positions and page addresses.
Capture logs and `mtp_context` counters identify their activity.

Ordinary and DSpark prefill select final target outputs through `Batch.output_indices` and `Batch.output_attention`.
Layer 63 keeps all required K/V rows and computes attention and MLP for selected rows.
MTP keeps target hidden needed by its context and boundary features.
Feature tap 63 keeps complete layer output.
The semantic `prepare_context` and `prepare_query` operations define persistent writes and required queries.
Their CUDA implementations keep the full preparation kernel's BF16 rounding and FP64 phase arithmetic.

At a full cache budget, replacement admission must have four observations.
Its decayed count must be more than twice the coldest evictable entry.
An available budget lets capture start on the first miss.

Counts decay each 512 observations.
The cache skips new captures with less than 4 GiB of free GPU memory.
The worker still uses the compiled unit.
Capture failures back off for 64 observations.
If MTP captures the same shape again in a 4096-cache-decision window, a shared 32768-decision capture cooldown starts.

Resident graphs continue to replay.

Pools are shared in each graph family, with different target/draft/proposal families and no overlapping replay.
Capture restores persistent state and FA writes. Output copies precede pool reuse.

Target prefill has different eight-entry ordinary and DSpark families.
These families capture fixed query/context sizes with native or ragged attention and at least 32 GiB capture headroom.
KV pages, state slots, tokens, and attention metadata stay dynamic.
Warmup and capture restore all destination FA/GDN values in `finally`.
Replay validates all metadata replacements and restores captured bindings before copies.

FA 2 prefill uses the compiled path.
Dynamo limits are 4096 with warnings at 256.

These limits do not show bounded compiler memory with indefinite shape changes.
See [open audit work](audit.md).

## CUDA streams and PDL

The GDN preparation operation has two branches from the same normalized input.
The origin stream computes the FP8 QKV/Z projection and convolution.
One side stream computes the BF16 decay/beta projection and gates.
The side stream waits for the input and joins the origin on success or failure.
Stream records keep tensors alive until their consumers finish.
The branches do not share a writable FP8 workspace.

Owned pointwise CUDA launches use programmatic stream serialization.
The producer trigger lets a dependent grid start.
The consumer waits before dependent activation or state reads.
The owned CUTLASS launcher enables the same dependent-launch protocol.
PDL does not guarantee concurrent residency.
CUDA Graph capture records the fork/join and dependent launches.

The default allocator uses `graph_capture_record_stream_reuse:True` for joined graph frontiers.
An explicit `PYTORCH_ALLOC_CONF` takes precedence, then the legacy `PYTORCH_CUDA_ALLOC_CONF` value.
The wrapper normalizes the selected value to `PYTORCH_ALLOC_CONF`.
[Development](development.md) lists diagnostic switches.

## Phase model

`performance/` maps observed semantic work to official compute and HBM limits.
Rust records request submission, first-token, and last-token times and the corresponding steps.
The binary has a build stamp for Cargo inputs and Rust sources.
The collector validates this stamp, source identities, checkpoints, loaded libraries, and separate repetition intervals.
Unknown GPU work blocks acceptance.
See the [latency model](latency-model.md).

## Extension boundaries

New models must have validated loading, semantic contracts, and cache layouts.
Multiple GPUs must have new process ownership and collective contracts.
Other NVIDIA backends must have device-specific providers and measured acceptance.
Current interfaces make no such support claim.

DSpark runtime support applies to the validated local checkpoint architecture.
Different draft checkpoints must have new loading, semantic, numerical, and acceptance tests.
Use the [acceptance index](acceptance.md) for source identities and measured scope.
