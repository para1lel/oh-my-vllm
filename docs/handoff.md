# Work state

Work continues on `nightly`.

## Current service paths

Qwen keeps its single-worker path and shared Rust service, scheduler, logical cache, DEALER transport, and msgpack protocol.
`requirements/development.txt` includes `requirements/runtime.txt`.

Qwen decode and single-query verification use native operators.
[Sampling](../python/oh_my_vllm/models/qwen38/sampling.py#L28)<!-- source: self.full_history_view = self.full_history.numpy() --> packs masks and metadata with NumPy views.
[Uploads](../python/oh_my_vllm/operators/runtime/uploads.py#L207)<!-- source: self._native_key = key --> keep pinned snapshots and reuse native bindings.
Verification batches with at least two requests upload metadata once per group.
Other target and context inputs use separate uploads.

[Acceptance](../python/oh_my_vllm/models/qwen38/speculation.py#L232)<!-- source: self.metadata_inputs = Metadata(device, arrays, single_upload=True) --> packs gather indices, drafts, slots, and RNG into one upload.
Prepared [metadata writers](../python/oh_my_vllm/worker/metadata.py#L116)<!-- source: self.uploads.update(self._write, sources) --> update pinned snapshots after input validation and event completion.

Grammar-constrained MTP4 chains are disabled.
For at least two requests, constrained drafts can overlap GPU computation and CPU sampling preparation.
Unused computations restore KV rows.
Batch 1 keeps separate draft uploads and masks.
The [draft pipeline check](../development/operators/check_draft_pipeline.py) compares overlapping and sequential execution.

Drafts reuse repetition proofs and exact filtered masks.

Qwen uses [HTTP reference v5](models/qwen38/performance.md#lower-bound-calculation) and [operator reference v4](models/qwen38/operator-performance.md).
Next: long-output measurements again, the 24-case HTTP matrix, and AgentX acceptance.

The DeepSeek-V4.1-Flash adapter loads the original 48-shard checkpoint with rank-local placement.
Its four-rank Python bridge supports text targets and DSpark5 greedy verification.
It has `tp4`, `tep4`, and request-owned Eager `dep4` paths.

The adapter has model descriptors, cache groups, identity checks, and native operators.
Prefill uses working chunks of at most 8192 total rows by default, and `OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS` selects a value up to 32768.
Completed chunks release their hidden tail before the next native call.

DSpark verification sends the accepted prefix and draft rows to one target call.
Proposal queries share the full draft-index block, project only the required logits, and seed accepted target hidden states in one MTP operation.
Rejected drafts stay private and the target correction is emitted.

Two Engram buffers overlap lookup and GPU decoding.
Events protect storage reuse and failure recovery.
Lookup refreshes Graph buffers before replay.

Released shapes use native operators with bounded workspaces.
Prefill uses at most 32768 rows per launch.
Set a smaller chunk to use less temporary storage.

Native SwiGLU accepts up to 1048576 routed rows, which keeps six-route 8192-row prefill chunks on the native activation path.


DeepSeek uses the released FP4 and FP8 scale rules and nearest-even rounding.
Dense and grouped projections reset raw dots for each K32 group before scale multiplication and FP32 accumulation.
FP4 weights convert accurately to FP8 before the raw dot.
FP4 conversion keeps sign bits.
Supported [dense prefill shapes](models/deepseek_v41/gemm.md#dense-prefill-tiles) use TMA and TMEM with exact FP16 conversion and ordered K32 scale updates.
Other FP8 shapes keep the existing kernels.

FP8 decode can calculate K32 dots in parallel.
Scale accumulation keeps K32 order.

Native RMS keeps the released square-sum reduction and reciprocal multiplication order, and native routing keeps the released softplus and selected-score sum order.

In-place quantization rounds before dequantization, and the native quantizer writes encoded values and scales without context-sized temporaries.
Native attention keeps window and compressed KV in two tables and reads the model BF16 tables without quantization.
The reference path joins the tables before computation.

Shared experts use full replicated FP8 weights and scales.
Their output is added after routed rank reduction.
The shared output is not reduced across ranks.
Sinkhorn mixing adds epsilon before the first column normalization.

Native mHC keeps the released product, sum, and row-dependent mean order.
Its single-source-row FP32 projection uses 32 interleaved split dots.

Multi-row HC, router, and compressor projections select measured chunks for 2 through 32768 logical rows.
Unmeasured shapes or row counts stop the call.
Shared tiles keep per-output FMA and chunk-add order.
Single-chunk HC calls use asynchronous copies and two shared buffers.
Source chunks use row tiles and the 32 MiB workspace.
The model supplies this workspace by default.

An explicit empty projection workspace selects the per-thread reference path.
Prepared HC workspaces keep FP32 inputs, statistics, and 32 MiB partial storage for Eager and Graph execution.
Ratio-1 compressor BF16 projections use the released non-split-K cuBLAS configuration.
Startup checks the accepted BF16 projection pattern.

Nonzero target chunks keep source-step projection and reduction layouts.
HC and FP32 source positions share [HC kernel launches](../python/oh_my_vllm/operators/hyperconnection/cuda/hc_projection.cuh#L615)<!-- source: int steps = rows / logical_rows; --> and [FP32 projection launches](../python/oh_my_vllm/operators/hyperconnection/cuda/hyperconnection.cu#L193)<!-- source: int step_rows = logical_rows ? static_cast<int>(logical_rows) : rows; -->.
The logical row count continues to select each position's split, chunk, and statistics layout.
Their target attention and MoE execution stay batched.
Native Engram arithmetic uses the logical request batch for its per-copy reductions.

Sparse attention rounds unnormalized probabilities to BF16 in ordered 64-slot blocks before multiplication by V.
Its denominator uses the released cross-warp sum order and FMA.
BF16 tables use groups of four eight-value loads.
One warp calculates the same cross-warp sum tree, and the other warps read its results.

The sink enters the denominator after these blocks and does not change their running maximum.
BF16 attention with at most four rows computes 64-slot partials in parallel with caller-owned scratch.
Output channels run independently and keep the original running maximum, denominator, and numerator update order.

An independent development reference uses official MP4 placement and a recorded quantizer synchronization patch.
Frozen text fixtures hold full vocabulary vectors, DSpark5 proposal and verification vectors, and fresh accepted-only restoration endpoints.
Fixtures and numerical limits stay fixed during candidate acceptance.

Sparse attention status failures disable the native path.
The native call accumulates its status on the device, and the step boundary checks it once before the worker commits cache state.
A rejected step raises and does not fall back to the reference path.
Native sparse attention is the default, and long queries use it because the reference path scales poorly.
Set `OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE=0` to select reference attention for short queries.

DeepSeek token pages keep compressed KV, index keys, and Engram history across request slots.
Accepted-state pages keep target windows, MTP windows, and compressor tails at the accepted endpoint.
Cache operations support logical block store, load, drop, partial prefix copy, and
page-backed host snapshots.

Accepted DSpark target states seed the MTP window.
Private endpoint snapshots keep overwritten target windows and compressor tails during verification.
The state commit selects the accepted endpoint before it restores the active state.
Nonzero multi-token prefill uses the fused chunk path.
Target and MTP seed chunks keep the total native rows below the operator limit.
Aligned prefill batches use four-request subbatches by default and up to 16 requests with the 32768-row setting.

MoE dispatch gathers sorted routed rows once and reuses slices for local experts.
TP4 and TEP4 use native grouped FP4 up and down projections.
TP4 routed weights and scales use output-channel shards for all three projections.
TP4 gathers the BF16 activation product before complete-K32 down computation.
Each expert output rounds to BF16 before the local FP32 token sum.

For at least 128 token rows, the kernel reuses each token's sorted projection row indices across output columns.
TP4 gathers the FP32 token sums before adding the replicated shared output, and TEP4 keeps its routed rank reduction.

DeepSeek permits preemption and restart of one request in an Execute, but finished requests cannot also be scheduled or preempted there.

Grouped projection uses device offsets, compact nonempty experts, and persistent FP16 tiles with native K32 order.
At most 64 routed rows use parallel raw dots and ordered scales.

The TP4 device count path has no routed-row limit.
Grouped execution keeps token order.
Non-FP4 weights and test experts use the reference loop.

Routed messages have at most 1024 rows.
TP4 and TEP4 prefill use larger [projection waves](models/deepseek_v41/gemm.md#routed-prefill-waves).
Shared and routed gate/up projections share input quantization.
Shared experts process full prefill chunks.

[Fixed-capacity kernels](models/deepseek_v41/gemm.md#fixed-capacity-projections-and-sums) zero projection tails for SwiGLU and use device status.
TEP4 Graph and DEP4 caller integration is incomplete.

TP4, TEP4, and DEP4 keep [device expert pointers](../python/oh_my_vllm/models/deepseek_v41/model.py#L1846)<!-- source: pointers = state.pointers --> and FP4 bindings for each weight set.
Weight or expert rebinds, child `_apply`, and state loading clear them.

Native projections share compatible encoded weight storage.
Packed scales and padded weights use different storage.
Packed entries keep source owners and versions.
State loading clears shared entries.
DEP4 uses full dense weights, owner-local caches, full Engram reads, distinct token/destination expert exchange, and all-rank completion.

Non-final prefill skips the vocabulary projection.
DSpark keeps only the target hidden tail needed by its MTP window.

Full initial prompts can run as one aligned request batch, and shared query attention state resets at each target call.
Compressed KV sources publish selected slots after the current cache write.
Compressed KV and index copies contain only the active context range.

CUDA sequence staging uses demand-mapped storage for active request slots.
Slot restoration maps the required context before copying state.
Slot release waits for device work before unmapping storage.
Index scoring uses bounded query chunks and compact candidate-block masks.

The serving engine enforces the configured `max_num_seqs` resident limit.
It keeps one more request in preparation to overlap CPU admission with GPU work.

DeepSeek TP4 target decode uses captured CUDA graphs.
TEP4 and DEP4 read the routed counts on the host and use Eager execution.
A graph is captured for each cache-slot set, context bucket, and position residue.
The graph holds the collectives.
The Engram host lookup, sampling, and the token read use the host path.

After each replay, serving exports target windows and tails.
Graphs keep cached tensor inputs and reject changed expert bindings.
Uncaptured groups and steps beyond the graph cache limit use Eager.
Qwen Graphs keep bounded workspaces, stream handoff, graph upload, and cleanup.
Cleanup failures keep operations and input owners with the failed Graph.
DeepSeek prefill and draft verification use Eager execution.

Experimental batched targets keep request-local positions, recurrent endpoints, and per-request projection shapes.
`OH_MY_VLLM_DEEPSEEK_BATCH_DECODE=1` selects this path.
The default is off.

DEP4 records routes once per layer and call before expert exchange.
Hooks keep one route record for nested expert calls.
Accepted frontiers exclude repeated rows.
After all ranks finish Execute, one MAX joins expert presence and one packed read supplies counters.

TP4/TEP4 counters exclude replayed history.
The [trace contract](models/deepseek_v41/accounting.md#runtime-trace-contract) defines mixed-phase errors.
Batch checkpoint and HTTP trace validation are incomplete.

DeepSeek startup reads only `config.json` and `model.safetensors.index.json` for its descriptor identity and does not hash weight shards before Ready.

Rank-local checkpoint loading uses 65536-row tiles, 16-byte-aligned contiguous parameters, and bounded staging.
It copies required shard ranges and skips unused Engram payloads.
`OH_MY_VLLM_DEEPSEEK_LOAD_TILE_ROWS` selects the tile size.
`OH_MY_VLLM_STARTUP_TRACE` selects external startup traces.

DeepSeek uses the [v3.1 reference](models/deepseek_v41/accounting.md) for HTTP trace binding.
The harness binds `GPU-` UUIDs, worker identities, counters, and one configurable client/service timeout.
v2/v3 stay reproducible.
The full workloads are not accepted.
Next: run the [matrix](../development/serving/deepseek_matrix.py) and AgentX with final source identities.

## Next work

- Run the fixed Qwen operator cases in the [operator microbench design](models/qwen38/operators.md).
- Measure window storage at each state endpoint with the approved long-context workloads.
- Measure persistent encoded KV storage against direct BF16 table reads.
- Measure raw and prepared projection storage before long-context capacity tuning.
- Keep the independent frozen DeepSeek text endpoint checks before performance acceptance.
- Measure TP4 activation and output gathers and their temporary storage before long-context capacity tuning.
- Reduce duplicate cuBLAS operator workspace storage before long-context capacity tuning.
- Add independent image endpoints before image acceptance.
- Report the configured context limit and the observed AgentX request lengths independently.
- Tune AgentX physical block and temporary operator workspace capacity for 100K to 200K token traces at concurrency 16.
- Validate batched decode, per-request projection layouts, Graph endpoints, and cross-request verification with four-rank checkpoint inputs.
- Run higher-capacity AgentX cases after increasing physical cache and sequence capacity.
- Complete DEP4 checkpoint numerical, state, capacity, and HTTP acceptance.
- Measure the native DeepSeek paths against the reference path and public baselines.
- Move large-row native temporary storage into adapter-owned graph operator workspaces.
- Validate DEP4 HTTP manifests against owner trace records.
- Move the DeepSeek Engram lookup into the graph, or make its host copy asynchronous.
- Evaluate the replicated `wo_b` against the measured collective cost.
- Run the approved DeepSeek AgentX cases and publish the CSV and dark scatter plots.
- Run the 38 HTTP cases and the three AgentX cases on GPUs that have no locked clock.

Keep raw reports and temporary process records external to the repository.
Put host settings and model paths in `LOCAL.md`.
