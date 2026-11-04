# Work state

Work continues on `nightly`.

## Current service paths

Qwen keeps its single-worker path and shared Rust service, scheduler, logical cache, DEALER transport, and msgpack protocol.
`requirements/development.txt` includes `requirements/runtime.txt`.

Qwen decode and single-query verification use native operators.
[Sampling](../python/oh_my_vllm/models/qwen38/sampling.py#L28)<!-- source: self.full_history_view = self.full_history.numpy() --> packs masks and metadata with NumPy views.
[Uploads](../python/oh_my_vllm/operators/runtime/uploads.py#L207)<!-- source: self._native_key = key --> keep pinned snapshots and reuse native bindings.
[Context inputs](../python/oh_my_vllm/models/qwen38/speculation.py#L482)<!-- source: self.context_prefill = Context( --> and verification batches use one upload, except single-request verification and single-request staged decode.
Other target inputs upload separately.

[Acceptance](../python/oh_my_vllm/models/qwen38/speculation.py#L245)<!-- source: self.metadata_inputs = Metadata(device, arrays, single_upload=True) --> packs gather indices, drafts, slots, and RNG into one upload.
Prepared [metadata writers](../python/oh_my_vllm/worker/metadata.py#L116)<!-- source: self.uploads.update(self._write, sources) --> update pinned snapshots after input validation and event completion.

Grammar-constrained MTP4 chains are disabled.
For at least two requests, constrained drafts can overlap GPU computation and CPU sampling preparation.
Unused computations restore KV rows.
Batch 1 keeps separate draft uploads and masks.
The [draft pipeline check](../development/operators/check_draft_pipeline.py) compares overlapping and sequential execution.

Drafts reuse repetition proofs and exact filtered masks.

Qwen uses [HTTP reference v5](models/qwen38/performance.md#lower-bound-calculation) and [operator reference v4](models/qwen38/operator-performance.md).
The Qwen [HTTP and AgentX reports](models/qwen38/results.md) are completed.

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


DeepSeek uses released FP4 and FP8 scales with nearest-even rounding.
Dense and grouped projections reset each K32 dot before scaling and FP32 accumulation.
FP4 weights convert exactly to FP8 with sign bits.
Routed FP4 prefill converts [four values](../python/oh_my_vllm/operators/projection/cuda/half_k32.cuh#L29)<!-- source: __device__ __forceinline__ uint2 fp4_half_four(uint16_t packed) { --> together and reuses [scales](../python/oh_my_vllm/operators/projection/cuda/half_k32.cuh#L162)<!-- source: float activation_scales[2][2], weight_scales_tile[2][2]; -->.
Supported [dense prefill shapes](models/deepseek_v41/gemm.md#dense-prefill-tiles) use TMA and TMEM with exact FP16 conversion and ordered K32 scale updates.
Other FP8 shapes keep existing kernels.

FP8 decode can calculate K32 dots in parallel.
Scale accumulation keeps K32 order.

Native RMS keeps the released square-sum reduction and reciprocal multiplication order, and native routing keeps the released softplus and selected-score sum order.

In-place quantization rounds before dequantization, and the native quantizer writes encoded values and scales without context-sized temporaries.
Native attention reads BF16 window values and encoded compressed history from separate tables.
The reference path decodes compressed history before joining the tables.

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

Token pages keep encoded compressed KV, index keys, and Engram history.
Active slots use page-table views.
Host transfers keep encoded values and scale bytes.
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

Grouped projection uses device offsets, nonempty expert compaction, persistent FP16 tiles, native K32 order, and token order.
Grouped FP4 BF16 [grids](../python/oh_my_vllm/operators/projection/cuda/half_k32.cuh#L226)<!-- source: constexpr int blocks_per_sm = --> have at most 1036 blocks.
At most 64 routed rows use parallel raw dots and ordered scales.

TP4 device counts have no routed-row limit.
Non-FP4 weights and test experts use reference loops.

Routed messages have at most 1024 rows.
TP4 and TEP4 prefill use larger [projection waves](models/deepseek_v41/gemm.md#routed-prefill-waves).
Shared and routed gate/up projections share input quantization.
Shared experts process full prefill chunks.

[Fixed-capacity kernels](models/deepseek_v41/gemm.md#fixed-capacity-projections-and-sums) zero projection tails for SwiGLU and use device status.
TEP4 Graph and DEP4 caller integration is incomplete.

TP4, TEP4, and DEP4 keep [device expert pointers](../python/oh_my_vllm/models/deepseek_v41/model.py#L1840)<!-- source: pointers = state.pointers --> and FP4 bindings for each weight set.
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

Index scoring uses bounded query chunks and compact candidate-block masks.
Index-key workspaces decode the selected request context.
Graph leases keep physical pages alive.

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
DSpark verification renews proposals and restores accepted state.
Same-position Eager groups use scalar positions.
Mixed-position groups and Graph replay use device positions.
`OH_MY_VLLM_DEEPSEEK_BATCH_DECODE=1` selects this path.
The default is off.

DEP4 records routes once per layer and call before expert exchange.
Hooks keep one route record for nested expert calls.
Accepted frontiers exclude repeated rows.
After all ranks finish Execute, one MAX joins expert presence and one packed read supplies counters.

TP4/TEP4 counters exclude replayed history.
The [trace contract](models/deepseek_v41/accounting.md#runtime-trace-contract) defines mixed-phase errors.
[Packed page checks](../development/model/check_deepseek_batch.py#L19)<!-- source: return cpu(vars(value), torch) --> [close Graph leases](../development/model/check_deepseek_batch.py#L297)<!-- source: graph.close() -->.
Full B64 checkpoint and HTTP trace validation are incomplete.

GPU acceptance of the DEP4 candidate is not completed.

DeepSeek startup reads only `config.json` and `model.safetensors.index.json` for its descriptor identity and does not hash weight shards before Ready.

Rank-local checkpoint loading uses 65536-row tiles, 16-byte-aligned contiguous parameters, and bounded staging.
It copies required shard ranges and skips unused Engram payloads.
`OH_MY_VLLM_DEEPSEEK_LOAD_TILE_ROWS` selects the tile size.
`OH_MY_VLLM_STARTUP_TRACE` selects external startup traces.

DeepSeek keeps [v3.1](models/deepseek_v41/accounting.md) as the HTTP default and source trace contract.
[v4](models/deepseek_v41/performance.md) keeps a conditional partial floor and separate identities.
[v5](models/deepseek_v41/performance.md#packed-project-reference-and-19-row-check) counts packed history and checks 19 long-output rows.
v2/v3 stay reproducible.
Full workloads are not accepted.
Run [the matrix](../development/serving/deepseek_matrix.py) and AgentX with the last source identities.

## Next work

- Run fixed Qwen operator cases from the [microbench design](models/qwen38/operators.md).
- Measure window storage at each state endpoint with the approved long-context workloads.
- Validate [encoded KV storage](research/deepinfer-precision.md) with the model workloads.
- Measure projection and TP4 gather storage before long-context capacity tuning.
- Keep independent frozen DeepSeek text endpoint checks before performance acceptance.
- Reduce duplicate cuBLAS workspace storage before capacity tuning.
- Add independent image endpoints before image acceptance.
- Write configured context limits and observed AgentX request lengths independently.
- Tune AgentX physical block and temporary operator workspace capacity for 100K to 200K token traces at concurrency 16.
- Validate batched decode, per-request projection layouts, Graph endpoints, and cross-request verification with four-rank checkpoint inputs.
- Run higher-capacity AgentX cases after increasing physical cache and sequence capacity.
- Complete DEP4 checkpoint numerical, state, capacity, and HTTP acceptance with current native identities.
- Measure the native DeepSeek paths against the reference path and public baselines.
- Move large-row native temporary storage into adapter-owned graph operator workspaces.
- Validate DEP4 HTTP manifests against owner traces with 64 concurrent requests.
- Move the DeepSeek Engram lookup into the graph, or make its host copy asynchronous.
- Evaluate the replicated `wo_b` against the measured collective cost.
- Run approved DeepSeek AgentX cases. Publish CSV and dark scatter plots.
- Run 38 HTTP and 3 AgentX cases without locked GPU clocks.

Keep raw reports and temporary process records external to the repository.
Put host settings and model paths in `LOCAL.md`.
