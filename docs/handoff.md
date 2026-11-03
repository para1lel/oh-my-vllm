# Work state

Work continues on `nightly`.

## Current service paths

Qwen keeps its existing single-worker service path.
The shared Rust service, scheduler, logical cache, DEALER transport, and msgpack protocol stay active.
The runtime packages use `requirements/runtime.txt`, which `requirements/development.txt` includes for one development installation.

Qwen decode and single-query verification use the existing native operators.
The [sampling host input path](../python/oh_my_vllm/models/qwen38/sampling.py#L28)<!-- source: self.full_history_view = self.full_history.numpy() --> packs masks with NumPy.
It writes the host inputs through NumPy views of the CPU Torch tensors.
Qwen verification uses one metadata upload per group when the batch has at least two requests.
Batch 1 and prefill target and context inputs keep separate uploads.

The grammar-constrained MTP4 draft chain is disabled.
For at least two requests, ordinary draft stages reuse fixed row views and packed sampling uploads.
The CPU grammar path keeps packed masks through the Qwen sampling rules.
Batch 1 keeps the separate upload and mask paths.
Cached row views keep their source tensors and rebuild after a rebind.
Next work: diagnose prefill time with the complete HTTP workload.

DeepSeek-V4.1-Flash has a model adapter and a four-rank Python bridge.
The adapter loads the original 48-shard checkpoint with rank-local placement.
It supports text target execution and DSpark5 greedy proposal verification.
The bridge supports `tp4`, `tep4`, and a documented DEP4 fallback.

The adapter uses the DeepSeek descriptor, cache groups, tokenizer and inference-config identity, native projection, routing, activation, mHC, RMS, sparse attention, and transformer support.
Prefill uses working chunks of at most 8192 total rows by default, and `OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS` selects a value up to 32768.
Completed chunks release their hidden tail before the next native call.

DSpark verification sends the accepted prefix and draft rows to one target call.
Proposal queries share the full draft-index block, project only the required logits, and seed accepted target hidden states in one MTP operation.
Rejected drafts stay private and the target correction is emitted.
Engram tables use file-backed host mappings.
Lookup stages only the selected rows on the GPU.

Released DeepSeek shapes support native projection, routing, activation, mHC, RMS, and sparse attention paths.
The native contracts accept long row ranges, and small rows use bounded workspaces.
The default prefill launch size cuts long-context launch count, and the environment override can lower temporary storage.

Native SwiGLU accepts up to 1048576 routed rows, which keeps six-route 8192-row prefill chunks on the native activation path.


DeepSeek uses the released FP4 and FP8 scale rules and nearest-even rounding.
Dense and grouped projections reset raw dots for each K32 group before scale multiplication and FP32 accumulation.
FP4 weights convert accurately to FP8 before the raw dot.
The FP4 conversion uses a register lookup table and keeps each sign bit.
The FP8 path reuses each weight tile for two 16-row input tiles when the input has at least 32 rows.
The reuse applies when the output width is above 1024 or the input width is below 1024.

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

Native mHC keeps the released product, sum, and row-count-dependent mean order, and its FP32 projection writes the released reduction order in native kernels.
A single source row uses 32 interleaved split dots.

A multi-row source step selects its split chunk from the measured released table.
The table covers row counts from 2 through 32768 and larger counts stop the call.
An explicit 32 MiB operator workspace holds the split and chunk partials.
A chunk-parallel kernel keeps the same arithmetic and removes the repeated weight reads.
An empty projection workspace selects the sequential reference order.

The router and compressor FP32 projections use their own measured chunk tables, and unmeasured shapes or row counts stop the call.
Ratio-1 compressor BF16 projections use the released non-split-K cuBLAS configuration.
Startup checks the accepted BF16 projection pattern.

Prepared HC operator workspaces keep FP32 input and statistics storage for Eager or Graph execution.
Nonzero target chunks keep source-step projection and reduction layouts.
HC and FP32 source positions share [HC kernel launches](../python/oh_my_vllm/operators/hyperconnection/cuda/hc_projection.cuh#L479)<!-- source: int steps = rows / logical_rows; --> and [FP32 projection launches](../python/oh_my_vllm/operators/hyperconnection/cuda/hyperconnection.cu#L194)<!-- source: int step_rows = logical_rows ? static_cast<int>(logical_rows) : rows; -->.
The logical row count continues to select each position's split, chunk, and statistics layout.
Their target attention and MoE execution stay batched.
Native Engram arithmetic uses the logical request batch for its per-copy reductions.

Sparse attention rounds unnormalized probabilities to BF16 in ordered 64-slot blocks before multiplication by V.
Its denominator uses the released cross-warp sum order and FMA.
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
TP4 and TEP4 FP4 routed experts use native grouped projection for
the two up projections and the down projection.
TP4 routed weights and scales use output-channel shards for all three projections.
TP4 gathers the BF16 activation product before complete-K32 down computation.
Each expert output rounds to BF16 before the local FP32 token sum.
TP4 gathers the FP32 token sums before adding the replicated shared output, and TEP4 keeps its routed rank reduction.

DeepSeek permits preemption and restart of one request in an Execute, but finished requests cannot also be scheduled or preempted there.

Grouped projection calculates row offsets on the device and does not upload host descriptors.
The grouped operator compacts the nonempty expert groups in its device scratch.
Each block walks its group rows, so the launch has no host row bound.

The TP4 device count path has no routed-row limit.
The grouped path keeps the local token order and falls back to the reference
expert loop for non-FP4 weights and test experts.
The MoE processes at most 1024 token rows per routed expert batch.
MoE prepares per-expert device pointer arrays and local FP4 expert bindings once per routed weight set.
Weight or expert rebinds, child `_apply`, and state loading clear them.

Native projections share compatible encoded weight storage.
Packed scales and padded weights use different storage.
Prepared cache entries keep their original weight and scale owners.
DEP4 still uses the TEP4 fallback and has no rank-owned dispatch.

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

DeepSeek target decode uses captured CUDA graphs.
The graphs apply to TP4 only.
TEP4 and DEP4 read the routed counts on the host and use Eager execution.
A graph is captured for each cache-slot set, context bucket, and position residue.
The graph holds the collectives.
The Engram host lookup, sampling, and the token read use the host path.

The serving path exports the target window and tail state after each replay, and a graph keeps its captured step buffers for its lifetime.
A step that is not in a captured group or that exceeds the graph cache limit uses Eager.
The shared Qwen CUDA Graph path keeps its stream handoff, bounded operator workspaces, graph upload, and cleanup.
DeepSeek prefill and draft verification use Eager execution.

DeepSeek startup reads only `config.json` and `model.safetensors.index.json` for its descriptor identity and does not hash weight shards before Ready.

Rank-local checkpoint tiles use 65536 rows, and parameters use 16-byte-aligned contiguous storage.
GPU startup copies required tensor ranges from each shard through a bounded staging buffer and skips unused Engram payload ranges.
Set `OH_MY_VLLM_DEEPSEEK_LOAD_TILE_ROWS` to override this startup setting.
External startup traces use `OH_MY_VLLM_STARTUP_TRACE` and use a location external to the repository.

The approved DeepSeek workload definitions are in the [deepseek matrix](../development/serving/deepseek_matrix.py), and the full workloads are not yet accepted.
The [DeepSeek reference calculation](models/deepseek_v41/accounting.md) defines the trace contract.
The runtime trace does not hold its counters.
The HTTP CSV records the ideal-acceptance plan only, which is not an efficiency result.

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
- Add per-request varlen position metadata and batch decode.
- Check per-request projection layouts before enabling varlen decode batches.
- Run higher-capacity AgentX cases after increasing physical cache and sequence capacity.
- Replace the DEP4 fallback with stable request ownership and expert dispatch.
- Measure the native DeepSeek paths against the reference path and public baselines.
- Move large-row native temporary storage into adapter-owned graph operator workspaces.
- Extend grouped expert GEMM to the DEP4 ownership and communication path.
- Move the DeepSeek Engram lookup into the graph, or make its host copy asynchronous.
- Evaluate the replicated `wo_b` against the measured collective cost.
- Run the approved DeepSeek AgentX cases and publish the CSV and dark scatter plots.
- Run the 38 HTTP cases and the three AgentX cases on GPUs that have no locked clock.
- Add the DeepSeek accounting counters to the execution trace before efficiency reporting.

Keep raw reports and temporary process records external to the repository.
Put host settings and model paths in `LOCAL.md`.
