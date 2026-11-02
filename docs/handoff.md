# Work state

Work continues on `nightly`.

## Current service paths

Qwen keeps its existing single-worker service path.
The shared Rust service, scheduler, logical cache, DEALER transport, and msgpack protocol stay active.
The runtime packages use `requirements/runtime.txt`.
`requirements/development.txt` includes that file for one development installation.

DeepSeek-V4.1-Flash has a model adapter and a four-rank Python bridge.
The adapter loads the original 48-shard checkpoint with rank-local placement.
It supports text target execution and DSpark5 greedy proposal verification.
The bridge supports `tp4`, `tep4`, and a documented DEP4 executor fallback.

The adapter uses the DeepSeek descriptor, cache groups, tokenizer identity, inference-config identity, native projection, routing, activation, mHC, RMS, sparse attention, and transformer support.
Prefill uses working chunks of at most 8192 total rows by default.
Set `OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS` to select a value up to 32768.
Completed chunks release their hidden tail before the next native call.

DSpark verification sends the accepted prefix and draft rows to one target call.
DSpark proposal queries share the full draft-index block.
It projects only the required logits and seeds accepted target hidden states in one MTP operation.
Rejected drafts stay private and the target correction is emitted.
Engram tables use file-backed host mappings.
Lookup stages only the selected rows on the GPU.

Released DeepSeek shapes support native projection, routing, activation, mHC, RMS, and sparse attention paths.
The native contracts accept long row ranges. Small rows use bounded operator workspaces.
Configured large prefill calls use bounded launches up to 32768 rows.
The default reduces long-context launch count. The environment override can lower temporary storage.

Native SwiGLU accepts up to 1048576 routed rows.
This row limit keeps six-route 8192-row prefill chunks on the native activation path.

DeepSeek rank workers reuse pinned host and device token buffers for prefill input copies.

DeepSeek uses the released FP4 and FP8 scale rules and nearest-even value rounding.
Dense and grouped projections reset raw dot products for each K32 group before scale multiplication and FP32 accumulation.
FP4 weights convert accurately to FP8 before the raw dot.
The FP4 conversion uses a register lookup table and keeps each sign bit.
The FP8 path reuses each weight tile for two 16-row input tiles when the input has at least 32 rows.
This reuse does not apply when the output width is at most 1024 and the input width is at least 1024.

Native RMS keeps the released square-sum reduction and reciprocal multiplication order.
Native routing keeps the released softplus and selected-score sum order.

In-place quantization rounds before dequantization.
The native quantizer writes encoded values and scales without context-sized temporary tensors.
Native attention keeps window and compressed KV in two tables.
It reads the model's BF16 tables without quantization.
The reference path joins their tables before computation.

Shared experts use full replicated FP8 weights and scales.
Their output is added after routed rank reduction.
The shared output is not reduced across ranks.
Sinkhorn mixing adds epsilon before the first column normalization.

Native mHC keeps the released product, sum, and row-count-dependent mean order.
Its FP32 projection uses the tensor runtime's cuBLAS images and an explicit 32 MiB operator workspace.
Prepared HC operator workspaces keep FP32 input and statistics storage for Eager or Graph execution.
Nonzero target chunks keep source-step projection and reduction layouts.
Their target attention and MoE execution stay batched.
Native Engram arithmetic uses the logical request batch for its per-copy reductions.

Sparse attention rounds unnormalized probabilities to BF16 in ordered 64-slot blocks before multiplication by V.
Its denominator uses the released cross-warp sum order and FMA.
The sink enters the denominator after these blocks and does not change their running maximum.

An independent development reference uses official MP4 placement and a recorded quantizer synchronization patch.
Frozen text fixtures contain full vocabulary vectors, DSpark5 proposal and verification vectors, and fresh accepted-only restoration endpoints.
The fixtures and numerical limits stay fixed during candidate acceptance.

Sparse attention status failures disable the native path and use the reference path.
Native sparse attention is the default for supported inputs.
Set `OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE=0` to select reference attention for short queries.
Long queries use native sparse attention because the reference path scales poorly.

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

TP4 gathers the FP32 token sums before adding the replicated shared output.
TEP4 keeps its routed rank reduction.

DeepSeek permits preemption and restart of the same request in one Execute.
Finished requests cannot also be scheduled or preempted in that Execute.

Grouped projection calculates row offsets on the device and does not upload host descriptors.
The grouped path keeps the local token order and falls back to the reference
expert loop for non-FP4 weights and test experts.
The MoE processes at most 1024 token rows per routed expert batch.
Static packed-expert pointer arrays use a bounded LRU cache.

MoE prepares local FP4 expert bindings and reuses their projection tuples.
Weight or expert rebinds, child `_apply`, and state loading clear these bindings.
The next grouped call prepares them again.

Native projections share compatible encoded weight storage.
Packed scales and padded weights use different storage.
Prepared cache entries keep their original weight and scale owners.
DEP4 still uses the TEP4 executor fallback and has no rank-owned dispatch.

Non-final prefill skips the vocabulary projection.
DSpark keeps only the target hidden tail needed by its MTP window.
Ordinary execution does not collect target hidden states.

Full initial prompts can run as one aligned request batch.
Shared query attention state resets at each target call.
Index sources always publish the cache view for the current request slots.
Compressed KV sources publish selected slots after the current cache write.
Compressed KV and index copies contain only the active context range.

CUDA sequence staging uses demand-mapped storage for active request slots.
Slot restoration maps the required context before copying state.
Slot release waits for device work before unmapping storage.
Index scoring uses bounded query chunks and compact candidate-block masks.

The serving engine enforces the configured `max_num_seqs` resident limit.
It keeps one additional request in preparation to overlap CPU admission with GPU work.

DeepSeek execution is currently Eager.
The shared Qwen CUDA Graph path has stream handoff, bounded graph operator workspaces, graph upload, and cleanup.
DeepSeek multi-rank Graph capture is not connected.
An audit against current vLLM and SGLang graph patterns keeps DeepSeek Eager until
static multi-rank operator workspaces and capture buckets are defined.

DeepSeek startup reads only `config.json` and `model.safetensors.index.json` for its descriptor identity.
It does not hash weight shards before Ready.

Rank-local checkpoint tiles use 65536 rows to reduce small host-to-device copies.
Rank-local parameters use 16-byte-aligned contiguous storage.
GPU startup copies required tensor ranges from each safetensors shard through a
bounded staging buffer and skips unused Engram payload ranges.
Set `OH_MY_VLLM_DEEPSEEK_LOAD_TILE_ROWS` to override this startup setting.
External startup traces use `OH_MY_VLLM_STARTUP_TRACE` and use a location external to the repository.

The approved DeepSeek workload definitions are in the [deepseek matrix](../development/serving/deepseek_matrix.py).
The full DeepSeek HTTP and AgentX workloads are not yet accepted.

## Next work

- Measure window storage at each state endpoint with the approved long-context workloads.
- Measure persistent encoded KV storage against direct BF16 table reads.
- Measure raw and prepared projection storage before long-context capacity tuning.
- Keep the independent frozen DeepSeek text endpoint checks before performance acceptance.
- Measure TP4 activation and output gathers and their temporary storage before long-context capacity tuning.
- Reduce duplicate cuBLAS operator workspace storage before long-context capacity tuning.
- Add independent image endpoints before image acceptance.
- Report the configured context limit and the observed AgentX request lengths independently.
- Tune AgentX physical block and temporary operator workspace capacity for 100K to 200K token traces at concurrency 16.
- Add per-request varlen position metadata and batch decode before DeepSeek CUDA Graph capture.
- Check per-request projection layouts before enabling varlen decode batches.
- Run higher-capacity AgentX cases after increasing physical cache and sequence capacity.
- Replace the DEP4 fallback with stable request ownership and expert dispatch.
- Measure the native DeepSeek paths against the reference path and public baselines.
- Move large-row native temporary storage into adapter-owned graph operator workspaces.
- Extend grouped expert GEMM to the DEP4 ownership and communication path.
- Define static multi-rank operator workspaces before adding DeepSeek CUDA Graph capture and replay.
- Run the approved DeepSeek AgentX cases and publish the CSV and dark scatter plots.
- Run the 38 HTTP cases and the three AgentX cases on GPUs apart from GPU 0.
- Get an independent code review before delivery.

Keep raw reports and temporary process records external to the repository.
Put host settings and model paths in `LOCAL.md`.
