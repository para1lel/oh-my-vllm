# DeepSeek V4.1 memory plan

## 1. Scope and authority

The CPU [planner](../../development/model/deepseek_v41/memory.py) supplies a storage plan, not runtime capacity acceptance.
It uses the independent [tensor layout](../../development/model/deepseek_v41/layout.py), not checkpoint headers or index totals.
The [source contract](model.md) binds `LOCAL-V41:inference/model.py` and its related sources.
The [state contract](state.md) supplies accepted-state and restoration rules.
The [architecture](design.md) defines the four-rank strategies.
Rank IDs are logical group IDs, not physical GPU indices.

The plan covers 64 admitted requests, with 1048576 input/output tokens per request.
DSpark has five draft rows and up to six target verification rows.
Smaller kernel fragments do not reduce admitted request counts.
No GPU computation, allocation measurement, throughput measurement, or numerical acceptance is part of this plan.

## 2. Input and execution

`plan_memory(config, strategy)` returns a JSON-serializable dictionary.
`strategy` is `TP4`, `TEP4`, or `DEP4`.
The configuration has these fields:

| Field | Meaning |
|---|---|
| `model` | Complete supported checkpoint configuration |
| `device_capacity_bytes` | Four positive device capacities, in rank order |
| `device_unavailable_bytes` | Four unavailable amounts, subtracted before budgeting |
| `host_pinned_capacity_bytes` | Available shared pinned-host budget |
| `host_reservations` | Explicit amounts for `kv_offload`, `uploads`, and `transfer_leases` |
| `reservations` | Four-rank byte lists for `graphs`, `staging`, `communication`, `chunks`, and `media` |
| `limits` | Optional limit overrides |

The default limits are `requests=64`, `context_tokens=1048576`, `draft_tokens=5`, `prefill_chunk_rows=128`, and `index_tile_positions=4096`.
Context counts input and output together.
Requests must be 1-64.
Context must be 1-1048576.
Draft capacity is 0 or 5 because the source block computes five rows together.
Zero selects ordinary decode, but keeps all draft weights loaded.

Chunk rows must be 1-67108864.
Index tiles must be 1-1048576.

Unknown fields, unsupported model shapes, invalid owners, invalid scale blocks, and non-four-rank topologies cause errors.
Byte arithmetic checks unsigned 64-bit overflow.
Budget differences can be negative.
Budgets that are not sufficient produce a false budget result, not a smaller request limit.

The command reads the complete planner configuration from `PLAN_CONFIG`:

```bash
scripts/with-env.sh python -m development.model.deepseek_v41.memory \
  --config "$PLAN_CONFIG" --strategy TP4 > "$PLAN_OUTPUT"
```

The command writes the complete plan to standard output.
No model-source import or GPU runtime import occurs.

## 3. Tensor placement

An exact-name rule assigns each expected tensor.
Unmapped tensors cause errors instead of default replication.
The output gives stored shapes, stored bytes, shard axes, half-open shard ranges, owner ranks, and per-rank resident bytes.
Ranges refer to stored axes, also for packed FP4 bytes and scale tensors.

| Tensor family | TP4 | TEP4 | DEP4 |
|---|---|---|---|
| Routed experts | Four tensor shards per expert | Complete expert on one owner | Complete expert on one owner |
| Shared experts | Tensor shards | Tensor shards | Replicas |
| Query expansion, grouped output projection, index queries | Tensor shards | Tensor shards | Replicas |
| Vocabulary embedding/head and Markov embedding/head | Vocabulary-row shards | Vocabulary-row shards | Replicas |
| Vision, aligner, image marker weights | Rank 0 | Rank 0 | Rank 0 |
| Engram tables and scales | One shared pinned store | One shared pinned store | One shared pinned store |
| Engram projections and norm weights | Device replicas | Device replicas | Device replicas |

Expert intermediate shards split `w1` and `w3` output rows and `w2` input columns.
Complete-expert ownership uses contiguous ranges: 96 target experts or 32 draft experts per rank.
Shared experts follow the dense strategy, not routed-expert ownership.
Vision runs on rank 0.
Feature transfer to request owners has a separate budget.

The attention layout follows the source classes.
`wq_b` and `wo_a` split output rows.
`wo_b` splits input columns.
`attn_sink` splits heads.
Index `wq_b` and `weights_proj` split output heads.
Latent projections `wq_a`, `wkv`, compressor weights, and index-key producers are replicated.

Query/KV norm vectors, routing gates, hyper-connection matrices, and small scalar/vector parameters are replicated.
They are not divided by four.
FP8 matrix shards respect 32-channel scale blocks.
FP4 reduction shards respect 32 logical input channels and two values per byte.
FP4 output-row shards keep complete per-row scales.
Small BF16 matrices do not acquire FP8 block padding.

All three DSpark blocks follow the same attention, dense, and expert rules as target blocks.
`main_proj`, its norm, the final draft norm, and the confidence projection are replicated.
Draft vocabulary embedding/head weights alias the target weights.
The plan adds no duplicate tensors.

## 4. Weight bytes and formats

Stored bytes and planned resident bytes are separate.
The plan keeps routed experts packed as FP4 and most dense matrices as FP8 with UE8M0 scales.
It follows the source's explicit parameter conversions: grouped `wo_a` becomes BF16.
Its checkpoint scales are consumed during loading.
Ratio-2 compressor projections, vocabulary heads, Markov heads, and the confidence projection become FP32.
Load scratch and overlapping source/destination allocations are not persistent weight bytes.

The [encoded reader](../../python/oh_my_vllm/models/deepseek_v41/checkpoint.py) loads selected vector and matrix slices without decoding.
Its source buffer and each destination have separate 64 MiB caps.
Tensor placement and precision conversions are separate operations.

The [rank loader](../../python/oh_my_vllm/models/deepseek_v41/weights.py) uses the same placement rules as this planner.
It checks configuration and layout identities before device allocation.
It converts only the specified weights and prepares native projection layouts.
One contiguous allocation per data type holds the weights, with each tensor aligned to 256 bytes.
A separate buffer supplies temporary packing storage.
Its supplied GPU byte limit includes alignment and the complete temporary buffer.

Temporary storage is reused only after packing completes and is released before the loaded weights are returned.

The caller must reserve separate space for allocator overhead, CUDA contexts, and other storage.
Each upload completes before its host tile is reused.
If the completion check fails, preparation keeps its storage until process termination.

These are class-derived byte counts, not allocation measurements:

| Strategy | Rank 0 weights | Each rank 1-3 weights |
|---|---:|---:|
| TP4 | 79217294664 | 78246757704 |
| TEP4 | 79217294664 | 78246757704 |
| DEP4 | 88392951624 | 87422414664 |

The shared Engram tables contain 196613849600 bytes.
Their scales contain 6144182800 bytes.
The authoritative pinned total is 202758032400 bytes, not four copies.
Engram projections stay on devices and are included in the rank totals.

The [Engram loader](../../python/oh_my_vllm/models/deepseek_v41/engram.py) streams encoded tensors into one shared file, without weight expansion.
It checks whole-shard identities before sealing, with a stream buffer of at most 64 MiB.
The coordinator must set CPU NUMA policy before loading.
Source page cache is separate from the stream buffer and shared storage.

The [registration owner](../../python/oh_my_vllm/operators/engram/lookup.py) maps the sealed file.
It uses read-only CUDA registration when supported.
Otherwise, supported HMM systems use `mlock` with host placement and GPU access advice.
Page-rounded mappings must fit one process-wide pinned budget.
Workspace owners keep mappings through Eager completion and graph release.
Service allocation enforcement and model integration are pending.

## 5. Persistent state

TP4 and TEP4 schedule the same request slots on all ranks.
Their single latent KV vector is replicated despite query-head sharding.
DEP4 assigns stable request slot `s` to rank `s%4`, with no implicit migration.
At 64 requests, each DEP4 attention owner stores 16 requests.
Routed-expert owners can receive work when their attention partition is empty.

The cache plan uses proposed packed formats, not the reference's dequantized BF16 cache allocations.
Packed counts exclude kernel-specific row padding.
Main global KV uses 288 bytes per row: FP4 values with E4M3 scales per 16 channels.
Index keys use 68 bytes per row: FP4 values with UE8M0 scales per 32 channels.

Sources 2, 8, and 14 store `floor(context/2)` rows.
Source 20 stores `context` rows.
Consumers do not allocate duplicate global stores.

At the combined limit, global state is 59726888960 bytes per TP4/TEP4 rank, or 14931722240 bytes per DEP4 rank.

The following capacities are per owned request at the default limits:

| State | Planned capacity |
|---|---|
| Accepted target windows | 40 layers, 128 rows per layer, 528 bytes per row |
| Accepted draft windows | 3 layers, 128 rows per layer, 528 bytes per row |
| Partial compressors | 3 sources, 2 slots, FP32 values and gate scores at width 512 |
| Target window shadow | 40 layers, 6 rows per layer |
| Draft window shadow | 3 layers, 6 rows per layer |
| Global shadow | 15 combined main-KV/index-key rows |
| Draft private KV | 3 layers, 5 rows per layer |
| Target features | 3 mean residual inputs at width 5120, BF16 |
| Feature shadow | 6 target-feature rows |
| Proposal distributions | 5 complete FP32 vocabulary rows |

Window rows contain 512 FP8 values and 16 UE8M0 scales.
Partial compressors have at most one valid unfinished token per source.
Two allocated slots follow the source buffer shape.
A separate partial snapshot protects rejection across compression boundaries.
Shadows do not replace full accepted windows or authorize truncated replay.

The plan also counts token history, penalty counters, Engram history/snapshots, Markov features, token mapping, hash constants, and two shared rotary tables.
Allocation contracts for grammar state, logical identities, prefix sharing, and offload records are not complete.

## 6. Allocation caps and open bounds

`plan_memory` charges the greater of supplied reservations and buffer floors.
Its `fits_supplied_budgets` result checks accounted storage only.
The [allocation wrapper](../../development/model/deepseek_v41/allocation.py) consumes this plan once through `plan_allocation(plan, **budgets)`.
It keeps persistent weight/cache bytes and checks the full default limits.

Each device cap is the greater of its design cap and the corresponding planner charge.
Host `kv_offload`, `uploads`, and `transfer_leases` reservations also stay as lower bounds.
Larger reservations increase explicit spare capacity and totals.

The output gives one device buffer table, per-rank caps, and separate host arenas.
Buffer records give shapes, element sizes, and counts.
Each arena supplies alignment, lifetime, reuse conditions, and checked child sums.
Unsupported limits, unsigned 64-bit overflow, child overflow, or startup budget shortage cause errors.

The full 64-request step has 384 target rows, 320 predictor rows, 2304 target routed edges, and 960 draft routed edges.
It does not impose 16-request fragments.
The 4 GiB compute cap includes 1 GiB kernel scratch.

Each rank has a separate 4 GiB native weight-layout cap.
It charges only additional bytes for encoded FP4 padding and packed scales.
Do not keep replaced compact storage.
FP4 reduction widths round up to multiples of 128. FP8 widths stay unchanged.
The per-rank output gives exact growth, component bytes, and unused capacity.

The 64 MiB staging cap holds two Engram banks.
Communication, state, graph, native, media, padding, and startup storage have separate charges.
Source buffer details belong in the callable output.

Supply four `device_reserve_bytes` values.
Device startup must fit capacity minus unavailable bytes from the input plan, minus this separate reserve.
Supply `host_capacity_bytes`, `host_unavailable_bytes`, and `host_reserve_bytes` for the host total.
Pinned startup must also fit the input plan's available pinned budget.
Startup adds 128 MiB per device, 512 MiB pinned host storage, and 512 MiB pageable host storage.
These buffers overlap runtime storage.

`pageable_reservations` has six caps: `control`, `request_payloads`, `media`, `processes`, `registration`, and `kernel_io`.
Bind `request_state_max_bytes` and `response_max_bytes` to the existing global CLI budgets.
The `request_payloads` cap charges their sum after alignment.
For example, 16 GiB and 8 GiB are design settings, not defaults or per-request limits.
This wrapper adds no API payload or grammar limit.

Native, graph, and decoder allowances are caps, not proven physical bounds.
Grammar parser storage and `deep_copy` byte bounds are unresolved.
Request/response physical copies and media preparation peaks also have no proven bounds.
The wrapper lists these gaps and sets `runtime_feasibility_verified=False`.
It does not implement runtime allocation guards or complete Phase 0.

Before Ready, the runtime must enforce arena caps and reject unsupported allocations.
The [capacity procedure](validation.md#capacity-checks) must validate peak allocations, exact continuation, and simultaneous admission before service capacity claims.
