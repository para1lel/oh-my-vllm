# DeepSeek reference calculation

The [HTTP procedure](performance.md) selects the calculator and validates measured results.
Efficiency is `reference seconds / full HTTP seconds`.

| Calculator | Version | Use |
|---|---|---|
| Default reference and source trace | `deepseek-http-reference-v3.1` | Historical estimate and authenticated target counters |
| Packed protocol estimate | `deepseek-http-protocol-estimate-v5` | Historical protocol estimate, not a latency floor |
| Conditional partial floor | `deepseek-http-resource-floor-v4` | Necessary arithmetic and limited MoE wire |
| Fixed-algorithm roofline | `deepseek-http-roofline-v6` | Current HTTP efficiency reference |

The [source validator](../../../development/serving/deepseek_accounting.py) authenticates all four calculators.
The [roofline calculator](../../../development/serving/deepseek_accounting_roofline.py) uses the reviewed [partial arithmetic](../../../development/serving/deepseek_accounting_floor.py).
The [resource functions](../../../development/serving/deepseek_accounting_math.py) give dimensions, formats, and peaks.

V4 reports `dep4_communication_modeled=False` because its DEP4 protocol bytes are diagnostic values only.
V5 and v6 report `dep4_communication_modeled=True` for DEP4.
For TP4 and TEP4, these calculators report `dep4_communication_modeled=False`.
This flag identifies the selected calculator's capability for the selected strategy, independent of the trace's communication volume.
V4, v5, and v6 report `dep4_communication_measured=False` because their communication estimates are mathematical.

## Roofline scope

V6 is a conditional resource lower bound for the fixed target algorithm, numerical rules, and four-rank placement.
It is not a minimum for all algorithms with the same output.
Maximum overlap and favorable data reuse are permitted.
Extra weight copies, projection recomputation, wire compression, and cross-token wire-value reuse are not permitted.

TP4 keeps output-channel expert shards.
TEP4 keeps expert owners and tensor-parallel attention.
DEP4 keeps request owners and full experts on their owners.

Each request credits `input_tokens + output_tokens - 1` target rows and `output_tokens` logit rows.
Proposals, rejected rows, padding, copies, scheduler costs, and MTP work receive zero credit.
Replicated matrices add one copy on each consuming rank.
Expert arithmetic uses received assignments, also on ranks with no owned requests.

| Resource | Peak per rank |
|---|---:|
| HBM | 7.7 TB/s |
| Dense FP8 tensor | 4.5 PFLOPS |
| Dense BF16 tensor | 2.25 PFLOPS |
| FP32 vector | 75 TFLOPS |
| NVLink 5 | 900 GB/s per direction |
| PCIe 5.0 x16 | 63.02 GB/s per direction |

The HGX B200 column on page 8 of the [NVIDIA Blackwell datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf#page=8) gives the HBM, FP32, and tensor peaks per GPU.
Footnote 2 gives dense tensor peaks as half the listed sparse values.

FP4 expert storage uses FP8 dot products.
FP8 and BF16 share the tensor resource.
[Ordered K32 arithmetic](quantization.md) adds at least `FP8_FLOPs / 32` FP32 FLOPs.
`wo_a`, attention products, and index scores use BF16.
Router, HC projections, Engram gate, head, and ratio-above-one compressors use FP32.
Scalar nonlinear estimates receive zero credit.

For each rank, tensor seconds are `FP8_FLOPs/4.5e15 + BF16_FLOPs/2.25e15`.
Each other resource uses its quantity divided by its peak.
The reference takes the maximum across ranks and resources.
A fractional four-rank pool is permitted for TP4 and TEP4 index query work.
Pooled necessary wire and host bytes divide by four times the related peak.
There is no empirical latency, observed-call serial sum, or `S/2` term.

## HBM and host necessity

The retention allowance is `F = 1 GiB` per rank, with `1 GiB = 2^30 bytes`.
It includes registers, data caches, shared memory, and tensor memory.
It does not include HBM, Torch allocations in HBM, and host memory.

The [NVIDIA guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) gives 126 MB L2, 256 KiB registers per SM, and 256 KiB combined L1/shared memory per SM.
The [B200 layout](https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/blackwell_chip) gives a conservative 160-SM physical layout.
The [SM description](https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/b200_sm) gives 256 KiB tensor memory per SM.
Using 126 MiB for L2 gives 246 MiB for these data stores.
The 1 GiB allowance includes the other small caches and favorable retention.

Let `W_union` be the necessary whole-HTTP weight union on one rank.
Assignment counts give an expert-union bound without assuming new experts on each call.
Let `W_dense` contain only matrices necessary in each target round, with the head.
`W_dense` does not include intermittent compressor and index-key matrices.

A target round gives at most one new row in ordinary mode or six in DSpark5.
One request has at least `N = 1 + ceil((output_tokens-1)/limit)` rounds, with prompt completion.
The next round on that request depends on the preceding result.
Each round uses the dense matrices.
At most `F` weight bytes can survive from before that interval.
Use the longest owned request chain on each rank:

```text
weight_read_bytes = max(0, W_union-F, N*max(0,W_dense-F))
```

The union and chain arguments can represent the same reads, so they must not add.
An empty DEP4 owner still credits its received-expert weight union.

[Native history](cache.md#current-storage) uses 288-byte compressed KV, 68-byte index keys, and 1024-byte BF16 windows.
The four physical sources receive credit once.
Consumers do not make new storage.
For each request, let `P` be the final encoded KV and index payload.
Initially empty state must publish this history without recomputation.
At most `F` bytes can stay unwritten, so mandatory cache writes are `max(0,P-F)` per request.

Weight reads and cache writes represent different transfers and add.
Repeated history reads, window writes, workspace decoding, and intermediate activation traffic receive zero credit without a necessity proof.

Engram starts in host storage.
Its 24 disjoint columns use at least 24 packed rows per active layer, each with 264 bytes.
Full reuse across all tokens and requests is permitted.
TP4/TEP4 pool this initial host transfer.
Each active DEP4 owner uses its local rows.
Repeated lookups receive zero credit without token and hash identities.

## Communication and dependencies

The floor and full protocol diagnostics are separate.
For fixed full BF16 MoE delivery, TP4/TEP4 use at least `3*rows*H*2` aggregate wire bytes per layer.
TP4 also uses `3*rows*K*I*2` for the activation product.
Here `H=5120`, `K=6`, and `I=2304`.
Favorable rank balance and final BF16 delivery are permitted in these aggregate bounds.
They do not charge a full FP32 reduction as necessary information.

Embedding, `wo_b`, head, index, and Engram wire receive zero floor credit.
Greedy selection does not depend on a full vocabulary gather.

DEP4 uses direct BF16 dispatch and reverse FP32 local-sum delivery.
Let `D[s,d]` count distinct credited `(source token,destination rank)` pairs, also with local pairs.
For rank `r`, do not count `s=d`:

```text
out_r = sum_d D[r,d]
in_r  = sum_s D[s,r]
sent_r     = H*(2*out_r + 4*in_r)
received_r = H*(2*in_r  + 4*out_r)
```

Sum each direction over the whole HTTP workload before taking its resource maximum.
Do not add independent dispatch and combine maxima as a serial floor.
Local routes, metadata, count exchange, padding, and trace reductions receive zero credit.

The protocol diagnostics put each useful collective at its dependency:

| Strategy | Ordered dependency |
|---|---|
| TP4/TEP4 | Embedding lookup, BF16 reduction, first layer |
| TP4/TEP4 | Engram lookup, BF16 reduction, Engram projection, layer attention |
| TP4/TEP4 | Index scoring, FP32 reduction, sparse attention |
| TP4/TEP4 | `wo_b`, FP32 reduction, router |
| TP4 | Expert up/gate, BF16 product gather, down, FP32 token-sum gather |
| TEP4 | Expert up/gate, down, FP32 output reduction |
| DEP4 | Router, BF16 dispatch, expert up/gate, down, reverse FP32 combine |
| TP4/TEP4 | Head projection, FP32 vocabulary gather, sampling |

Different rows can overlap these stages.
V6 calculates a resource floor without a dependency latency lower bound.
The `dependency_seconds=0` field means dependency latency is not included.
The target algorithm still has dependencies.
The aggregate resource constraints stay necessary with maximum overlap.
V6 does not calculate the full minimum physical latency.

For four ranks, ring all-reduce sends and receives `1.5*N` bytes per rank for an `N`-byte input.
Ring all-gather sends and receives `0.75*N` bytes for an `N`-byte full output.
With four target rows, TP4 product gather uses 82944 bytes per direction per rank.
Its FP32 token-sum gather uses 61440 bytes.
TEP4 output reduction uses 122880 bytes.

For `D[0,1]=2` and `D[1,0]=1`, rank 0 dispatch sends 20480 bytes and receives 10240 bytes.
Its reverse combine sends 20480 bytes and receives 40960 bytes.
These protocol bytes are diagnostics, not extra floor credit.

## Trace and identities

The top-level trace has `binding`, `manifest`, and `calls`.
The HTTP response finalizes the manifest:

```text
manifest = {mode: "ordinary" | "dspark5", requests: [
  {ordinal, id, response_id, input_tokens, output_tokens, owner_rank}
]}
call = {rank, call, phase: "prefill" | "decode", request_rows: [
  {id, start, count, accepted, proposed, logit_rows}
], expert_assignments: int32[40,4], dispatch_rows: int32[40,4],
   touched_experts: int32[40]}
```

Each call must have four rank records, also for empty owners.
Request ranges join without gaps or overlap.
Decode credits `count=accepted+1`.
Prefill credits one logit row only at prompt completion.
TP4/TEP4 repeat request and assignment counters.
DEP4 records owned requests only.

Each layer has six assignments per credited row, bucketed by `expert_id//96`.
TP4 uses all buckets, TEP4 its bucket, and DEP4 its bucket summed across source ranks.
DEP4 joins expert presence after Execute and records distinct dispatch pairs.
Validation rejects missing ranks, repeated work, invalid routes, incomplete layers, and incorrect HTTP coverage.
Mixed phases emit `accounting_error`.
The assembler rejects the trace.

Binding includes source, native modules, checkpoint, tokenizer, four GPU UUIDs, placement, sampling, and manifest digest.
Runtime identity records batch and Graph settings, but not each step's capture or Eager fallback.
V6 independently binds its version, peaks, model dimensions, retention allowance, and assumptions digest.
The acceptance checker recalculates the selected reference from the bound trace.
A changed identity or stored value fails validation.

V6 qualification checks four B200 model names in measurement topology.
Missing names and other models cannot pass.
Historical diagnostics do not use this hardware check.

## Historical estimates

V3.1 keeps BF16 historical traffic and empirical NCCL median latency.
The [packed v5 calculator](../../../development/serving/deepseek_accounting_packed.py) keeps per-call weight reloads and its two-microbatch estimate.
These identities and calculations stay unchanged for old records.
V5 does not include the sharded Engram reduction in its protocol stages.
Its end-of-layer Engram placement also differs from the runtime's layer-entry dependency.
V6 diagnostics correct these details without adding floor credit.

Splitting the same accepted work into more calls can increase v5 seconds.
The estimate label and a 33% ratio do not make v5 a latency lower bound.
V4 gives zero HBM, host, and DEP4 wire credit and stays a partial reference.
The [acceptance procedure](performance.md#check-the-long-output-rows) uses v6 for current efficiency acceptance.
Static case scales are estimates, not HTTP evidence or numerical, capacity, or AgentX acceptance.
