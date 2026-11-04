# DeepSeek reference calculation

The [HTTP procedure](performance.md) selects the reference and validates measured results.
The calculators have separate identities:

| Calculator | Version | Source |
|---|---|---|
| Default HTTP reference and source trace | `deepseek-http-reference-v3.1` | [Trace and binding](../../../development/serving/deepseek_accounting.py), [resource mathematics](../../../development/serving/deepseek_accounting_math.py) |
| Packed protocol estimate | `deepseek-http-protocol-estimate-v5` | [Packed calculation](../../../development/serving/deepseek_accounting_packed.py) |
| Conditional partial resource floor | `deepseek-http-resource-floor-v4` | [Floor calculation](../../../development/serving/deepseek_accounting_floor.py) |

## Credit rule

The default resource reference is a lower-bound relaxation for the same accepted output and the stated TP4, TEP4, or DEP4 strategy.
Collective latency is an empirical estimate, not a proven physical lower bound.
Efficiency is `reference seconds / full HTTP seconds`.

Each decode call credits accepted drafts plus one target token per request.
Proposals, rejected rows, padding, masked slots, temporary copies, HTTP, queueing, and scheduling receive zero credit.
Replicated work adds one copy per rank.

## Rates and formats

| Resource | Rate per rank |
|---|---:|
| HBM | 7.7 TB/s |
| Dense FP8 tensor | 4.5 PFLOPS |
| Dense BF16 tensor | 2.25 PFLOPS |
| FP32 vector | 75 TFLOPS |
| NVLink 5 | 900 GB/s per direction |
| PCIe 5.0 x16 host link | 63.02 GB/s per direction |

Dense projections use FP8.
Routed experts keep FP4 storage and use FP8 dot products.
FP4 uses half a byte per value and one scale per K32 group.
FP8 uses one byte per value and one E8M0 scale per 32 by 32 block.
Projection groups count input, output, and weight bytes.

`wo_a`, sparse attention, and index scores use BF16.
Router, Hyper-Connections, Engram gate, output head, and compressors with ratio above one use FP32.
The version and rate digest bind formats, rates, placement, reuse, and overlap rules.

The [ordered K32 arithmetic](quantization.md) adds at least one FP32 FMA per output and K32 group:

```text
scale_fp32_flops = 2 * rows * local_outputs * (local_K / 32)
                 = projection_fp8_flops / 32
```

This credit applies to FP8 dense and FP4 routed projections in prefill and decode, on each rank.
It permits exact scale combination but does not assume that all scale pairs permit combination.
BF16 and FP32 projections receive no K32 scale credit.

## Sparse work and placement

Position `p` has `min(128,p+1)` window pairs.
A compressed layer adds `min(512,floor((p+1)/r))` pairs.
Index sources 2, 8, 14, and 20 score all reachable compressed positions.
Sources 24, 28, 32, and 36 score at most 16384 candidates per query.

TP4 and TEP4 compare head and query partition at each index source.
Head partition uses eight heads per rank and a packed FP32 score all-reduce.
Query partition gathers BF16 queries and head weights, computes all 32 heads on each query owner, and gathers valid top-512 INT32 positions.
Its score FLOPs and necessary key bytes divide by four in a global resource pool.
This fractional balance does not prove achievable placement.

A query-to-head transition after layer 20 gathers candidate block IDs.
Later query partitions keep those IDs on the query owner.
The calculator compares placement combinations with their resource, communication, and overlap costs.
Dense and expert partition definitions stay fixed.

Engram compares sharded packed host reads plus BF16 all-reduce with full packed host reads on each GPU.
Each packed row is 264 bytes.
Without token or hash identities, the reference credits only 24 disjoint column rows per active layer and call.
It permits full reuse across requests and tokens.
DEP4 uses full local host reads and local index scoring.

## History and weights

History has one read per target call, shared physical source, and consuming rank.
All queries and heads on that rank reuse the read.
Windows differ at each layer.
Compressed KV and index keys can share sources.

The first use credits the full necessary historical read.
A later use of `U` shared bytes credits `max(0,U-128MiB)` bytes from HBM.
This cache allowance permits favorable retention without unlimited L2 storage.
Placement comparison includes changes in the necessary subset.

For request range `[s,e)`, the historical window union has `min(s,127)` rows.
Let `n=floor(e/r)-floor(s/r)` be the new compressed rows.
Historical selected KV has at least `max(0,min(512,floor(e/r))-n)` rows.
An uncapped index source reads `floor(s/r)` historical keys.
A candidate source reads at least `max(0,min(16384,floor(e/r))-n)` historical keys.

The default calculation counts 512 BF16 values per KV row and 128 BF16 values per index-key row.
Selection reuse is maximal inside a call, with no second read for QK and PV.
Current projections count new cache writes.
New index-key projections use each request's position and compression boundary.

Prefill credits dense weights once across chunks.
For each layer and rank, the cumulative maximum touched-expert count supplies a conservative weight-union bound.
A chunk credits only the increase in that maximum.
The output head enters the first chunk with logits, only at each request's final prefill row.
Decode credits resident dense weights and touched expert weights once per call.

## Communication and overlap

TP4 counts embedding reduction, `wo_b` reduction, routed BF16 product gather, FP32 token-sum gather, and output-head gather.
TEP4 replaces routed gathers with a routed FP32 output reduction.
DEP4 sends distinct `(token,destination GPU)` BF16 rows and returns one FP32 local sum per dispatched token.
Local routes use no wire bytes.
Per-rank wire pressure is `max(sent,received)`, with a separate rate per direction.
Expert work uses received assignments, also on ranks with no owned requests.

For serial groups, `S=sum_g max_rank(time(rank,g))`.
Each group uses its maximum resource time.
FP8 and BF16 times add on the shared tensor resource.
For at least two credited target rows:

```text
R_fixed = max_rank,resource(sum(fixed_work) / rate)
R_pool = max_resource((sum_rank(fixed_work) + sum(query_work)) / (4 * rate))
R = max(R_fixed, R_pool)
B2 = max(R, S / 2)
call_seconds = min(S, B2)
```

A single target row uses `S`.
The balanced two-microbatch relaxation permits shared weight tiles and fractional resource balance.
It does not prove an achievable schedule.
Resource totals also count collective estimate seconds.

Each collective uses `max(latency_model_seconds,wire_bytes/900e9)` without a second transfer charge.
The idle, unlocked NCCL profile has 95 stored median points.
Its raw SHA256 is `c6b55ee5354c7ef6fb58606cca9268fcef90bc535935d4f07750029a81506e49`.
Lookup uses per-rank input bytes, cumulative maximum, linear interpolation, and endpoint clamps.
All-gather wire bytes use full output size.
DEP4 uses the BF16 all-to-all proxy with profile input `ceil(wire_bytes*4/3)`.

The profile digest binds the raw digest and stored table.
Medians cannot reconstruct sample minima or prove a lower bound.
DEP4 communication is modeled, not measured.
A TEP4 fallback supplies no DEP4 evidence.

## Runtime trace contract

The top-level object has `binding`, `manifest`, and `calls`.
The manifest is final only after HTTP responses complete.

```text
manifest = {mode: "ordinary" | "dspark5", requests: [
  {ordinal, id, response_id, input_tokens, output_tokens, owner_rank}
]}
call = {rank, call, phase: "prefill" | "decode", request_rows: [
  {id, start, count, accepted, proposed, logit_rows}
], expert_assignments: int32[L,4], dispatch_rows: int32[L,4],
   touched_experts: int32[L]}
```

`L=40`.
Counts are integers.
Positions and ordinals start at zero.
IDs are positive, and `response_id` starts with `chatcmpl_` and ends with `_<id>`.
Manifest lengths are HTTP usage tokens per request.
DEP4 `owner_rank` stays fixed.

Each target call has four rank records, also for ranks with no owned requests.
Each prefill chunk is a separate call.
Request `start/count` ranges join without gaps or overlap.
Decode has `count=accepted+1` and `logit_rows=count`.
Prefill has zero draft counters and one logit row only when its prompt completes.

TP4/TEP4 repeat request rows and assignment matrices on all ranks.
DEP4 records only owned request rows.
Assignments use global bucket `d=expert_id//96`.
Each matrix row sums to `credited rows*6`.
TP4 computes all four columns, TEP4 its own column, and DEP4 its column summed across source ranks.

`dispatch_rows[layer,d]` counts distinct source token/destination pairs, with local pairs.
It is zero for other strategies.
Combine reverses these pairs after the local sum.
`touched_experts[layer]` counts local expert weights for credited rows only.
DEP4 joins expert presence after Execute with one packed MAX and counter read.
Empty ranks supply zero routes and receive the expert union.

Binding holds `version`, `rate_profile`, `manifest_sha256`, source, native module digests, config/index digests, tokenizer, and four GPU UUIDs.
It also holds strategy, effective strategy, owner plan, sampling, case ID, and accepted/proposed totals.
The manifest digest uses SHA256 of sorted compact JSON without NaN values.

Validation rejects changed identities, missing ranks, duplicate calls, invalid routing, incomplete layers, repeated work, and incorrect request coverage or output counts.
A call has one phase.
Mixed phases emit `accounting_error` with the call ID and reason.
Inference continues without partial counters, but the HTTP assembler rejects the full trace.

## Packed estimate

The packed calculator uses the [physical cache formats](cache.md#current-storage).
It separates shared physical history from per-consumer traffic.
Index workspace decoding counts once per physical source and call.
Each index consumer reads the decoded keys, and each attention consumer reads packed KV.
Weight credit assumes no unproved retention between calls.
Wire time uses bandwidth alone, without the empirical NCCL profile.

TP4 supplies activations after expert up computation and before down computation.
DEP4 dispatch completes before expert computation.
Reverse FP32 combine follows expert computation.
Each direction counts separately.
Route metadata and count exchanges are diagnostics without extra reference credit.

The estimate keeps the resource and two-microbatch relaxation.
Serial protocol seconds and the conditional partial floor are separate diagnostics.
Cache retention, padding, launch latency, overlap, and MTP work do not form a full execution-time model.
The estimate is not a universal physical minimum.

`source_trace_version` and `source_trace_rate_profile` bind the default source trace.
`accounting_version`, `rate_profile`, and `assumptions_sha256` bind the calculator.
Runtime identity also records each rank's batch-decode and Graph settings, with the Graph count limit.
Those settings do not identify each step's capture or Eager fallback.

## Conditional partial floor

The floor uses fixed target arithmetic, numerical order, formats, parameter placement, and replicated projections.
Its MoE wire credit assumes fixed output shards, full BF16 row delivery, no extra weight copies, no recomputation, and no unmodeled compression or reuse.

HBM and host strict credit is zero.
Embedding, `wo_b`, head, index, Engram, and DEP4 dispatch/combine wire also have zero strict credit.
Proposals, rejected work, padding, copies, scheduler costs, empirical latency, and `S/2` receive zero strict credit.
Nonlinear scalar estimates also receive zero strict credit.
Weight unions and `direct_protocol_*` traffic are diagnostics.
This partial floor proves neither full execution time nor resident capacity.
