# DeepSeek reference calculation

The HTTP reference uses `deepseek-http-reference-v3.1` in `development/serving/deepseek_accounting.py`.
Files `deepseek_accounting_v2.py` and `deepseek_accounting_v3.py` keep earlier versions reproducible.
Use `reference_case(..., version="deepseek-http-reference-v3")` to calculate v3.

## Credit rule

The resource reference is a lower-bound relaxation for the same accepted output and the stated TP4, TEP4, or DEP4 strategy.
The collective latency component is a measured estimate. It is not a proven physical lower bound.
Efficiency is `reference seconds / full HTTP seconds`.

The DeepSeek target is at least 33% for each approved 8192-output case.

Each decode call credits accepted drafts plus one target token per request.
Draft proposals, rejected rows, padding, masked slots, temporary copies, HTTP, queueing, and scheduling receive zero credit.
Qwen credits its fixed MTP4 work, with proposals and rejected verification rows.

The two targets therefore use different credit rules.
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

Dense projections use FP8. Routed experts keep FP4 storage and use FP8 dot products.
FP4 uses half a byte per value and one scale per K32 group.
FP8 uses one byte per value and one E8M0 scale per 32 by 32 block.

`wo_a`, sparse attention, and index scores use BF16.
Router, Hyper-Connections, Engram gate, output head, and compressors with ratio above one use FP32.
Projection groups include input, output, and weight bytes.

The version and rate digest bind formats, rates, placement, reuse, and overlap rules.

### Necessary K32 scale accumulation

The released [FP8 kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py#L269) and [FP4 kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py#L553) clear each K32 raw dot before the next group.
They add `raw * sa * sb` to the ordered FP32 accumulator.
Tensor-core accumulation across groups has different rounding.

Each output and K32 group must have at least one FP32 FMA, or two FP32 FLOPs.
For the released K32-aligned shapes, v3.1 adds this work at 75 TFLOPS:

```text
scale_fp32_flops = 2 * rows * local_outputs * (local_K / 32)
                 = projection_fp8_flops / 32
```

This applies to all FP8 dense and FP4 routed projections, in prefill and decode, for each strategy and rank.
The minimum permits safe scale combination. It does not claim that combination is safe for all inputs.
Unsafe combination has more work. BF16 and FP32 projections do not get this added credit.


## Sparse work and placement

Position `p` has `min(128,p+1)` window pairs.
A compressed layer adds `min(512,floor((p+1)/r))` pairs.
Index-source layers 2, 8, 14, and 20 score all reachable compressed positions.

Layers 24, 28, 32, and 36 score at most 16384 candidates per query.
A head-partitioned FP32 score reduction contains only those valid positions, with the causal limit during prefill.

TP4 and TEP4 compare head partition and query partition at each index-source layer.
Head partition uses eight index heads per rank and a packed score all-reduce.
Query partition gathers BF16 query vectors and head weights from their projection shards, then computes all 32 heads on each query owner.

It gathers valid top-512 INT32 positions for attention.
Query ownership can divide all requests and positions across ranks.
The query price uses a fractional resource floor: total score FLOPs and necessary historical key bytes divided by four.
This is a relaxation of concrete query placement, not an achievable placement.
Query tensor and HBM work use a global resource pool. They do not force average work onto an already busy expert rank.

If layer 20 uses query partition and a later layer uses head partition, a gather transfers selected candidate block IDs.
Candidate IDs stay on the query owner when later layers also use query partition.
The calculator compares all placement combinations with their compute, HBM, and collective work with the final overlap formula.

Dense and expert TP/EP definitions stay fixed.

Engram compares sharded packed host reads plus BF16 all-reduce with complete packed host reads on each GPU.
Each packed row is 264 bytes.
The two alternatives include host transfer and HBM work. Projection and gate work are common.

Lengths do not identify repeated n-grams or hash collisions.
The reference therefore credits only the 24 disjoint column rows per active Engram layer and call, with complete reuse across tokens and requests.
This is a conservative floor, not an observed unique-row count.

A more accurate credit must have token or hash identities.
DEP4 uses complete local host reads and local index scoring.

## History and weights

Historical data has one read per target call, shared source, and rank that uses it.
All queries and heads on that rank reuse that read.
Window KV is different at each layer. Compressed KV and index keys can have a shared source across layers.

The first use credits the full necessary historical read.
A later use of `U` shared bytes credits at least `max(0,U-128MiB)` bytes from HBM.
The [Qwen operator cache policy](../qwen38/operator-performance.md) supplies this 128 MiB capacity allowance.
The reference permits the most favorable retention of the selected subset, but does not permit unlimited L2 storage.
A change of query ownership can increase the necessary subset.
Placement comparison includes this reuse credit.

For request range `[s,e)`, the historical window union has `min(s,127)` rows.
Let `n=floor(e/r)-floor(s/r)` be the new compressed rows.
Historical selected KV has at least `max(0,min(512,floor(e/r))-n)` rows.

An uncapped index source reads `floor(s/r)` historical keys.
A candidate source reads at least `max(0,min(16384,floor(e/r))-n)` historical keys.
Each KV row has 512 BF16 values. Each index-key row has 128 BF16 values.

The rules allow maximum selection reuse inside the call.
They do not add a second read for QK and PV.
Current projections include their new cache writes.

New index-key projections use each request's position and compression boundary.
Each index-key weight enters its first chunk with a new key.

Prefill credits dense weights once across its chunks.
The largest touched-expert count across chunks is a conservative lower bound for the weight union, separately for each layer and rank.
Each chunk credits only the increase of the cumulative maximum touched count.
The total stays equal to the maximum count, without charging later expert weights in an earlier chunk.
Request ownership does not control this credit.

The output-head weights enter the first chunk that actually computes logits.
Only the final prefill row of each request computes logits.
Decode credits resident dense weights and touched expert weights once per call.

## Communication and overlap

TP4 includes embedding reduction, `wo_b` reduction, routed BF16 activation-product gather, FP32 token-sum gather, and output-head gather.
TEP4 replaces routed gathers with the routed FP32 output reduction.
DEP4 dispatch sends distinct `(token,destination GPU)` BF16 rows.

The destination sums local expert outputs before returning one FP32 row per dispatched token.
Local routes use no wire bytes.
Per-rank wire pressure is `max(sent,received)`. Each direction has its own link rate.

Expert work uses received assignments separately for each layer, with ranks with no owned requests.

For serial groups, `S=sum_g max_rank(time(rank,g))`.
A group uses the maximum resource time. FP8 and BF16 share the tensor resource and their times add.
For at least two credited target rows, the call price is:

```text
R_fixed = max_rank,resource(sum(fixed_work) / rate)
R_pool = max_resource((sum_rank(fixed_work) + sum(query_work)) / (4 * rate))
R = max(R_fixed, R_pool)
B2 = max(R, S / 2)
call_seconds = min(S, B2)
```

Resource totals also include summed collective estimate seconds.
A single target row uses `S`.
`B2` is a balanced two-microbatch lower-bound relaxation, not an achievable execution time.
It keeps inside-microbatch dependencies through half the serial chain and keeps each resource total.

The split permits weight tiles to serve the two microbatches before eviction.
It therefore credits the weight union once. It does not add two complete weight reads.
A schedule that reloads weights has more work.

Communication bytes and estimated latency stay in the resource totals.

Each collective uses `max(latency_model_seconds,wire_bytes/900e9)`.
Transfer time is already inside the measured latency, so it is not added twice.
The version-bound idle, unlocked NCCL profile keeps 95 stored median points.

The raw SHA256 is `c6b55ee5354c7ef6fb58606cca9268fcef90bc535935d4f07750029a81506e49`.
Lookup uses per-rank input bytes, cumulative maximum, linear interpolation, and endpoint clamps.
All-gather wire bytes use full output size.

DEP4 uses the BF16 all-to-all proxy at equal byte size, with profile input `ceil(wire_bytes*4/3)`.
FP32 combine keeps four-byte values.
The profile digest binds the raw digest and stored table.

The stored medians cannot reconstruct per-size sample minima. Neither statistic proves a lower bound.
DEP4 communication stays modeled, not measured.

A TEP4 fallback cannot supply DEP4 evidence.

## Runtime trace contract

The top-level object has `binding`, `manifest`, and `calls`.
The manifest is finalized after HTTP responses complete.

```text
manifest = {mode: "ordinary" | "dspark5", requests: [
  {ordinal, id, response_id, input_tokens, output_tokens, owner_rank}
]}
call = {rank, call, phase: "prefill" | "decode", request_rows: [
  {id, start, count, accepted, proposed, logit_rows}
], expert_assignments: int32[L,4], dispatch_rows: int32[L,4],
   touched_experts: int32[L]}
```

`L=40`. All counts are integers. Positions and ordinals start at zero.
IDs are positive. `response_id` starts with `chatcmpl_` and ends with `_<id>`.
Manifest lengths are HTTP usage tokens per request.

`owner_rank` is stable and applies to DEP4.
Each executed target call has four rank records. This includes ranks with no owned requests.
A prefill chunk is a separate call. Request `start/count` ranges must join without gaps or overlap.

`count` contains only credited rows. Decode must have `count=accepted+1` and `logit_rows=count`.
Prefill must have zero draft counters and one logit row only when the prompt completes.
TP4/TEP4 repeat request rows and assignment matrices on all ranks.

DEP4 records only each rank's owned request rows.

Each assignment uses global expert bucket `d=expert_id//96` in all strategies.
Each matrix row sums to `credited rows*6`, not four copies of that value.
TP4 computes all four columns. TEP4 computes its own column.

DEP4 computes its column summed over all source rank records.
`dispatch_rows[layer,d]` counts distinct source token/destination pairs, with local pairs.
It is zero in strategies other than DEP4. Combine reverses those same pairs after the local sum.

`touched_experts[layer]` counts local expert weights for credited rows only.
The device can accumulate all three arrays with the accepted-row mask, also inside CUDA Graphs.
DEP4 defers the expert-presence MAX until all ranks finish Execute.
It packs all global calls into one MAX and one counter read.
Empty ranks supply zero source routes and receive the expert union.
Other strategies read counters after a call or request.

Sparse pairs, index pairs, and history bytes are calculated from positions and model configuration.

Binding holds `version`, `rate_profile`, `manifest_sha256`, source, native module digests, checkpoint config/index digests, tokenizer, and four GPU UUIDs.
It also holds strategy, effective strategy, owner plan, sampling, case ID, and total accepted/proposed counters.
The manifest digest uses SHA256 of sorted compact JSON with no NaN values.

Validation rejects changed identities, missing ranks, duplicate calls, invalid routing bounds, and incomplete layers.
Per-request state checks full prefill coverage, contiguous accepted positions, correct output count, and no duplicated work, even with new call IDs.

A v3.1 call has one phase.
A target call with mixed phases emits `accounting_error` with its call ID and reason.
Inference continues without partial counters or extra call records.
This event is not a v3.1 call.
The HTTP assembler rejects the full trace if any rank records this error.

## Known relaxations

Query ownership and two-microbatch totals permit fractional balance.
They do not prove an achievable schedule.
Without token identities, historical selections use their smallest possible union.
Engram permits full reuse of hash rows in a call.

Prefill weight credit uses maximum counts instead of measured expert unions.
The 128 MiB cache allowance can favor history without reserving space for other work.
These rules can decrease the reference.
The collective profile is a version-bound empirical estimate.

## Ideal reference values

The 19 long-output rows use ideal acceptance, balanced synthetic routing, and expected distinct expert counts.
They are reference scales, not measured HTTP efficiency results.
Each DSpark5 call accepts all five drafts, except the final truncated call.

The runtime must supply the bound trace before measured efficiency reporting.

| Case | v3/lower_bound s | v3.1/lower_bound s |
|---|---:|---:|
| `dspark5:tp4:4:400000:8192` | 12.996 | 17.417 |
| `dspark5:tp4:4:800000:8192` | 21.819 | 29.933 |
| `dspark5:tp4:16:400000:8192` | 43.428 | 61.115 |
| `dspark5:tp4:16:800000:8192` | 78.722 | 111.179 |
| `dspark5:tp4:64:400000:8192` | 144.266 | 215.015 |
| `dspark5:tp4:64:800000:8192` | 285.440 | 415.269 |
| `dspark5:tep4:4:400000:8192` | 12.794 | 17.407 |
| `dspark5:tep4:4:800000:8192` | 21.427 | 29.923 |
| `dspark5:tep4:16:400000:8192` | 42.692 | 61.144 |
| `dspark5:tep4:16:800000:8192` | 77.221 | 111.207 |
| `dspark5:tep4:64:400000:8192` | 141.087 | 214.895 |
| `dspark5:tep4:64:800000:8192` | 279.203 | 415.148 |
| `dspark5:dep4:4:400000:8192` | 10.421 | 12.152 |
| `dspark5:dep4:4:800000:8192` | 16.284 | 18.287 |
| `dspark5:dep4:16:400000:8192` | 29.712 | 36.636 |
| `dspark5:dep4:16:800000:8192` | 53.166 | 61.179 |
| `dspark5:dep4:64:400000:8192` | 85.678 | 113.374 |
| `dspark5:dep4:64:800000:8192` | 179.494 | 211.546 |
| `ordinary:4:400000:8192` | 27.932 | 32.354 |

Use the [test scope](../../testing.md) and [reporting policy](../../testing.md#reporting).
