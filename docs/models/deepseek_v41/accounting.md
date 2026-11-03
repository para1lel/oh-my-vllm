# DeepSeek reference calculation

The DeepSeek HTTP report uses the `deepseek-http-reference-v2` calculation.
The previous `deepseek-http-reference-v1` calculation stays available with the `wire` communication model.
The code is in `development/serving/deepseek_accounting.py`.

## Rule

The reference is a lower bound on the time to make the same accepted output.
It prices the necessary resource work of each rank from a frozen execution trace.
It takes the maximum of four ranks for each group.
It adds the serial groups.

The reference does not price queueing, HTTP, or scheduling.
It does not price padding or masked slots.
It does not price weight reads for each prefill chunk or temporary copies.
It does not price draft proposals or rejected rows.
Replicated work adds one copy to each rank.
The calculation uses the legal placement with the smallest cost.

## Rates

| Resource | Rate |
|---|---:|
| HBM | 7.7 TB/s |
| Dense FP8 tensor | 4.5 PFLOPS |
| Dense BF16 tensor | 2.25 PFLOPS |
| FP32 vector | 75 TFLOPS |
| NVLink 5 | 900 GB/s per direction |
| Idle unlocked collective latency | Maximum of the latency model and NVLink byte time |
| Host link | 63.02 GB/s per direction |

The host link is PCIe 5.0 x16.
The accounting identity holds the rate profile digest.
A change to a rate, a format, an overlap rule, or a credit rule makes a new version.

## Formats

| Work | Price |
|---|---|
| Dense projections | FP8 |
| Routed experts | FP4 storage, FP8 dot product |
| `wo_a`, sparse attention, indexer scores | BF16 |
| Router, Hyper-Connections, compressor with `ratio > 1`, Engram gate, output head | FP32 |

The adapter changes FP4 weights to FP8 before the raw dot product.
FP4 storage keeps half a byte for each value and one scale for each K32 group.
FP8 storage keeps one byte for each value and one E8M0 scale for each 32 by 32 block.

## Sparse work

A query at position `p` selects `min(128, p+1) + min(512, floor((p+1)/r))`
attention pairs for a layer with compress ratio `r`.

The 8 index-source layers score compressed positions.
Layers 2, 8, 14, and 20 score each reachable position.
Layers 24, 28, 32, and 36 score at most 16384 candidates for each query.
TP4 and TEP4 partition index heads, then all-reduce the FP32 score buffer.
Each index-source layer uses its own effective context width.

Engram host transfer uses the packed FP8 row and its scales.
One row is 264 bytes.
TP4 and TEP4 add the 24 by 256 BF16 Engram all-reduce for each token and layer.
DEP4 reads complete rows from the host mapping and has no Engram all-reduce.

## Prefill and decode

Prefill reads each weight one time.
A chunked prefill trace uses the maximum touched expert count of its chunks.
That value is a lower bound for the union.
The first chunk with rows carries the weight credit.
The reference counts the output head only for the last prefill row of each request.

Decode uses the target-call count and the accepted counters from the trace.
Each call credits the accepted drafts and one target token.
Draft proposals and rejected verification rows cost zero.
Each call reads the resident dense weights and the touched expert weights one time.

## Communication

TP4 counts the activation-product all-gather, the FP32 token-sum all-gather,
the `wo_b` all-reduce, the embedding all-reduce, and the head all-gather.

TEP4 counts the routed output all-reduce for each layer and the TP4 dense collectives.

DEP4 partitions requests.
Each rank reads the full dense weights for each call.
The reference counts dispatch with BF16 rows and combine with FP32 rows.
The dispatch and combine wire bytes come from the 4 by 4 route matrix.
The matrix holds counts for one layer.
The reference multiplies the wire bytes by the layer count.

A zero-row rank adds no work.
A TEP4 fallback trace is not DEP4 evidence.

Wire bytes use `max(sent, received)` for each rank.
Each direction has its own 900 GB/s.
Collective operations are serial with the dependent computation.

Version `v2` uses `max(latency_model_seconds, wire_bytes / 900e9)` for each collective call.
The measured latency already includes data transfer, so the formula does not add that time twice.
The application NCCL total is diagnostic data only.

### Profile and byte units

The profile has 95 points from four idle, unlocked ranks.
Each rank median uses nine graph replays, with 64 calls in each replay.
Each stored point is the median of those four rank medians.
The raw SHA256 is `c6b55ee5354c7ef6fb58606cca9268fcef90bc535935d4f07750029a81506e49`.
The profile contains no FP32 all-to-all points.

Profile lookup uses input bytes on each rank.

All-gather wire calculation uses the complete output bytes on each rank.
The latency model first takes a cumulative maximum over the measured points, then uses linear interpolation.
Below and above the measured range, it holds the nearest endpoint value.
It does not extrapolate a latency slope or create measured points.

Wire time still increases with bytes beyond that range.

The profile digest includes the raw digest and the current in-memory point table.
The v2 rate digest includes the profile digest and the communication rule versions.
These digests detect a changed identity. They do not prove measurement quality or match each point to the raw data.
The external raw-data check verifies all point sizes and four-rank medians.

For one TP4 decode row, the counts and byte units are:

| Operation | Calls | Input B/rank | Output B/rank | Wire B/rank/call |
|---|---:|---:|---:|---:|
| Routed product, BF16 all-gather | 40 | 6912 | 27648 | 20736 |
| Routed output, FP32 all-gather | 40 | 5120 | 20480 | 15360 |
| `wo_b`, FP32 all-reduce | 40 | 20480 | 20480 | 30720 |
| Embedding, BF16 all-reduce | 1 | 10240 | 10240 | 15360 |
| Output head, FP32 all-gather | 1 | 129280 | 517120 | 387840 |
| Engram, BF16 all-reduce | 2 | 12288 | 12288 | 18432 |

The six routed assignments are included in the product bytes.
The legacy wire formula also includes all six assignments and gives 20736 bytes per layer for this row.
Legacy reproducibility and physical byte checks are separate checks.
The 38 HTTP cases keep the exact legacy wire results.

### Index scores and ideal batching

The default index-source layers are `(2, 8, 14, 20, 24, 28, 32, 36)`.
Layers 2, 8, and 14 use ratio 2. The other five use ratio 1.

For effective context length `E`, each layer uses width `floor(E / ratio)`.
The FP32 all-reduce input is `rows * width * 4` bytes.
At `E=257` and one row, this gives three 512-byte calls and five 1028-byte calls.
Prefill and decode each include these reductions.

TEP4 adds 40 routed-output FP32 all-reduces with 20480 input bytes per row.
It has 40 dense, 40 routed, and eight index-score reductions for the default model.

The ideal lower bound combines the batch and counts each layer collective once per target call.
It reads the batch weights once.
It does not charge the 256 MiB score workspace chunks in `indexing.score_chunks`.
Those chunks can add all-reduces in the execution.

The eager index path uses `end_pos // ratio`, as this model does.
Captured graphs use `compressed_context_width()` to round the context to a bucket before division.
At 257 tokens, graph widths can be 256 for ratio 2 and 512 for ratio 1.
The corresponding eight calls carry more bytes. The lower bound excludes that padding.

The [decode request loop](../../../python/oh_my_vllm/models/deepseek_v41/inference.py#L1058)<!-- source: outputs_by_id[row["request_id"]] = self._execute_row(row) --> executes each request separately.
Its execution collective count is approximately the batch size times the ideal count.
Batch decode can remove this execution cost. The lower bound does not price it.

### DEP4 model

DEP4 has no accepted executor and no measured reference result.
Its route matrix models 40 BF16 dispatch calls and 40 FP32 combine calls per target call.
The `dep4-bf16-all-to-all-proxy-v1` rule uses the BF16 all-to-all latency table at the same byte size for each dtype.
All-to-all transfers equal data volumes at equal byte sizes, so the rule borrows latency data without changing the volume.

The microbenchmark input includes the local quarter. Route wire bytes exclude it.
The rule converts route wire bytes to profile input bytes with `ceil(wire_bytes * 4 / 3)`.
FP32 route wire bytes still use four bytes per value.

These results keep `communication_model=lower_bound`, `communication_is_modeled=True`, and `communication_is_measured=False`.
The result also stores the rule version and profile digest.
A wire calculation instead stores `communication_model=wire` and `legacy-wire-v1`.
A TEP4 fallback does not supply a DEP4 result.

## Trace contract

The runtime trace holds one record for each rank and target call.
Each record holds these counters:

| Counter | Meaning |
|---|---|
| `rank` | Rank number from 0 to 3 |
| `call` | Unique call number |
| `phase` | `prefill` or `decode` |
| `requests` | Active requests on this rank |
| `rows` | Credited target rows on this rank |
| `global_rows` | Credited target rows on four ranks |
| `logit_rows` | Credited logit rows on this rank |
| `context_end` | Last credited position on this rank |
| `accepted` | Accepted draft tokens on this rank |
| `proposed` | Proposed draft tokens on this rank |
| `sparse_pairs` | Credited attention pairs on this rank |
| `index_pairs` | Credited compressed positions on this rank |
| `engram_unique_rows` | Selected Engram rows on this rank |
| `touched_experts` | Local expert weights read on this rank |
| `layer_ids` | Complete backbone layer set |
| `route_matrix` | 4 by 4 DEP4 per-layer route counts |

`sparse_pairs` and `index_pairs` hold the credited count for this rank.
For TP4 and TEP4, divide the total pair count by four.
For DEP4, count the pairs of the owned requests.

The Engram tables use row sharding.
A rank can fetch more than one quarter of the unique rows.
Its bound is the causal lookup count.

Each TP4/TEP4 rank holds 16 of 64 attention heads and 8 of 32 index heads.
The index-score compute count follows the head partition.
The credited compute pair count is one quarter. The collective buffer is the complete score sum.
A TEP4 rank can receive more than one quarter of the routed assignments.
Its expert bound is the causal maximum for its local expert set.

The trace binding holds the source commit and the native module digests.
It holds the checkpoint config and index digests and the tokenizer.
It holds the four GPU UUIDs, strategy, effective strategy, owner plan, and sampling.
It holds the case id and the rate profile digest.
It holds the accepted and proposed counters.

The check rejects a missing rank record or a duplicate call.
It rejects an incomplete layer set or a counter above the causal limit.
It rejects a missing DEP4 route matrix or a changed identity.
It rejects trace counters that differ from the bound counters.

## Ideal plan

`reference_case` makes the plan of one approved HTTP case.
It credits all five drafts, so the plan shows the best acceptance.
The plan is a reference scale, not an efficiency result.
The benchmark CSV labels this plan with `reference_acceptance=ideal`.
A strategy fallback uses `reference_acceptance=unavailable`.
It records `ideal_reference_seconds` and `ideal_efficiency_percent`.

CSV rows also store `accounting_version`, `communication_model`, `profile_digest`, and `communication_rule_version`.
DEP4 rows include `dep4_communication_modeled` and `dep4_communication_measured`.

## Reference values for the 8192-output cases

These values use ideal acceptance.
Each DSpark5 call accepts all five drafts.
The ordinary row uses one target row for each call.
These values are reference scales, not efficiency results.

The first table uses `v1/wire` only.

| Case | Strategy | v1/wire seconds |
|---|---|---:|
| `dspark5:tp4:4:400000:8192` | TP4 | 21.230 |
| `dspark5:tp4:4:800000:8192` | TP4 | 38.833 |
| `dspark5:tp4:16:400000:8192` | TP4 | 76.849 |
| `dspark5:tp4:16:800000:8192` | TP4 | 147.261 |
| `dspark5:tp4:64:400000:8192` | TP4 | 278.512 |
| `dspark5:tp4:64:800000:8192` | TP4 | 560.159 |
| `dspark5:tep4:4:400000:8192` | TEP4 | 20.831 |
| `dspark5:tep4:4:800000:8192` | TEP4 | 38.051 |
| `dspark5:tep4:16:400000:8192` | TEP4 | 75.317 |
| `dspark5:tep4:16:800000:8192` | TEP4 | 144.200 |
| `dspark5:tep4:64:400000:8192` | TEP4 | 272.149 |
| `dspark5:tep4:64:800000:8192` | TEP4 | 547.680 |
| `dspark5:dep4:4:400000:8192` | DEP4 | 14.427 |
| `dspark5:dep4:4:800000:8192` | DEP4 | 24.240 |
| `dspark5:dep4:16:400000:8192` | DEP4 | 45.904 |
| `dspark5:dep4:16:800000:8192` | DEP4 | 85.157 |
| `dspark5:dep4:64:400000:8192` | DEP4 | 152.000 |
| `dspark5:dep4:64:800000:8192` | DEP4 | 309.011 |
| `ordinary:4:400000:8192` | TP4 | 25.891 |

The second table uses `v2/lower_bound`.
DEP4 rows use the documented model rule and have no measured executor.

| Case | v2/lower_bound seconds |
|---|---:|
| `dspark5:tp4:4:400000:8192` | 53.039 |
| `dspark5:tp4:4:800000:8192` | 154.376 |
| `dspark5:tp4:16:400000:8192` | 194.223 |
| `dspark5:tp4:16:800000:8192` | 599.706 |
| `dspark5:tp4:64:400000:8192` | 737.554 |
| `dspark5:tp4:64:800000:8192` | 2359.488 |
| `dspark5:tep4:4:400000:8192` | 51.824 |
| `dspark5:tep4:4:800000:8192` | 152.778 |
| `dspark5:tep4:16:400000:8192` | 191.558 |
| `dspark5:tep4:16:800000:8192` | 595.512 |
| `dspark5:tep4:64:400000:8192` | 728.169 |
| `dspark5:tep4:64:800000:8192` | 2343.986 |
| `dspark5:dep4:4:400000:8192` | 16.765 |
| `dspark5:dep4:4:800000:8192` | 26.578 |
| `dspark5:dep4:16:400000:8192` | 48.415 |
| `dspark5:dep4:16:800000:8192` | 87.668 |
| `dspark5:dep4:64:400000:8192` | 153.989 |
| `dspark5:dep4:64:800000:8192` | 311.000 |
| `ordinary:4:400000:8192` | 74.490 |

## Missing runtime counters

The DeepSeek runtime trace does not hold the accounting counters.
These counters are missing:

- execution trace records for ranks 1 to 3
- the case id for each call
- the accepted and proposed draft counts
- the per-rank routed assignment count and local touched expert count
- the per-rank selected attention pair count
- the per-rank scored indexer pair count
- the per-rank selected Engram row count
- the DEP4 4 by 4 dispatch and combine route matrix
- the native module digests and four GPU UUIDs in the trace

Use `reference_trace` when the runtime supplies these counters.
Do not report a measured efficiency before that trace exists.

Use the [test scope](../../testing.md) and the [reporting policy](../../testing.md#reporting).
