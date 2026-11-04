# HTTP performance measurement

Use four B200 GPUs and the full Rust service.
Select GPUs that have no locked clock.
Keep raw artifacts in a separate location.
Set `TMPDIR` to a directory with write permission.
The directory must have a short path for Unix sockets.

The approved matrix has 38 cases.

| Mode | Strategy | Batch | Input tokens | Output tokens | Cases |
|---|---|---:|---:|---:|---:|
| DSpark5 | TP4, TEP4, DEP4 | 4, 16, 64 | 400000, 800000 | 64, 8192 | 36 |
| Ordinary | TP4 | 4 | 400000 | 64, 8192 | 2 |

Input and output lengths apply to each request.
Use uncached prefixes and exact tokenizer lengths.
Use greedy text execution until the model sampling contract is extended.
The runner sets `ignore_eos` and checks the full usage record.

Run the matrix with:

```bash
OH_MY_VLLM_GPU_COUNT=4 scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.deepseek_benchmark \
  --model "$MODEL_ROOT" \
  --artifacts "$ARTIFACT_ROOT" \
  --csv docs/models/deepseek_v41/assets/http-latest.csv
```

Use `--case` with the identifier from `development/serving/deepseek_matrix.py` for a subset.
Set `OH_MY_VLLM_DEEPSEEK_CACHE_CONTEXT` through the runner's `--cache-context` option.
Set `OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS=32768` for long-context throughput.

Use `--request-timeout-seconds` to set the same timeout for the HTTP client and service.
The default is 7200 seconds. The measurement identity records the selected value.
The runner records source, executable, native modules, checkpoint, tokenizer, topology, and configuration identities.
It also records the harness Python, PyTorch, CUDA, NCCL, and numerical settings.
These settings do not independently identify each worker's runtime.

The full timer includes request submission, prompt preparation, queueing, model execution, and response parsing.
It excludes model loading and initialization.
The CSV records mode, strategy, batch, exact lengths, request count, full seconds, and output rate.
It also records `effective_strategy`, `reference_seconds`, `reference_acceptance`, and `efficiency_percent`.

## Optional v4 accounting

The default is `deepseek-http-reference-v3.1`.
Add `--reference-version deepseek-http-resource-floor-v4` to select the reviewed v4 calculator.
Its result is a **conditional partial resource floor**.
It does not prove a complete roofline, resident capacity, or a floor for all algorithms with the same output.

The target matrix arithmetic, numerical order, formats, parameter placement, and replicated projections stay fixed.
The MoE wire credit has these necessary conditions:

- Output shards stay fixed, with no extra weight replication or projection recomputation.
- Each credited delivery uses complete BF16 rows.
- There is no unmodeled value compression or cross-row value reuse.

For quantized transfer, reuse, or resharding, get a new review and identity.
HBM and host strict credit is zero.
Retention and fresh-transfer proofs are pending.
Embedding, `wo_b`, vocabulary-head, index, Engram, and DEP4 dispatch/combine wire have zero strict credit.
MTP proposals, rejected or repeated work, padding, state copies, scheduler overhead, empirical latency, `S/2`, and nonlinear scalar estimates receive zero credit.
Weight unions and `direct_protocol_*` traffic are diagnostics only.

The harness validates four-rank runtime identities and HTTP usage before v4 calculation.
`source_trace_version` and `source_trace_rate_profile` identify the v3.1 source binding.
`accounting_version`, `rate_profile`, and `assumptions_sha256` identify the selected calculator.
The identity manifest, result JSON, CSV, and reference event record its scope, conditions, zero-credit items, and diagnostics.
CSV structured fields contain JSON.
The result JSON also keeps per-rank resource values.

`reference_case` supplies static scales only.
The HTTP path uses `reference_trace`.
This option does not prove frozen numerical, GPU, HTTP, or 33% acceptance.
The 33% target is unchanged for each of the 19 long-output rows.

## Packed project reference and 19-row check

Select `--reference-version deepseek-http-protocol-estimate-v5` for the packed-aware project reference estimate.
The default stays v3.1.
Reference values for the old versions do not change.
The v5 reference uses BF16 window rows, FP4 compressed rows with G16 scales, and FP4 index rows with G32 scales.
Shared physical history and per-consumer traffic have separate counters.
Index workspace decoding is charged once per physical source and call.

Collective stages follow their data dependencies.
TP4 activation delivery is between expert up and down projections.
DEP4 dispatch is before expert computation, and reverse FP32 combine is after it.
Each direction is counted separately.

Routing metadata and count exchanges are protocol diagnostics, not extra reference credit.
The reference keeps the stated resource and two-microbatch relaxation.
Serial protocol seconds and the unchanged v4 partial floor are separate diagnostics.

This project reference is not a universal physical minimum.
Cache retention, padding, launch latency, overlap, and MTP work are not complete execution-time models.
The run identity records the batch-decode and Graph settings from each rank.
The Graph settings include the graph count limit.
These settings do not identify each step's capture or Eager fallback.

Check completed result JSON files with:

```bash
scripts/with-env.sh python -m development.serving.deepseek_acceptance "$ARTIFACT_ROOT"/*-8192.json
```

The check recomputes the selected reference and percentage from the bound runtime trace.
The check uses all 19 long-output cases, one measurement identity, and at least 33% for each row.
Missing, duplicate, failed, mixed-identity, static, or changed-reference records cannot pass.
The v4 partial floor is ineligible for this full project-reference check.
The old v3.1 history format is ineligible for current packed results.

Exit status is 0 for a pass, 1 for invalid or failed rows, and 2 for an ineligible reference.
A pass does not replace numerical, capacity, or AgentX acceptance.

No complete approved matrix result is recorded yet.
The ideal plan credits all five drafts. It is not a measured efficiency result.
A strategy fallback has no reference value.
Use the [reference calculation](accounting.md) for the trace contract and the missing counters.
The session-only SOTA target does not become a repository performance gate.
