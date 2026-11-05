# HTTP performance measurement

Use four B200 GPUs and the full Rust service.
Select GPUs with the [GPU wrapper](../../distributed.md#gpu-leases).
Do not use GPUs with locked clocks for performance measurements.
Keep raw artifacts external to the repository.
Set `TMPDIR` to a writable directory with a short path for Unix sockets.

## Approved workloads

The approved matrix has 38 cases:

| Mode | Strategy | Batch | Input tokens | Output tokens | Cases |
|---|---|---:|---:|---:|---:|
| DSpark5 | TP4, TEP4, DEP4 | 4, 16, 64 | 400000, 800000 | 64, 8192 | 36 |
| Ordinary | TP4 | 4 | 400000 | 64, 8192 | 2 |

Input and output lengths apply to each request.
Use uncached prefixes and exact tokenizer lengths.
Use greedy text execution.
The runner sets `ignore_eos` and checks the full usage record.
No full approved matrix result is recorded.

## Run the matrix

Build the service and native modules with the [service setup](../../serving.md#start).
Set `MODEL_ROOT` to the checkpoint directory.
Set `ARTIFACT_ROOT` to a new external directory.
Run the matrix:

```bash
OH_MY_VLLM_GPU_COUNT=4 scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.deepseek_benchmark \
  --model "$MODEL_ROOT" \
  --reference-version deepseek-http-roofline-v6 \
  --artifacts "$ARTIFACT_ROOT" \
  --csv docs/models/deepseek_v41/assets/http-latest.csv
```

Use `--case` with a [matrix identifier](../../../development/serving/deepseek_matrix.py) for a subset.
Use `--cache-context` to set physical cache context capacity.
Use `OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS=32768` for long-context throughput.
Use `--request-timeout-seconds` for the HTTP client and service timeout.
The default timeout is 7200 seconds.

The timer includes submission, prompt preparation, queueing, model execution, and response parsing.
It does not include model loading and initialization.
The CSV identifies mode, strategy, batch, exact lengths, request count, and full HTTP seconds.
It also records `effective_strategy`, `reference_seconds`, `reference_acceptance`, and `efficiency_percent`.

Artifacts bind source, executable, native modules, checkpoint, tokenizer, topology, configuration, sampling, and command identities.
Harness runtime versions identify the harness, not each worker.
Reference calculation also validates four-rank runtime identities and HTTP usage.
Result JSON keeps per-rank resources, reference scope, assumptions, and diagnostics.
Structured CSV fields contain JSON.

## Select the reference

The harness default stays the historical `deepseek-http-reference-v3.1`.
Use `--reference-version deepseek-http-roofline-v6` for current efficiency acceptance.
V6 is a resource lower bound with fixed target arithmetic, placement, wire formats, and maximum overlap.
Its whole-HTTP quantities do not increase when the same accepted work is split into more calls.

Use `--reference-version deepseek-http-protocol-estimate-v5` to examine the historical packed protocol estimate.
Use `--reference-version deepseek-http-resource-floor-v4` for the conditional partial floor.
The [reference contract](accounting.md) gives formats, necessity arguments, zero-credit items, and identities.

`reference_case` supplies static scales only.
HTTP results use `reference_trace` with the bound runtime trace.
A strategy fallback has no reference value.
An ideal draft-acceptance assumption is not measured efficiency.

## Check the long-output rows

Current efficiency acceptance uses `deepseek-http-roofline-v6` and its fixed-algorithm resource roofline scope.
The check uses all 19 approved 8192-output cases, one measurement identity, and at least 33% efficiency for each row.
Run the check on completed result JSON files:

```bash
scripts/with-env.sh python -m development.serving.deepseek_acceptance "$ARTIFACT_ROOT"/*-8192.json
```

The check calculates the selected reference and efficiency again from each bound runtime trace.
Missing, duplicate, failed, mixed-identity, static, or changed-reference records cannot pass.
V3.1, v4, and v5 are ineligible for current efficiency acceptance.
Their recalculated threshold diagnostics stay available without a target pass.

Old calculator identities stay unchanged.
Old records cannot silently acquire v6 qualification.
Full protocol stage bytes are diagnostics and do not add serial seconds to the v6 floor.

Exit status is 0 for a pass, 1 for invalid or failed rows, and 2 for an ineligible reference.
A pass does not replace [numerical acceptance](evaluation.md), capacity checks, or the [AgentX workloads](agentx.md).
