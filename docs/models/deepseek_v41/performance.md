# HTTP performance measurement

Use four B200 GPUs and the full Rust service.
Select GPUs with the [GPU wrapper](../../distributed.md#gpu-leases).
Performance measurements exclude GPUs with locked clocks.
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
  --artifacts "$ARTIFACT_ROOT" \
  --csv docs/models/deepseek_v41/assets/http-latest.csv
```

Use `--case` with a [matrix identifier](../../../development/serving/deepseek_matrix.py) for a subset.
Use `--cache-context` to set physical cache context capacity.
Use `OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS=32768` for long-context throughput.
Use `--request-timeout-seconds` for the HTTP client and service timeout.
The default timeout is 7200 seconds.

The timer includes submission, prompt preparation, queueing, model execution, and response parsing.
It excludes model loading and initialization.
The CSV identifies mode, strategy, batch, exact lengths, request count, and full HTTP seconds.
It also records `effective_strategy`, `reference_seconds`, `reference_acceptance`, and `efficiency_percent`.

Artifacts bind source, executable, native modules, checkpoint, tokenizer, topology, configuration, sampling, and command identities.
Harness runtime versions identify the harness, not each worker.
Reference calculation also validates four-rank runtime identities and HTTP usage.
Result JSON keeps per-rank resources, reference scope, assumptions, and diagnostics.
Structured CSV fields contain JSON.

## Select the reference

The default is `deepseek-http-reference-v3.1`.
Use `--reference-version deepseek-http-protocol-estimate-v5` for packed protocol estimates.
Use `--reference-version deepseek-http-resource-floor-v4` for the conditional partial resource floor.
The [reference contract](accounting.md) defines their formats, credit rules, limitations, and identities.

`reference_case` supplies static scales only.
HTTP results use `reference_trace` with the bound runtime trace.
A strategy fallback has no reference value.
An ideal draft-acceptance assumption is not measured efficiency.

## Check the long-output rows

Full packed acceptance uses the packed protocol estimate.
The check uses all 19 approved 8192-output cases, one measurement identity, and at least 33% efficiency for each row.
Run the check on completed result JSON files:

```bash
scripts/with-env.sh python -m development.serving.deepseek_acceptance "$ARTIFACT_ROOT"/*-8192.json
```

The check calculates the selected reference and efficiency again from each bound runtime trace.
Missing, duplicate, failed, mixed-identity, static, or changed-reference records cannot pass.
The conditional partial floor is ineligible for full packed acceptance.

Exit status is 0 for a pass, 1 for invalid or failed rows, and 2 for an ineligible reference.
A pass does not replace [numerical acceptance](evaluation.md), capacity checks, or the [AgentX workloads](agentx.md).
