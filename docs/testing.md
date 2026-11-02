# Test scope

Use the [contribution procedure](../CONTRIBUTING.md) before adding a test.
Keep a small set of behavior checks for current contracts and failure paths.
Prefer existing cases. Remove obsolete migration checks and duplicate assertions.
Agents run applicable tests and [hooks](development.md#git-hooks) for the maintainer.

## Shared checks

Rust checks service behavior, scheduling, logical state, and worker lifecycle.
HTTP service tests use the explicit Python placeholder through the process boundary.
Their timing is not inference performance.
CPU checks cover model selection, readiness, constraints, evidence identities, accounting, and manifests.

The [execution checks](../tests/test_execution.py) cover adaptive selection, preparation reuse, binding changes, and Graph resource ownership.
They check recording failure, replay stream, interruption, and closure order.
The [worker checks](../tests/test_offload_worker.py) cover the offload worker and execution boundaries.
Transfer checks cover cancellation and storage lifetime.
Use Compute Sanitizer when native memory or synchronization behavior changes.

Document-only changes use document and whitespace hooks.
They do not trigger GPU measurements or change recorded performance values.
Code changes use the relevant CPU, numerical, state, memory, and performance checks.
Do not rerun unrelated completed checks without a new concern.

## Model acceptance

Each model owns its mathematical reference, fixtures, limits, and workload definitions.
Keep frozen expected arrays and numerical limits unchanged during candidate acceptance.
Usual acceptance loads fixtures. It does not generate expected outputs.
Compare complete endpoint vectors with independent reference computation.
Do not use inference outputs as their own reference.

| Adapter | Numerical and state scope | Performance scope |
|---|---|---|
| Qwen | [Full-model endpoints](models/qwen38/evaluation.md), [operator cases](models/qwen38/operators.md), applicable MTP, media, and restoration cases | [24 HTTP cases](models/qwen38/performance.md), [AgentX](models/qwen38/agentx.md) |
| DeepSeek V4.1 | Text target, DSpark5 verification, four-rank readiness, and model-local operator contracts | [38 HTTP cases](models/deepseek_v41/performance.md), [AgentX](models/deepseek_v41/agentx.md) |

Shared runtime changes use applicable adapter regressions.
Qwen runtime delivery includes its 24 HTTP cases.
Model-local changes do not trigger another model's full matrix unless a shared path changes.
Generic descriptor or state checks do not prove numerical correctness, capacity, or performance.

The existing [draft-chain check](../development/operators/check_draft_chain.py) covers Graph sampling, RNG offsets, and history against sequential sampling.
Run GPU commands through the wrapper:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators.check_draft_chain
```

## Reporting

Keep current instructions and the latest complete results for each approved workload.
Use each model's result document and its full Chinese translation.
Replace a table only after its new matrix completes.
Do not create a report for each change or put result tables in README.
Link completed reports from README. Keep raw records and unrelated test results external.

HTTP rows identify mode, strategy, batch, input/output lengths, and measured request count.
Report full HTTP seconds, theoretical reference seconds, efficiency percentage, and draft acceptance.
Efficiency is theoretical reference seconds divided by full HTTP seconds, expressed as a percentage.
Do not include Input/s, Output/s, TTFT, or old-prefill comparisons in these tables.
Define the calculation, acceptance counters, and units with each table.
Report failures and per-rank memory separately.

Keep source, executable, native module, checkpoint, tokenizer, topology, configuration, sampling, and command identities with the results.
Use portable identifiers or artifact digests, not host paths.
Each model must have a reviewed reference calculation that counts its native formats, communication, replicated work, and host transfers.
Do not reuse another model's efficiency calculation without that review.
A theoretical resource estimate is not proof of minimum physical latency.

AgentX reports keep one section per approved concurrency and strategy.
Include output-rate distribution, prefix reuse, queueing, request status counts, profiling and drain durations, dataset identity, and observed HTTP concurrency.
Do not present incomplete requests as successful samples.
Keep raw traces and profiler records external. Approved CSV and figures belong with the model's results.

Follow the [document rules](AGENTS.md) and [delivery procedure](../CONTRIBUTING.md#delivery).
Stop owned workers and temporary services after checks.
