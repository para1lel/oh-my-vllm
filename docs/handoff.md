# Work state

Use `nightly` for development.
Read the continuation index in ignored `LOCAL.md` before work.
It gives host settings, external artifact locations, coordinator rules, and session closure status.
Use the last external records in the `LOCAL.md` index for resource and Git delivery status.

## Harness continuation

Use the [development loop](development.md#development-loop) for repository checks.
The code limits are 500 physical lines per file and 80 physical lines per function.
These limits also apply to tests, checkers, and skill helpers.
The [verification skill](../.claude/skills/verify-oh-my-vllm/SKILL.md) drives the Rust app with external evidence.
Its CPU worker fixture checks service and transport. It does not prove GPU inference, numerical correctness, or performance.

The harness work divides Rust, Python, CUDA, and tool code into smaller modules.
Use current source and binary identities for new acceptance.
The operator identities in the preceding handoff apply only to their recorded source and binary bytes.
They do not identify reorganized source or prove its acceptance.
Check moved source links and native build identities after the source freeze.

## Delivery target

The target is DeepSeek-V4.1 through the full Rust HTTP service and AgentX.
The approved HTTP matrix has 38 cases.
For each of the 19 long-output cases, `deepseek-http-roofline-v6` efficiency must be at least 33%.
The full approved HTTP matrix, the three AgentX workloads, and the 33% target are not completed.
The old TP4 AgentX records do not prove that all session trees completed.
After source changes, old acceptance records do not show acceptance for the changed source.

The coordinator prepares plans, gives instructions, and completes acceptance checks and Git delivery.
Use `opus` at `max` for all implementation work.
The local index gives the agent replacement procedure.

## Current implementation

Do not change the production source for this delivery.
Do not increase the test scope.

The production source contains the MTP proposal Graph, the M16 kernel, and the M32 prefetch change.
It also contains the HC dot and router prefill changes.
The [execution guide](models/deepseek_v41/README.md#mtp-proposal-graph) gives the MTP Graph conditions and default setting.
The [quantization guide](models/deepseek_v41/quantization.md) gives the kernel parameters and numerical contract.
Acceptance records show acceptance only for the source and binary identities in those records.

The `weight-store-v1` and grouped TMEM candidates are rejected.
The candidate that uses rawdot scheduling for more small batches is rejected.
The BF16 Compressor Graph candidate is not in the production source.
No accepted HTTP measurement shows that this candidate increases performance.
Do not restore these candidates as the default next step.

## Next work

1. Read the last coordinator status and combined source identity in `LOCAL.md`.
   Compare the production source and binary identities with that record.
   Make sure that resources are released before you give GPUs to new work.

2. Read the acceptance scope for the combined source identity in `LOCAL.md`.
   Complete only the checks that those records do not cover.
   Use the model guides for the applicable numerical, state, Graph, batch, request-owner, and capacity checks.
   A case subset or operator comparison does not replace the full-model check done independently.
   Complete the applicable acceptance checks before formal HTTP and AgentX measurements.

3. Do not change numerical order, checkpoint weights and scales, error status, fixtures, or acceptance limits.
   For capacity changes, measure cache pages, state endpoints, projection storage, and temporary storage first.
   If the source changes again, complete the applicable checks for the new source identity.

4. Complete the [HTTP matrix](models/deepseek_v41/performance.md) and [AgentX workloads](models/deepseek_v41/agentx.md).
   Select `deepseek-http-roofline-v6` because the HTTP runner uses the old calculator unless you select a different calculator.
   Use the performance guide's acceptance command for all 19 long-output rows.
   Do not replace formal HTTP rows with operator microbenchmarks.

## Guides and delivery

The [v6 calculator](../development/serving/deepseek_accounting_roofline.py) supplies the approved efficiency reference.
The [accounting guide](models/deepseek_v41/accounting.md) defines necessary communication, native formats, and parallel placement.
Use the [reporting policy](testing.md#reporting) for approved result documents.
Replace a result table only after its new matrix completes.
Keep raw records external to the repository.

Use the [architecture](architecture.md), [runtime contracts](requirements.md), and [execution rules](execution.md) for shared behavior.
Use the [DeepSeek guide](models/deepseek_v41/README.md) for model details and acceptance scope.
The [Qwen results](models/qwen38/results.md) give its last completed HTTP and AgentX results.
Use the fixed [Qwen operator cases](models/qwen38/operators.md) for applicable shared operator changes.

Obey the [contribution procedure](../CONTRIBUTING.md#delivery) before delivery.
Get a different agent to examine the documents and implementation.
Correct review errors before delivery.

The coordinator commits and pushes `nightly` before the squash into `main`.
The coordinator commits the squash on `main` and pushes `main`.
The coordinator then sets `nightly` to the same revision as `main`.
The coordinator pushes `nightly` again.
Push only `main` and `nightly`.

Stop your workers and temporary services.
Use the last external coordinator record in the `LOCAL.md` index to check session closure and GPU lease release.
