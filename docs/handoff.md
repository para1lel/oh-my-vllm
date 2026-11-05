# Work state

Use `nightly` for development.
Read the continuation index in ignored `LOCAL.md` before work.
It gives host settings, external artifact locations, coordinator rules, and session closure status.

## Delivery target

The target is DeepSeek-V4.1 through the full Rust HTTP service and AgentX.
The approved HTTP matrix has 38 cases.
For each of the 19 long-output cases, `deepseek-http-roofline-v6` efficiency must be at least 33%.
The full approved matrix and the 33% target are not completed.
Operator measurements and old functional records do not show acceptance for this source.

The hard deadline is 2026-11-06 at 14:00 in `Asia/Shanghai`.
Start final checks and delivery preparation at 11:40 on that date.
The coordinator prepares plans, gives instructions, and accepts work.
Use `opus` at `max` for all implementation work.
The local index gives the agent replacement procedure and authorized Git sequence.

## Current implementation

The coordinator accepted these operator candidates for their recorded source and binary identities:

- The `weighted_heads` tail-width path with SHA256 prefix `52b2ac1a`.
- Dense FP8 projection with narrow output, `N=512` and `K=5120`.
- The initial grouped FP4 winner with a `128x128` tile.
- Dense FP8 `Compact` projection with `N=2304` and `K=5120`.
- The full DeepSelect BF16 `count512` top-k implementation in the DeepSeek Indexer.
- Sparse FP4 pair dequantization, with the Graph test correction for two global flags.

A different agent examined the sparse test correction. The coordinator accepted it.
Its production source and binary identities did not change with that correction.
No agent has independently completed full-model acceptance of any of these operator changes with the frozen reference.

The grouped TMEM candidate is rejected.
The repository again has the initial grouped winner.
The subsequent `weight-store-v1` candidate changes the bit3/bit4 thread permutation.
It is only in external storage, with a CPU build, no GPU work, and no acceptance by a different agent.
Its owner handoff, patch, binary identity, and closure record are in the local continuation index.

## Next work

1. Read the last coordinator closure status in `LOCAL.md`.
   Compare the accepted sparse patch with its final identity.
   Compare the grouped winner identity with the local record.
   Make sure that resources are released before you give GPUs to new work.

2. Use the saved `weight-store-v1` entry for the next short grouped comparison.
   First, compare literal outputs calculated independently.
   Then compare the frozen FP32 vectors.

   Then measure the full operation with the existing short procedure.
   Use separate processes for different projection binaries.
   Do not generate new expected arrays for this comparison.

3. Do not repeat rejected TMEM, old QK fusion, or the old custom top-k CDF experiments.
   Do not replace implementation work with more collection or supervision tools.
   Use short comparisons to select changes that improve performance before long HTTP or AgentX measurements.

4. Do not change numerical order, checkpoint weights and scales, error status, fixtures, or acceptance limits.
   For capacity changes, measure cache pages, state endpoints, projection storage, and temporary storage first.
   Use the model guides for the applicable numerical, state, Graph, batch, request-owner, and capacity checks.

5. After implementation changes that improve performance, run the full [frozen-reference procedure](models/deepseek_v41/evaluation.md).
   A case subset or operator comparison does not replace the full-model check done independently.
   Complete the applicable acceptance checks before formal HTTP and AgentX measurements.

6. Complete the [HTTP matrix](models/deepseek_v41/performance.md) and [AgentX workloads](models/deepseek_v41/agentx.md).
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
Stop your workers and temporary services.
Use the final coordinator record to make sure that sessions are closed and GPU leases are released.
