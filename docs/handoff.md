# Current state and open work

Updated: 2026-10-14. Refactor work is active. Final delivery is pending.

## Accepted scope

Keep single-B200 Qwen3.8-27B-FP8 inference, Rust service/scheduler/logical cache, Python GPU ownership, and ZMQ DEALER/msgpack.
Keep block size 784, 16 FA layers, 48 GDN layers, `mamba_cache_mode="align"`, and 262144 total tokens.
Keep API behavior, numerical tolerances, ordinary/MTP4/DSpark, and unchanged DSpark weights.

Use native CUDA libraries through repository CUDA. Remove external Python operator providers and project comparison dependencies.
Keep independent numerical/state contracts. Framework phase latency must meet the two-times lower-bound gate, with the prefix-hit prefill exception.
Keep the theoretical model, two warmups, five repetitions, and 10% spread gates.

## Milestones

1. Document hierarchy, source anchors, word limits, and definition spacing: complete and independently reviewed.
2. Cohesive code modules and critical test layers: complete and independently reviewed.
3. Native CUDA provider migration, independent references, TileFoundry rebase, and harness adaptation: complete and independently reviewed.
4. Profiler, request plots, performance tuning, and complete acceptance: pending.

Obtain independent review before each milestone commit. Commit on `main` and continue until all work passes.
Do not push or open pull requests.

## Current source verification

Source `3324a7f` keeps the attention sigmoid's BF16 rounding before its BF16 product.
See [attention gate](../python/oh_my_vllm/ir/attention_gate.py).
The unchanged full-model MTP compile check passes after this correction.
The gate matches Torch for all BF16 gate bit patterns.

The formal operator matrix passes all 539 cases with complete coverage and no interference.
It adds 22 gate cases to the unchanged previous 517 case IDs.
The complete non-boundary GPU suite passes 589 cases.
Six ordinary/MTP4 capacity cases at batches 1, 2, and 4 finish 258048 input and 4096 output tokens per request with zero OOM and recompute preemption.

MTP and DSpark each pass six service clients: constraints, Chat/Responses lifecycle, long context, and Chat/Responses agentic interaction.
Each mode passes allocator OOM recovery at four stages, with worker replacement, cache reuse, and final release.
These collections have independent reviews of original requests, source identities, and cleanup records.

The complete CPU repeat passes all 1592 cases and 153 subtests.
The previous run passed 1568 cases and 153 subtests but failed 23 fixture control-flow cases.
All failures identified the missing attention-gate configuration in the fixture inventory.
The repair adds that configuration without a production implementation change.
The original failed record is kept.

Required Rust/style/document checks passed before `3324a7f`.
The current tutorial repeat passes 14 Node tests and 9 browser tests. Its build passes after source coverage updates.
The default-off bounded Torch profiler and dark cache-hit request plots have selected CPU/GPU verification.

## One-hour evidence

MTP4 and DSpark each collect four persistent OMP goal sessions for one hour in new disposable sandbox workspaces.
The measured window has no intermediate drain or model warmup.
The same service and worker process identities have zero restart and OOM.
The four sandboxes and their owned image in each collection are deleted.

DSpark's collector reports operational completion. Independent reviews verify each mode's framework continuity, sampled resource bounds, normal streams, and final release.
MTP's collector reports an SSE audit failure on its final client-aborted request, whose generation finished at the collection deadline.
Its failed record is kept. Independent review binds the tail abort to native RPC and SDK deadline evidence.
The two external drivers report an unproved adopted-process cleanup error after service shutdown.

The recorded service and worker identities have exited, and temporary listeners and GPUs are released.
The missing adopted-process identity record is a verification limit.

Agent task completion, goal transitions, and workload journal diagnostics are not inference-framework gates.
Original collection summaries keep their results. Separate reviews must bind the unchanged raw evidence.
Source-bound request figures and a portable evidence summary are in the [acceptance index](acceptance.md).
Worker RSS increases during collection. The evidence does not prove a memory plateau or complete allocation attribution.

## Performance and next work

The formal ordinary 32768-input/4096-output batch-one attempt on `3324a7f` collects two warmups and five measurements without OOM or preemption.
Median prefill is 2.394 seconds against a 0.442-second lower bound, a ratio of 5.419.
Median decode is 36.055 seconds against a 15.484-second lower bound, a ratio of 2.329.
The two phase spreads pass 10%. The two latency ratios fail the two-times gate.

The theoretical equations are unchanged. All fifteen final-source framework rows are pending.
External CUDA candidates are diagnostic experiments. They are not selected production changes or acceptance evidence.
Continue tuning without weakening numerical, state, workload, or theoretical contracts.

After selection, refresh source identities and run affected checks, complete final-source acceptance, generate collected request plots, and obtain independent review before delivery.
See the [acceptance index](acceptance.md) for collection scope.
Obsolete working-tree reports were removed. Git history keeps their original results.

## Active exception and cleanup

The user-requested CPU tutorial preview stays active. Its host address and process birth identity are in ignored `LOCAL.md`.
Stop task-owned GPU workers, children, and temporary inference listeners after each run.
Agent tests use new external workspaces in disposable sandboxes. Remove copied tools, generated files, containers, and owned images afterward.
Keep raw measurements and temporary tuning in external storage.
