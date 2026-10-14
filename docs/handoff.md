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

Obtain independent review before each milestone commit. Codex commits on `main` and pushes reviewed milestones after checks pass.
OMP develops assigned GitHub Issues in a separate checkout and submits pull requests from its own branches.
Rebase each pull request onto current `main`, repeat applicable checks, then squash into one single-parent commit.
Update `main` only by fast-forward. Continue until all work passes.

The user authorized publication of the TileFoundry fork update.
Its `main` follows upstream `e9820eb`. Its `oh-my-vllm-integration` branch matches the pinned `5b9a310` compatibility source.

## Previous source verification

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

## Selected subgroup scan checkpoint

The selected [precision path](../python/oh_my_vllm/kernels/cuda_backend/operators/gdn_precision.cuh) keeps the token recurrence, FP32 state, and BF16 boundaries.
Its FP32 reduction order changes. Existing numerical and state tolerances stay equal.
Independent review approved the source and four identity updates in the semantic contract, with no equation or lower-bound change.

The complete CPU repeat passes 1592 cases and 153 subtests.
Selected GPU verification passes 99 cases, with the unchanged complete-model MTP compile checks.
Four added [alignment and cancellation cases](../tests/numerical/test_gdn_prefill_replay.py) pass Compute Sanitizer with zero errors.
The complete 539-case operator numerical collection passes with no interference.
Its uncommitted-source report is diagnostic and has `passed=false` with `selected_passed=true`.

The complete non-boundary GPU repeat passes 593 cases.
The clean-source formal repeat at `f05a35d` passes all 539 operator numerical cases, with complete coverage and no interference.
Capacity, service, and one-hour records belong to `3324a7f`, not this checkpoint.
Final-source acceptance and the fourth milestone are pending.
The source-bound checkpoint summary is in the [acceptance index](acceptance.md).

## Narrow FP8 checkpoint

The [native narrow projection](kernels/narrow-fp8.md) keeps FP32 block scales and the BF16 output boundary.
Independent source review approves the orientation change and the source/header identity refresh.
Twelve shapes have three rounds of twenty interleaved pairs. GPU projection latency decreases by 13.4%–18.4%, with positive confidence lower bounds.

The full CPU repeat passes 1592 cases and 153 subtests. The full non-boundary GPU repeat passes 621 cases.
The GPU suite contains 28 new critical replay/fault cases and the unchanged full-model compiled-unit checks.
Two selected replay cases pass Compute Sanitizer with zero errors.
All 539 operator diagnostic cases pass, with full coverage and no interference.

The first CPU attempt fails one stale source-contract check. Its original record stays available.
The reviewed identity refresh passes the selected checks and full repeat.

The ordinary 32768-input/4096-output batch-one diagnostic collects two warmups and five measurements.
Prefill is 1.971 seconds with ratio 4.461. Decode is 34.116 seconds with ratio 2.203.
The spread checks pass. The two latency checks fail. This is uncommitted-source evidence, not formal acceptance.

Capacity, service, one-hour, and the full framework matrix await final-source verification.

Further [precision-path lessons](kernels/precision-lessons.md) reject finite history, fixed direct FP32 prefixes, and slow native FP64 panels.
The CPU rounding counterexample rejects an ideal-arithmetic-only residual guard. No GDN candidate from these experiments is selected.

## Service process ownership checkpoint

The [dedicated supervisor](testing/process-ownership.md) gives the engine tree an isolated process scope.
The collector's other subprocess roots do not enter that scope.
Selected CPU tests cover lost session leaders, root binding failure, partial control input, external cancellation, and cleanup error records.

The full CPU repeat passes 1604 cases and 153 subtests. Rust, style, and document checks pass.
The previous full repeat fails eight fixture collections after pytest reads an unrelated temporary-directory symlink.
The fixtures now keep each comparison collection in its own configuration scope. The original failed record stays available.

Independent review found a test assertion that rejected a valid barrier cancellation.
The assertion is corrected. The original failed record stays available.

Five external driver CPU cases pass: normal completion, engine exit, collector failure, supervisor loss, and an invalid collector result.
These checks verify process ownership and keep original errors. They do not verify inference or a one-hour collection.
Deadline-tail evidence and final-source one-hour acceptance are pending.

A later external CUDA candidate failed with an illegal memory access. It was rejected and did not enter production.
The [safety guide](kernels/cuda-safety.md) records the reviewed local scale-copy defect and verification limits.
Keep that failed candidate off all devices until a new reviewed implementation meets the candidate procedure.

## Deadline evidence checkpoint

The [deadline-tail rules](testing/deadline-tail.md) keep checks for successful SSE streams and usage unchanged.
New control timestamps bind original RPC bytes, acknowledgements, the native aborted result, and its SDK export.
Server-completed/client-aborted tails have a different count and do not enter scatter plots.
Verification fails if proof is missing, changed, or does not bind one request. Historical collection results stay unchanged.

At the deadline checkpoint, the full CPU repeat passes 1658 cases and 153 subtests.
Critical checks also cover subprocess RPC capture, output budgets, and more than one aborted request.

The proxy records the disconnect's wall and monotonic clocks before handler cleanup.
An early disconnect, changed clock offset, or missing timestamp fails deadline attribution.
Independent review approves these corrections. Final-source GPU and service acceptance are pending.
Rust checks pass. Final-source one-hour verification is pending.

One-hour collections must start only after all fifteen framework performance rows pass.
The next workload translates the same full English book from Project Gutenberg in four isolated sandboxes.
OMP must run inside each sandbox with no external network or GPU access.
Book download is completed. Workload and local-socket transport adaptation are pending.

OMP implements this work on `omp/issue-4-translation-soak`. Codex owns architecture, review, and end-to-end acceptance.

The official `deepseek/deepseek-flash` collaborator passes two tool-call rounds against the official API in one native session.
Credentials stay in the user store. Host configuration stays in ignored `LOCAL.md`.

There is no API cost cap. New OMP Issue tasks cannot start during official peak pricing in `Asia/Shanghai`.
Existing tasks can finish across a boundary. Codex handles new work during peak periods.

DSH tuning is stopped at the user's request. Its owned sandbox resources are released.
No candidate from that round is selected.

## Collaborator dispatch checkpoint

The [collaboration procedure](collaboration.md) records assigned Issue branches, independent review, rebase, and squash integration.
Codex publishes reviewed milestones promptly. GitHub permits squash merges only.
Host proxy and provider configuration stay in ignored `LOCAL.md` and the native credential store.

The dispatcher uses the reviewed supervisor from the primary checkout when OMP works on an older branch.
Scope and pricing checks run again after GitHub lookup. Native session records bind task continuation.
The supervisor audits descendant release after finite tasks. Service shutdown still must use the controller's stop command.

Independent review approves the dispatcher. Its 35 targeted CPU cases pass.
They cover launch-record failure, cleanup errors, an older checkout, early leader exit, and peak boundaries.
These checks do not verify GPU inference, performance, or a one-hour collection.

OMP's first translation-workload implementation is awaiting Codex review on Issue 4.
Verify its reported checks and isolation end to end before integration.

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

The formal ordinary 32768-input/4096-output batch-one attempt on `f05a35d` collects two warmups and five measurements without OOM or preemption.
Median prefill is 1.960 seconds against a 0.442-second lower bound, a ratio of 4.436.
Median decode is 35.918 seconds against a 15.484-second lower bound, a ratio of 2.320.
The two phase spreads pass 10%. The two latency ratios fail the two-times gate.

The theoretical equations are unchanged. The complete fifteen-row final-source framework matrix is pending.
External CUDA candidates are diagnostic experiments. They are not selected production changes or acceptance evidence.
Continue tuning without weakening numerical, state, workload, or theoretical contracts.

Primary-source [algorithm research](kernels/optimization.md) and reproduced [measurement guidance](profiling/measurement.md) document the next optimization directions.
Current scan/GEMM counters, isolated FP64 emulation, and three-layer wavefront probes have source-bound diagnostic summaries.
Eager FP64 emulation is slower for the tested shapes. Token slicing lacks complete equivalence proof.
Neither candidate is selected.

Later [native affine diagnostics](kernels/affine-diagnostics.md) find complete-operation layout benefits but final-state cancellation failures.
Three affine variants fail all eight cancellation workloads. Repository CUDA passes all eight.
Larger finite-only FP64 emulation is also slower. Continue precision-path work without selecting these candidates.

The instrumented ordinary model recomputes route flags on 48 GDN layers for one repeated-token input.
Of those layers, 46 use the precision route for all chunks. Other prompt distributions are unmeasured.

After selection, refresh source identities and run affected checks, complete final-source acceptance, generate collected request plots, and obtain independent review before delivery.
See the [acceptance index](acceptance.md) for collection scope.
Obsolete working-tree reports were removed. Git history keeps their original results.

## Active exception and cleanup

The user-requested CPU tutorial preview stays active. Its host address and process birth identity are in ignored `LOCAL.md`.
Stop task-owned GPU workers, children, and temporary inference listeners after each run.
Agent tests use new external workspaces in disposable sandboxes. Remove copied tools, generated files, containers, and owned images afterward.
Keep raw measurements and temporary tuning in external storage.
