# Current state and open work

Updated: 2026-10-15. Refactor work is active. Final delivery is pending.

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

Apply [contribution](../CONTRIBUTING.md) and [collaboration](collaboration.md): independent review, checks, prompt publication, assigned OMP branches, rebase, repeat checks, squash, and fast-forward.
Codex uses `main`. OMP uses a separate checkout. Continue until all work passes.

The user authorized publication of the TileFoundry fork update.
Its `main` follows upstream `e9820eb`. Its `oh-my-vllm-integration` branch matches the pinned `5b9a310` compatibility source.

## Previous source verification

Source `3324a7f` keeps attention sigmoid BF16 rounding before the BF16 product.
Full-model MTP compile passes. The gate matches Torch for all BF16 bit patterns.
Its formal 539-case matrix adds 22 gate cases to the previous 517 IDs. Non-boundary GPU passes 589 cases.
Six ordinary/MTP4 capacity cases finish 258048 input and 4096 output tokens at batches 1/2/4, without OOM or recompute preemption.

MTP and DSpark each pass six service clients and four-stage allocator OOM recovery, with worker replacement, cache reuse, and final release.
Independent review covers original requests, source identities, and cleanup.
CPU passes 1592/153. Its predecessor passes 1568/153 but fails 23 missing attention-gate fixtures. Keep original failures.
Rust/style/document checks passed before that commit.

Tutorial passes 14 Node and 9 browser tests plus its build. Bounded profiling and dark cache-hit plots have selected CPU/GPU checks.

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

[Native narrow projection](kernels/narrow-fp8.md) keeps FP32 scales and BF16 output. Review approves its orientation and source/header identities.
Twelve shapes with three rounds of twenty pairs reduce GPU latency 13.4%-18.4%, with positive confidence bounds.
CPU passes 1592/153. Non-boundary GPU passes 621, with 28 new replay/fault cases and unchanged full-model checks.
Two memchecks and all 539 operator diagnostic cases pass without interference.

The first CPU source-contract failure stays available. Corrected identities pass selected checks and the full repeat.
Ordinary 32768/4096 batch-one diagnostics use two warmups and five measurements: prefill 1.971 seconds, ratio 4.461. Decode 34.116 seconds, ratio 2.203.

Spreads pass. Latency gates fail. This dirty-source evidence is not formal acceptance.
Capacity, service, one-hour, and the complete framework matrix await final-source verification.
[Precision lessons](kernels/precision-lessons.md) reject finite history, fixed FP32 prefixes, and slow native FP64 panels.

## Service process ownership checkpoint

The [dedicated supervisor](testing/process-ownership.md) gives the engine tree an isolated process scope.
The collector's other subprocess roots do not enter that scope.
Selected CPU tests cover lost session leaders, root binding failure, partial control input, external cancellation, and cleanup error records.

The full CPU repeat passes 1604 cases and 153 subtests. Rust, style, and document checks pass.
Its predecessor fails eight fixtures after pytest reads an unrelated temporary-directory symlink.
Fixtures now use independent configuration scopes.

Independent review corrects an assertion that rejects valid barrier cancellation. Keep the original failed records.

Five external driver CPU cases pass: normal completion, engine exit, collector failure, supervisor loss, and invalid results.
They verify ownership and keep original errors. Inference, deadline-tail, and one-hour verification are pending.

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
Independent review approves these corrections. Rust checks pass.
Final-source GPU, service, and one-hour acceptance are pending.

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

OMP is correcting the fourth Issue 4 review: abort timing, optional cancellation evidence, clock correlation, and completed checks.
Startup, cleanup, sockets, attempt counts, and asymmetric hops pass selected reviews. Final-source two-API fixtures and canonical CPU evidence are pending.

HTTP fixtures use the selected test binary from separate Cargo output directories.
Its selected repeat passes 38 tests and 46 subtests, with 31 recorded launches.
Full CPU passes 1682 cases and 153 subtests. Keep the original eight failed checkout fixtures.
Final-source acceptance is pending.

The tool lock now supplies Ninja for fresh native CUDA builds.
A new external candidate build records its compiler, complete project header closure, and module hashes.
Compilation and module loading do not prove memory safety, numerical acceptance, or performance.

## Persistent precision state checkpoint

The [persistent precision path](kernels/persistent-state.md) keeps FP32 state across all-precision chunks with the same eight-lane arithmetic.
Mixed routes, BF16 boundaries, and numerical/state tolerances stay equal. Independent review approves four source/header identity updates.
Twelve memchecks, 105 selected GPU tests, and all 539 operator diagnostic cases pass without interference.
Main passes five fresh memchecks and 588 numerical tests with six subtests.

Six context cases pass. The full non-boundary set passes 2307 cases but fails one CPU timeout fixture.
The GPU-visible fixture fix passes seven tests and three subtests. The complete non-boundary repeat passes 2308/153. Original failures stay available.

Same-GPU framework collections use the same checkpoint bytes and release binary, with two warmups and five measurements each.
Separate caches pass steady-state audit. Prefill decreases 1.959 to 1.899 seconds, or 3.06%.
Decode medians are 33.970 and 33.947 seconds. Ratios 4.300/2.192 fail the latency gate. Spread checks pass.

The sixteen-lane regression and first shared-cache audit failures stay available.
Final-source numerical, fifteen-row performance, service, and one-hour acceptance are pending.

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

The [native affine diagnostics](kernels/affine-diagnostics.md) find complete-operation layout benefits.
Three variants fail all eight final-state cancellation workloads. Repository CUDA passes all eight.
Larger finite-only FP64 emulation is slower. No candidate is selected.

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
