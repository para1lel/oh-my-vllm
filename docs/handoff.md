# Current state and open work

Updated: 2026-10-08.

## Current implementation

The independent runtime supplies Rust service/scheduling/logical KV and Python GPU execution for the target model on one B200.
CUDA is the owned default backend. TileLang stays pinned for comparison.
Semantic IR compiles prefill, target decode, native MTP4, and DSpark inside manual graph lifecycles.

Block 784 and existing features stay active. Ordinary execution keeps FP32 GDN state.
MTP4 and DSpark keep BF16 GDN state with the specified rounding and snapshot contracts.

The process selects `--speculative-mode none`, `mtp`, or `dspark` before model loading.
The existing `--num-speculative-tokens 0/4` interface stays compatible.
Ordinary execution samples one target token per decode step.
MTP4 proposes four tokens and compares them with target sampling, with at most five target queries.
It keeps the matching draft prefix and the next target token.

DSpark uses up to seven candidates and at most eight target queries.
Its five-layer draft uses zero-based target features `(5,19,33,47,61)`, context injection, and the Markov head.
Greedy verification keeps the matching draft prefix and the next target token.

Stochastic verification uses full conditional distributions and `min(1,p/q)` acceptance.
Rejection sampling uses normalized `max(p-q,0)`. Full acceptance adds a bonus token sampled from the target distribution.
Grammar masks and sampling settings apply to each speculative prefix.
Only kept rows enter persistent DSpark context. Rejected target rows roll back computed progress and GDN snapshots.

`OH_MY_VLLM_DRAFT_MODEL` or `--draft-model` supplies the optional checkpoint.
Loading checks configuration, shapes, BF16 dtype, and the loaded configuration/weight hashes.

The user can set the confidence threshold. Its initial setting is `0.2` after selection with different text/tool-history inputs.
Set it to `0.0` for fixed-count proposals up to seven. Output, context, and grammar limits can decrease the proposal count.
Confidence can stop at the first candidate and return zero drafts. That step uses one target token.

Synthetic-input diagnostics supply tuning evidence. These diagnostics are different from threshold selection and formal acceptance.

The Twine tutorial sources have thirteen chapters and a DSpark chapter.
The user-requested preview on port 18084 uses the current build, with the new result mode field and confidence setting.
Its address, PID, and maintenance commands are in ignored `LOCAL.md`.

The [acceptance index](acceptance.md) binds each historical measurement to its source.
The latest full 147-case/twelve-row performance result applies to `619c9d9`.
Later IR correctness changes passed 549 GPU tests, 70 subtests, and 308 CPU tests without a new full performance collection.

## Document and portability refactor

Project English Markdown now uses ASD-STE100 Issue 9 rules with a project technical glossary and Chinese translations.
Architecture includes design. This file includes the former plan.
Kernel development combines CUDA/TileLang guides.
The audit keeps open problems and a 48-finding fix index.

The acceptance document is the sole evidence index.
Key ADR decisions and replacement relationships stay.
Daily repair narratives stay in Git history.

Environment selection uses explicit prefix, active virtual/conda environment, or repository `.venv`.
Model selection uses `OH_MY_VLLM_MODEL` or explicit `--model` consistently across execution tools.
Rust library launch rejects an empty model before it starts a child.
Full GPU tests reject missing model configuration before GPU selection.
CPU model integrations skip explicitly. Pure validation tests stay active.

All 66 initial tracked evidence files have byte-identical ignored local copies with SHA-256 comparisons.
Tracked derived summaries keep numeric measurements, versions, and original hashes and remove host identities and raw payloads.
Formal comparators reject derived envelopes and keep the current gates.
New servers must measure their own matching baseline.
Local configuration and original evidence are ignored by Git.

## Confidence threshold selection

`calibration-03` used project text and tool history with source code, with inputs different from the formal synthetic token IDs.
Each request had 32768 input tokens and 512 kept output tokens. The batch had two requests.
Each setting used greedy sampling, ignored EOS, and a template with `thinking=off`.
Each setting had two full warmups and two measured repetitions on one B200.

| Threshold | Median output TPS | Accepted / scheduled drafts |
|---|---:|---:|
| `0.0` | 177.287 | 28.487% |
| `0.05` | 183.097 | 44.957% |
| `0.1` | 184.198 | 53.406% |
| `0.2` | 184.323 | 62.5% |

Acceptance uses the sum of accepted drafts divided by the sum of verified drafts in the measured rows.
Verified drafts are scheduled candidates. Returned next-step proposals have a different counter.

The project selects `0.2` from this small collection. Throughput at `0.1` and `0.2` is close.
This selection is for these two inputs and two measured repetitions. Formal collections must measure performance and stability on other workloads.
The logs, corpus/source hashes, and tuning records stay in external storage.

## Verification for the DSpark implementation milestone

Rust workspace tests passed 130 cases. Formatting, the 100-character limit, Clippy, and the release build passed during implementation.
Ruff formatting/checks passed for Python, scripts, benchmarks, tests, and development tools.
The latest CPU collection passed 513 tests and 70 subtests, with 285 GPU tests deselected.

Subsequent selected tests include sampling, graph families, source identity, measured intervals, and service evidence.
Selected benchmark/context/service checks passed 120 CPU tests, with nine GPU tests deselected.
Eleven external CPU supervisor contracts passed.
They also include OMP source ranges and cancellation during response persistence and signal-handler restoration.

GPU tests passed 57 new and existing compiled cases, four supplemental reference cases, and three context-graph cases.
Two native-attention bucket cases passed in eager and compiled modes with changed metadata, FP64 references, and cache canaries.
Eager and compiled DSpark smoke runs each kept 64 output tokens.
A repeated-prefix diagnostic kept a 784-token prefix hit through 100 repetitions.
These checks do not replace the full operator, context, model, or performance collections.

The first full GPU attempt passed 66 tests and four subtests, with two failed cases.
Its DSpark batch-one boundary completed 258048 input tokens and 4096 output tokens, without OOM or recompute preemption.
The validator rejected that row because the generic benchmark result omitted the mode field.
The other failure was an ordinary batch-two run interrupted when the attempt stopped.
The benchmark result records `speculative_mode`.

The second full GPU collection passed 285 tests, with 513 CPU tests deselected.
Its nine ordinary/MTP4/DSpark boundary cases passed at batches 1, 2, and 4 without OOM or recompute preemption.
The GPU runtime source stayed at implementation `46f7529`.

The first DSpark service attempt passed twelve constraint cases and the Chat lifecycle cases.
The OMP Chat client exited with status 0 after about 250 seconds.
Its original evidence validator rejected source reads that completed without error with OMP line-range suffixes.
The corrected validator finds filenames and line ranges with bounds more than zero, checks the specified `schedule()` declaration, and keeps all forwarded source-read ranges.

CPU revalidation of the original JSONL and HTTP records passed, with a manual review of the last answer.
The original failed attempt stays unchanged. A different result binds original hashes and the corrected validator hash.
This proves the client source reads and follow-up model request for the checks in the revalidation.
The first service attempt recorded source identity after startup. It is diagnostic evidence, not current-source service acceptance.

New service supervision records source/binary identity before service launch and after all client work and cleanup.
It verifies the owned listener and observed mode/threshold, keeps failures, and checks descendant/GPU/port release.
Clients save received responses before subsequent assertions.
Cancellation keeps a failed result during response persistence and signal-handler restoration.
The first service worker exited and released its GPU resources and service port.

The second service attempt passed twelve constraint cases and Chat lifecycle cases on source `46f7529`.
The OMP Chat client completed twelve tool calls and exited with status 0.
Its scheduler read stopped before the `schedule()` definition. The source-content check failed.
The task text tells the agent to read the scheduler definition and worker class, with explicit read ranges when necessary.
The validator keeps its source-content and tool-result checks.

Each attempt keeps its original records.
The second service released all owned processes, GPU resources, and its port. Its source identity stayed the same.

The third service attempt passed twelve constraint cases and the Chat and Responses lifecycle cases on source `7e2d618`.
OMP Chat passed its tool-result checks. OMP Responses completed twelve tool calls and exited with status 0.
Responses validation failed because OMP keeps `call_id|item_id` in client records and `call_id` in HTTP requests.

The validator compares those IDs with recorded tool calls, read paths, and output.
If an HTTP request has an item ID, the IDs must agree.
It rejects IDs that match more than one recorded tool call.

CPU revalidation passed for the original Responses records. The failed attempt keeps its original result.
The revalidation calculates its server-log offset after the client run.
A new service run must record the offset before the client starts.

No long-context service case ran in this attempt.
Source stayed the same. All owned processes, GPU resources, and the port were released.
Selected service checks passed 63 CPU tests.

The fourth service attempt passed twelve constraint cases on source `fb432c6`.
The shared-batch check failed for two short requests. The attempt's content, storage, cancellation, and release checks completed without error.
The test starts its two clients together and tells the model to supply longer output.
The same-batch, output content, cancellation, and release checks stay active.
Run this test with a GPU service to show shared scheduling.

Source stayed the same. The fourth service released all owned processes, GPU resources, and its port.

An earlier thirteen-chapter tutorial build passed ten Node tests, nine Playwright tests, and the Rust trace test.
Trace formatting and Clippy passed.
Browser plugin not available: the configured Playwright Chromium supplied browser verification.
Desktop/mobile screenshots, the two themes, source links, navigation, formulas, fields, and the scheduler experiment passed examination.

The last tutorial build passed ten Node tests and nine browser tests.
Its 37 source excerpt anchors and 71 source-coverage entries match current source hashes.
Desktop/mobile light/dark screenshots include the confidence formula without missing Chinese glyphs.
The user-requested static preview stays on port 18084. All smoke workers exited and released GPU resources.

Reviews by other agents checked the mode interface, checkpoint loading, sampling, precision, cache rollback, graphs, benchmarks, and service evidence.
The document hook checks pairing, protected commands, numbers, glossary definitions, links, mechanical STE rules, and host information.
Approved meanings and full translation fidelity still depend on review, as [writing rules](writing.md) specify.

Responses service acceptance, long-context service acceptance, the 230-case operator collection, and twelve-row performance collection are not complete.
The three-row paired DSpark/MTP4 comparison and nine release-binary context-boundary runs are also open.
Current-source performance still depends on a new full collection. Historical acceptance keeps its measured source and configuration.
The user directed that these DSpark comparison statistics have no acceptance gate.
The original twelve workload gates and correctness, memory, recompute, steady-state, and evidence contracts stay active.

## CUDA kernel verification milestone

The DSpark CUDA update passed 49 GPU tests, with five CPU tests deselected.
It includes 20 new layout, address, and cache-write cases.
Selected fixture, output, comparison, and provenance checks passed 35 CPU tests, with ten GPU tests deselected.
All eight repository hooks passed.

Another agent checked BF16 rounding, NeoX pairs, slot ranges, address division, and source reads before cache writes.
A diagnostic showed that different destination addresses changed append bandwidth.
The reference used less time before the address exchange. CUDA used less time after the address exchange.
The append collector now verifies different caches first and uses the same cache for timing.
The pinned reference, initial 147 case IDs, and three-round statistical gates stay the same.

The service collection on `1bcf5d5` passed constraints, Chat/Responses lifecycles, the two OMP clients, and long-context prefix reuse.
All owned workers and the service port were released. Semantic reviews record answer errors and citation limits in the external evidence.
New full collections will use the committed CUDA update.

## Device timing verification

The full operator collection on `353c94a` passed 230 numerical checks.
Two statistical speed checks failed. The append case had less CUDA time in all three rounds.
The Q/K case had more CUDA time in its first round.
The full failed collection stays in external storage.

The timer on `353c94a` recorded events with different host calls before and after graph submission.
Host submission gaps can enter the elapsed time. The recorded spikes do not identify their cause.
The new timer records external events before and after all 100 operations in each graph.

A control used 10 ms host gaps before and after replay.
The initial timer changed from 2.18576 to 202.39872 microseconds for each operation.
The new timer changed from 1.63360 to 1.64576 microseconds.
Graph examination found two event nodes at the two ends of the full chain of 100 kernel nodes.

The first diagnostic's node counter used the wrong DOT label pattern and failed its assertion.
A different validation keeps the original records and checks the graph chain and hashes.
Two GPU tests passed with 20 ms host gaps before or after replay and full graph-node checks.

Runtime files and the loaded CUDA module stay the same as `353c94a`.
Its service collection passed all six client groups, with the two OMP tool loops and long-context prefix reuse.
All service processes and the service port were released.
New formal timing will use a clean commit with the corrected collector.

## Open work

- Complete user review of the expanded tutorial and address specific teaching gaps.
- Add derived selection evidence with its text/tool-history sources, hashes, and scheduled/accepted draft counts.
- Complete the open service, operator, performance, paired-comparison, and release-binary context collections on unchanged source.
- Test cross-shape 262k graph eviction/recapture with measured memory pressure, as `PY-04` specifies in [audit](audit.md).
- Show bounded compiler storage with sustained shape changes before a broader long-lived-worker claim.
- Keep `PY-06` host-overlap work measurement-driven. The pinned readback experiment showed no material full-call gain.
- Keep the rejected `KRN-06` candidate withdrawn until new full-operation evidence shows a benefit.
- Design performance thresholds from roofline analysis instead of vLLM measurements in a new requirement change.

The existing 95% throughput, 110% TTFT, and 10% spread gates stay active.
Read [requirements](requirements.md), [testing](testing.md), and [audit](audit.md) before the next implementation task.
