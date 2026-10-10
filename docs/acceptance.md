# Acceptance evidence index

Use [requirements](requirements.md) for gates and [testing](testing.md) for collection commands.
Each result applies to its stated source, configuration, and workload.
Do applicable collections again after inference changes.
Originals supply formal verification. Portable summaries give results for reading and offline analysis.

## Current collection

The [source-bound record](../bench/evidence/2026-10-10-current-source-acceptance.json) uses source `fabcede8c6e0e6e4fa2c90d5d97b1f98d9c7dfdb`.
The release binary SHA-256 is `4676eb10dde92d0349213980d618bc597543cf3e83769b9aa5cf504c5786f77d`.
The record keeps original hashes, measurements, failed attempts, and reviews by different agents.
The final collection-scope and document change keeps the measured inference sources and selected CUDA configurations.

### Phase latency

All fifteen workloads passed their active prefill and decode checks.
Each workload has two full warmups and five complete measurements, with 4096 output tokens per request.
Prefix-hit rows reuse 32144 tokens. Other rows use cold prefixes.
All measured requests have full output and zero recompute preemptions.

Each row comes from one complete, continuous workload collection.
The rows use the same measured source, binary, environment, capacity, and formal contract.
Failed or canceled parent collections keep their original failed status.
No row combines samples from different attempts.

The [README table](../README.md#measured-performance) gives prefill seconds, lower-bound ratios, decode TPS, and theoretical percentages.
Prefix-hit prefill records its ratio without a latency gate. Its spread and decode gates stay active.
Independent review checks execution records, trace arithmetic, source identities, statistics, and closed workers.
The evidence records the scope of additional model replay separately.

### Operator matrix

All 317 unique canonical cases passed numerical and speed checks in one complete collection.
The matrix keeps the original 230 cases and reference pins.
A different agent calculated 951 round medians and all confidence bounds again from 19020 pairs.
Each confidence calculation uses 10000 bootstrap samples.
The calculated results are equal to the original results.

Each timed graph contains 100 full operations. Each sample replays that graph once.

The review checks complete outputs, modified state, dispatch, loaded libraries, build inputs, and source headers.
It also checks the installed template tree and pinned comparison sources.
Two interrupted operator attempts keep their failed status.

### Context capacity

The six ordinary/MTP4 cases passed input 258048 and output 4096 at batches 1, 2, and 4.
They have no OOM or recompute preemption. Total length is 262144.
Each case uses zero warmups and one cold repetition. Compilation and capture can occur during capacity collection.
Capacity timing and memory observations do not supply statistical phase-performance results.

| Mode | Batch | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---:|---:|---:|
| ordinary | 1 | 120.625252 | 145.320312 |
| ordinary | 2 | 120.625252 | 145.625000 |
| ordinary | 4 | 121.479034 | 148.457031 |
| mtp4 | 1 | 117.010298 | 143.238281 |
| mtp4 | 2 | 117.010481 | 145.591797 |
| mtp4 | 4 | 119.753355 | 146.867188 |

The six-row summary derives from complete raw logs in a failed larger collection.
Its independent review checks each log, full output, memory, loaded modules, and worker exit.
The larger collection keeps its failed status. It has no completed collector aggregate.
DSpark context boundary tests and their acceptance requirement are removed.
DSpark keeps phase, numerical, operator-speed, and service acceptance.

The target loader has no in-process weight-digest record.
The original collector checks GPU exclusivity once per second and after exit.
Interference fully between polls cannot be excluded.

### Service and Python tests

MTP4 and DSpark each passed the two service APIs, twelve constraints, cancellation, mixed batches, and long-prefix reuse.
The independent review compares 48 completed MTP4 requests and 46 completed DSpark requests with client records.
Four oh-my-pi tasks read source, forward tool results, continue generation, and produce answers that use those results.
The review checks 54 original artifact hashes and 1834 conditions.

Each suite has four long-prefix requests with input 131099.
Cached-token counts are zero for the first request and 130928 for the other requests.
These are functional observations. The performance and capacity collections supply their own acceptance.
Normal shutdown audits passed. Owned workers exited and temporary service ports were released.

The proxy omits complete upstream SSE bodies. It keeps follow-up requests, tool results, reply identities, and answer hashes.
Source reads can contain outline elisions. Generated line citations are approximate.
The review did not reread complete model weights or the installed template tree for service collection.

The GPU suite passed 436 tests with maximum-context cases deselected.
The current CPU suite passed 760 tests and 94 subtests, with 442 GPU tests deselected.
Rust workspace tests, Clippy, formatting, line-width checks, Ruff, and document checks passed.
The tutorial build, ten excerpt tests, and nine browser tests passed.
[Audit](audit.md) keeps cross-shape graph-memory and host-overlap work.

## Selected CUDA implementation

The native module uses ten operator headers. All owned CUDA backend code files meet the 800-line limit.
Header content enters build keys, load provenance, formal contracts, and profiler checks.
The clang-format hook uses version 23.1.3.

Five FP8 projections at 624 through 2496 rows use an owned paired TMEM accumulator and a direct BF16 epilogue.
The full current operator collection verifies the selected implementation.
The [accumulator diagnosis](../bench/evidence/2026-10-10-fp8-accumulator-observations.json) keeps its earlier measurements and profiler records.
[Profiling](profiling.md) gives selected configurations, rejected candidates, and measured optimization results.
The [owned CUDA audit](../bench/evidence/2026-10-10-owned-cuda-audit.json) records source checks and external tuning candidates.
Those candidates supplied no new production selection during cleanup.

## DSpark threshold selection

DSpark keeps fixed weights and shares the target embedding and vocabulary head.
Its initial cumulative confidence threshold is `0.2`. Set it to `0.0` for fixed-count diagnostics.
Output, context, and grammar limits still apply.

The [threshold-selection record](../bench/evidence/2026-10-08-dspark-confidence-selection.json) uses inputs different from the formal synthetic workload.
It has two text/tool-history requests, with input 32768 and output 512 per request.
Each setting has two full warmups and two measured repetitions.
Thresholds `0.0`, `0.05`, `0.1`, and `0.2` have TPS medians 177.287, 183.097, 184.198, and 184.323.
Their acceptance rates are 28.487%, 44.957%, 53.406%, and 62.500%. These medians apply to this small collection.
