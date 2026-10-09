# Current state and open work

Updated: 2026-10-10.

## Current implementation

Rust controls service, scheduling, and logical KV. Python controls GPU computation on one B200.
The independent runtime uses CUDA as its default backend and pinned TileLang for comparison.
Block size is 784. The target has 16 FA layers and 48 GDN layers with `mamba_cache_mode="align"`.
The total input and output limit is 262144 tokens.

Select ordinary, native MTP4, or DSpark at worker startup.
Ordinary decoding keeps FP32 GDN state. The two speculative modes keep BF16 GDN state and rollback snapshots.
Chat, Responses, streams, tools, thinking settings, constraints, cancellation, and prefix reuse stay available.

DSpark keeps fixed checkpoint weights and shares the target embedding and vocabulary head.
Its five-layer draft uses zero-based target features `(5,19,33,47,61)`, context injection, and Markov and confidence heads.
Only kept target rows enter persistent draft context. Rejected target rows roll back computed progress and GDN snapshots.

DSpark proposes zero to seven candidates. The initial cumulative confidence threshold is `0.2`.
A small text/tool-history collection with inputs different from the formal inputs supplied threshold-selection evidence. Set the threshold to `0.0` for fixed-count debugging.
Output, context, and grammar limits can decrease the proposal count at each threshold.
Greedy verification keeps a matching draft prefix and the next target token.

Stochastic verification uses full conditional distributions and `min(1,p/q)` acceptance.
Rejection sampling uses normalized `max(p-q,0)`. Full acceptance adds a bonus token from the target distribution.
Grammar masks and sampling settings apply to each speculative prefix.

Semantic IR compiles prefill, target verification, native MTP, and DSpark units.
Manual CUDA Graphs keep bounded, different cache families and pools with capture restoration and memory guards.
The DSpark CUDA kernels keep the specified BF16 rounding and long-position YaRN precision.

Environment selection uses a prefix that you give, an active environment, or repository `.venv`.
Set `OH_MY_VLLM_MODEL` or `--model`, with `OH_MY_VLLM_DRAFT_MODEL` or `--draft-model` for DSpark.
Use [development](development.md) for commands and [architecture](architecture.md) for ownership and cache contracts.

## Performance implementation in progress

The collector uses fifteen fixed workloads and per-request submission, first-token, and last-token boundaries.
Prefill and decode have different wall-time and theoretical-bound checks.
The canonical model counts required parameter traffic, shared resources, semantic dependencies, effective queries, KV ranges, states, and drafts.
Independent mathematical review and fail-closed operation checks passed. The full current-source collection stays pending.

Execution uses CUDA Graph, multiple CUDA streams, and PDL.
GDN preparation joins a different gate branch. Bounded prefill graphs restore FA and GDN writes.
Allocator topology reuse decreases diagnostic capture storage from 29.13 to 7.82 GB.
Prefill capture can start only with at least 32 GiB of free GPU memory. Replay hits keep their previous path.

Owned FP8 GEMM keeps checkpoint FP32 scales and controls its stream and launch.
Large MN gate/up uses K256 tiles, three stages, and swizzle 16. The K-major layout reference keeps K128 tiles and five stages.
Paired full-operation measurements kept outputs bitwise equal.
Mean time decreased by about 0.487 ms at 32144 and 32290 rows.
The [profiling guide](profiling.md) records counters, selected configurations, and rejected candidates.

Four affected full-operation cases passed numerical and speed checks.
The [K-tile and scheduling diagnosis](../bench/evidence/2026-10-10-fp8-k-tile-scheduling-diagnosis.json) keeps source identities, fifteen counters per launch, loaded-library hashes, and original hashes.

Five projections at 624 to 2496 rows use two-block CTA clusters and TMA multicast.
Residual RMS and GDN output fuse normalization with FP8 quantization at their specified rounding points.
Small normalized projections keep the previous CUDA chain.
GDN values use a vector copy from packed rows. FP32 recurrent batches 2 and 4 use measured row/warp tiles.

Persistent MTP and final target prefill remove unused historical attention and MLP output.
They keep required KV, boundary features, and verification samples.
Different context graphs restore captures and validate full replay metadata.
Canonical work keeps logical rows, disjoint parameter slices, and absolute MTP KV ranges.

A rounded one-token prefill can wait only after a successful prefill chunk of at least one block.
Decode-only steps and speculative verification keep positive-input progress.
The production token budget stays 32768. Forty-seven scheduler tests passed, with new and continued prefill, four/seven drafts, and checkpoint progress.

Source `9c336e9` completed eight valid full-output phase rows.
Four passed the two phases: ordinary batch 1, MTP4 batches 1 and 2, and DSpark batch 1.
Ordinary batches 2 and 4, MTP4 batch 4, and DSpark batch 2 failed prefill and passed decode.
A different MTP4 batch-2 attempt failed the cache audit. Keep that attempt and its reason.

Subsequent dirty-source scheduling diagnosis completed five full-output groups before the fairness correction.
DSpark batch 2 failed its external-process guard. These attempts do not supply current-source acceptance.
All task-owned workers from those groups exited.

The current CPU collection passed 748 tests, with 435 GPU tests deselected and 65 subtests passed.
Twenty GPU projection checks passed, with changed graph inputs and original scales, layout equivalence, and storage checks.
Rust workspace tests, format, line width, Clippy, Ruff, and document checks passed.
Tutorial checks passed ten excerpt tests and nine browser tests.

The 317 operator cases, fifteen phase rows, nine capacity cases, and full service checks stay active.
Previous dirty-source DSpark capacity results have no OOM or recompute preemption, but precede subsequent kernel changes.
[Acceptance](acceptance.md) records source limits and original attempts. Current-source capacity and service collections stay pending.

## Documents and evidence

English Markdown follows ASD-STE100 Issue 9 with full Chinese translations.
Documents and the tutorial use current project contracts.
Tracked summaries keep measurement values, source identities, original hashes, and verification limits.
Original attempts and host configuration stay in ignored local or external storage.

The user-requested tutorial preview stays on port 18084.
Ignored `LOCAL.md` holds its address, PID, and maintenance commands.

## Open work

- Finish source-contract review after subsequent implementation changes. Current selected-output and cost-model changes have independent review.
- Complete fifteen phase-latency rows, then optimize failing paths and run full affected sets again.
- Run tutorial checks after subsequent source changes.
- Run applicable numerical, operator, context, and service regressions after inference changes.
- Complete user review of the tutorial.
- Test cross-shape 262k graph eviction/recapture with measured memory pressure, as `PY-04` specifies in [audit](audit.md).
- Show bounded compiler storage when shapes change indefinitely.
- Use measurements for `PY-06` host-overlap work.
- Keep the rejected `KRN-06` candidate excluded until full-operation evidence shows a gain.

Read [requirements](requirements.md), [testing](testing.md), and [audit](audit.md).
