# Current state and open work

Updated: 2026-10-09.

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
Prefill and decode have separate wall-time and theoretical-bound checks.
The model counts parameter traffic, shared execution resources, and semantic dependencies.

The execution trace records effective queries, KV lengths, state slots, and draft work.
GPU event intervals stay diagnostic.
The canonical model has mathematical review by another agent and fail-closed schedule checks.
The full fifteen-row collection is pending.

Execution uses CUDA Graph, multiple CUDA streams, and PDL.
Owned group-scaled FP8 GEMM keeps checkpoint FP32 scales and controls its stream and launch.
GDN preparation joins a separate gate branch. Bounded prefill graphs restore FA and GDN writes.
Allocator topology reuse decreases diagnostic prefill capture storage from 29.13 to 7.82 GB.

Large GEMM, convolution, quantization, and gated RMS dispatch use measured candidates.
[Profiling](profiling.md) records choices, counters, and rejected candidates.
The MLP gate/up path uses one operation for residual RMS and FP8 quantization, with the previous GEMM. One row uses the previous CUDA chain.
Its twenty GPU checks passed, with outputs equal to the previous chain, changed graph inputs, and rejected storage.
The extended operator matrix keeps all 230 previous cases and adds sixteen fused and twenty-eight other FP8 projection cases.

Offline registration shares the output budget with Python. Proposers reserve the scheduler's bonus token.

The current CPU collection passed 687 tests and 65 subtests, with 365 GPU tests deselected.
The affected GPU collection passed 104 tests.
Tutorial validation passed ten excerpt tests and nine browser tests.

Short-output diagnosis uses input 32768 and output 16. Its latest ordinary prefill median is about 1.39 s.
This interval still exceeds three times the current bound of about 0.453 s.
Formal performance uses output 4096. Full operator, boundary, and service regressions are pending.

The [acceptance index](acceptance.md) keeps operator, boundary, paired, and service measurements with their source scope.
The 274 operator cases and nine 262144-token boundary cases stay active.
Complete service verification stays active.

## Documents and evidence

English Markdown follows ASD-STE100 Issue 9 with full Chinese translations.
Documents and the tutorial use current project contracts.
Tracked summaries keep measurement values, source identities, original hashes, and verification limits.
Original attempts and host configuration stay in ignored local or external storage.

The user-requested tutorial preview stays on port 18084.
Ignored `LOCAL.md` holds its address, PID, and maintenance commands.

## Open work

- Finish source-contract review after subsequent implementation changes. The current fusion has independent review.
- Complete fifteen phase-latency rows, then optimize failing paths and run full affected sets again.
- Run tutorial checks after subsequent source changes.
- Run applicable numerical, operator, context, and service regressions after inference changes.
- Complete user review of the tutorial.
- Test cross-shape 262k graph eviction/recapture with measured memory pressure, as `PY-04` specifies in [audit](audit.md).
- Show bounded compiler storage when shapes change indefinitely.
- Use measurements for `PY-06` host-overlap work.
- Keep the rejected `KRN-06` candidate excluded until full-operation evidence shows a gain.

Read [requirements](requirements.md), [testing](testing.md), and [audit](audit.md).
