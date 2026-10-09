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

## Verification scope

The full CPU collection passed 548 tests and 70 subtests, with 323 GPU tests deselected.
The graph-budget checks passed 4 GPU tests. The full-output check passed again in 1 test.
The full GPU collection on `46f7529` passed 285 tests. It had nine boundary cases with total length 262144.
Each result keeps its source and test scope.

The paired comparison on current runtime `e83674c` passed at batches 1, 2, and 4.
Measured intervals had no capture or compilation. Input was 32768, output was 4096, and recompute preemptions were zero.
Operator, boundary, twelve-row framework, and full service acceptance for the current runtime wait for GPU availability.
GPU contention from another task stopped subsequent collections. The completed paired job and failed parent collections keep different statuses.

See the [acceptance index](acceptance.md) for full results, source identities, original hashes, and failed attempts.
Formal DSpark/MTP4 comparison reports TPS, TTFT, spread, acceptance, and paired confidence bounds with no DSpark TPS, TTFT, spread, or confidence-bound acceptance gates.
The initial twelve workload gates and the operator speed gates stay active.

The previous DSpark service collection on `de1591b` passed constraints, lifecycles, OMP tool loops, and long-context prefix reuse with the two APIs.
Its reports keep factual answer errors and source-read scope. The service collection for the current runtime will verify these features again.

Source stayed unchanged in the stopped collections. All task-owned processes and GPU resources were released.
No task-owned GPU worker or temporary service port stays active.
The user-requested tutorial preview stays on port 18084 for chapter review.
Ignored `LOCAL.md` holds its address, PID, and maintenance commands.

## Documents and evidence

English Markdown uses ASD-STE100 Issue 9 rules and the project technical glossary, with full Chinese translations.
The acceptance document is the only evidence index. ADRs keep important decisions and records of implementation replacements.
Repair records stay in Git history and the external attempt ledger.

Tracked evidence contains portable summaries, measurement values, software/hardware versions, and original hashes.
Original evidence and host configuration stay in ignored local or external storage.
The 66 initial tracked records have byte-identical local copies with SHA-256 comparisons.
Formal comparators reject derived envelopes. New servers must measure their own matching baseline.

The Twine tutorial has thirteen full Chinese chapters. One chapter gives the DSpark method.
Its excerpts and full-source pages bind current source hashes.
Node and browser checks include navigation, progress, formulas, fonts, fields, code, and the two themes.

## Open work

- Complete user review of the tutorial and correct the teaching gaps that the user identifies.
- Test cross-shape 262k graph eviction/recapture with measured memory pressure, as `PY-04` specifies in [audit](audit.md).
- Show bounded compiler storage when shapes change again and again before new long-lived-worker claims.
- Use measurements for `PY-06` host-overlap work. The pinned readback experiment showed no important full-call gain.
- Do not use the rejected `KRN-06` candidate until full-operation evidence shows a gain.
- Calculate new performance thresholds from roofline analysis in a new requirement change.

The 95% throughput, 110% TTFT, and 10% spread gates stay active.
Read [requirements](requirements.md), [testing](testing.md), and [audit](audit.md) before the next implementation task.
