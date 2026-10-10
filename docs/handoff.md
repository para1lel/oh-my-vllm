# Current state and open work

Updated: 2026-10-10.

## Current implementation

Rust controls service, scheduling, and logical KV. Python controls GPU computation on one B200.
The default backend is CUDA. Pinned TileLang supplies the operator reference.
Keep block size 784, 16 FA layers, 48 GDN layers, and `mamba_cache_mode="align"`.
The total input and output limit is 262144 tokens.

Ordinary, MTP4, and DSpark are startup choices.
Chat, Responses, streams, tools, thinking settings, constraints, cancellation, and prefix reuse stay available.
Ordinary decoding keeps FP32 GDN state. Speculative modes keep BF16 state and rollback snapshots.

DSpark keeps fixed weights and shares the target embedding and vocabulary head.
Its five-layer draft reads zero-based target features `(5,19,33,47,61)`.
Only kept target rows enter persistent context.

Markov proposals and cumulative confidence select zero through seven candidates. The initial threshold is `0.2`.
Set the threshold to `0.0` for fixed-count diagnostics. Output, context, and grammar limits still apply.

Stochastic acceptance uses `min(1,p/q)`, normalized `max(p-q,0)` after rejection, and a target bonus after full acceptance.

Use [development](development.md) for environment selection and commands.
Set `OH_MY_VLLM_MODEL` or `--model`, with `OH_MY_VLLM_DRAFT_MODEL` or `--draft-model` for DSpark.
[Architecture](architecture.md) defines ownership, states, and cache contracts.

## Current performance work

The fifteen-row collector records each request's submission, first token, and last token.
The collector records prefill and decode wall time, theoretical bounds, and spread checks.
Prefix-hit prefill has no latency ratio gate. Its 10% spread check stays active.
All other phases keep the three-times latency gate, with prefix-hit decode.

The reviewed semantic model counts required parameter reads, shared resources, dependencies, effective shapes, state writes, and draft work.
The source contract now includes the owned accumulator header. This identity update keeps cost equations and tolerances the same.

Execution uses CUDA Graph, multiple CUDA streams, and PDL.
GDN projection/convolution joins an independent gate branch.
Prefill graphs restore persistent writes and dynamic metadata.
Capture topology reuse decreased diagnostic storage from 29.13 to 7.82 GB. Start new prefill capture only with at least 32 GiB free memory.

Large gate/up uses K256 tiles, three stages, column-major scales, and swizzle 16.
The K-major reference keeps K128 tiles and five stages.
Five projections at 624 through 2496 rows use two-block clusters, M128/N128/K128 tiles, and five stages.
Their owned accumulator reads two independent TMEM tiles before one wait.
Their epilogue directly converts FP32 accumulators to BF16 with the same rounding.
Other shapes keep their previous dispatch.

Residual RMS and GDN gated RMS fuse quantization at the specified BF16 rounding points.
Selected-output paths keep required KV and boundary features.
The scheduler delays a rounded one-token prefill only after a successful chunk of at least one block in that step.
Decode and speculative verification keep progress. The production token budget is 32768.

The observability tool requests 21 metrics per observed kernel and separates native counters from representative TileFoundry HIR estimates.
It keeps each profiler worker's output, loaded CUDA identities, and process identity.
Counter records must match the selected process, case, and backend.
[Profiling](profiling.md) gives measurements and selected configurations.

## Verified source scope

The [source-bound record](../bench/evidence/2026-10-10-current-source-acceptance.json) applies to `c27ff1a`.
Thirteen of fifteen full-output rows passed. Prefix-hit batches 1 and 2 failed prefill.
Their prefill ratios were about 3.790 and 3.162. All decode phases and all wall-time spreads passed.

The full 317-case operator matrix passed numerical and speed checks.
All nine total-length 262144 cases passed with zero recompute preemptions and no OOM.
DSpark and MTP4 service suites passed the two APIs, constraints, cancellation, long prefixes, and observed oh-my-pi tool loops.

The full Python collection passed 1181 tests and 65 subtests. Two test assumptions failed.
Subsequent corrections select an invalid variant after the last registered value and remove an inherited GPU UUID from the selector mock.
Focused tests passed after these corrections.
The current CPU collection passed 753 tests and 89 subtests, with 445 GPU tests deselected.
Full GPU recollection stays pending.

The subsequent owned-accumulator production path passed 26 checks.
They cover dispatch boundaries, five shapes, changed graph data and weights, PDL choices, side-stream replay, and output guards.
All 18 affected full-operation cases passed numerical and speed checks.
These filtered dirty-source measurements have diagnostic scope. Current full-source recollection stays pending.
The [accumulator diagnosis](../bench/evidence/2026-10-10-fp8-accumulator-observations.json) keeps the selected cases, 105 profiler counter records, rejected candidates, and source identities.

[Acceptance](acceptance.md) gives original hashes, source limits, and review scope.
Keep failed and interrupted attempts. Past source results do not supply acceptance for subsequent changes.

## Documents and preview

English Markdown follows ASD-STE100 Issue 9 with full Chinese translations.
Documents and the thirteen-chapter Twine tutorial show current project code and contracts.
Portable evidence keeps measurements, source identities, original hashes, and limits.
Host configuration and original records stay in ignored or external storage.

The user-requested tutorial preview stays on port 18084.
Ignored `LOCAL.md` keeps its address, process identity, and maintenance commands.

## Open work

- Complete the all-owned-operator audit with Opus 5.5 and correct material findings.
- Record prefix-hit prefill latency without a ratio gate. Keep its spread and decode checks active.
- Complete current-source phase, operator, capacity, service, and full Python verification after inference changes.
- Finish evidence/document review by a different agent and the required build checks before the next commit.
- Run tutorial checks after source changes and keep its preview available.
- Complete user review of the tutorial.
- Test cross-shape graph eviction/recapture with measured 262k memory pressure, as `PY-04` specifies in [audit](audit.md).
- Show bounded compiler storage when shapes change indefinitely.
- Use measurements for `PY-06` host-overlap work.
- Keep the rejected `KRN-06` candidate excluded until full-operation evidence shows time saved.

Read [requirements](requirements.md), [testing](testing.md), and [audit](audit.md).
