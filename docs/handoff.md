# Handoff — 2026-09-28

## Current state

The CUDA migration's original acceptance is historical. The whole-repository
audit remediation is in progress. `96e4ecc` has affected-operator evidence;
later fixed-status review found additional P0/P1 and evidence gaps, repaired
in reviewed source commit `487f8de`. Current 12-row gates are unmeasured.

- **Kernel backend:** CUDA is the default on B200. Setting
  `OH_MY_VLLM_KERNEL_BACKEND=tilelang` before Python starts selects the frozen
  TileLang comparison instead. Runtime identity reports the selection actually
  used, and nothing falls back silently to TileLang.
- **Ownership:** Rust owns serving, scheduling and the logical KV cache. Python
  owns GPU computation. No vLLM runtime, source or environment dependency is used.
- **Operator cases (clean `c36d1c9`, explicit CUDA):** all 147 formal cases pass.
  Each passes three rounds of 20 interleaved pairs with a positive one-sided 95%
  bootstrap bound.
- **Framework rows (same commit):** all 12 rows pass.
  - Throughput is 97.52–125.09% of the frozen vLLM baseline.
  - TTFT is 76.01–97.50% of the baseline.
  - Every row meets the 10% spread rule.
  - The narrowest margin is MTP batch 4, at 97.52% throughput.
- **Default-selection change (`030f60f`):** only changes backend selection and
  identity. The 174-test plus 31-subtest suite and the MTP4 service constraint
  and lifecycle checks passed on the working tree after `ef07b2b` that became
  this commit (CUDA source unchanged), not on a separate run of `030f60f`.
- **Other passing checks:**
  - Six 258048+4096 boundary runs finish without OOM or preemption.
  - Real-text preemption and prefix checks.
  - Both oh-my-pi APIs.

Evidence is in [acceptance.md](acceptance.md) and in the
`bench/baseline/2026-09-22-cuda-{operators,framework,features}.json` artifacts.

## Latest remediation checkpoint (2026-09-28)

- `96e4ecc` repairs the remaining P0/P1 request-isolation, RPC cancellation,
  registration, and FA kernel findings (SRV-01/05/06/07 and KRN-01/02/03).
  A cancelled prepare reply cannot shift the next RPC. Failed or rejected
  preparation releases Python state. Request-local failures do not stop other
  active streams; CUDA/device failures still stop the engine.
- SRV-03's remaining error classification is repaired in `d33b844`:
  explicit validation errors return HTTP 400, unknown Python prepare failures
  return 500, and the 65th active request returns 503. Axum's 413/415 status
  is preserved. The targeted bridge, HTTP, and serving tests passed 31 tests
  and 26 subtests, including malformed schemas and sampling parameters.
- The clean `96e4ecc` [affected operator evidence](../bench/baseline/2026-09-28-audit-p1-operators.json)
  passes all 29 selected formal cases (13 prepare-attention, 16 attention) on
  B200 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`. The smallest
  positive one-sided 95% gain bound is 0.000796 ms. This is an affected
  subset, not a new 147-case full-matrix result.
- The full B200 pytest suite after `96e4ecc` passed 211 tests and 32 subtests,
  with six skipped context-boundary placeholders. Rust workspace tests,
  formatting, line width, Ruff, Clippy, and commit hooks passed. The SRV-03
  focused Rust/Python tests and static checks also passed.
- Current 12-row framework throughput and TTFT remain **unverified** after
  hot-path changes. The 2026-09-22 values below apply to clean `c36d1c9`.
  Re-run all 12 rows after P2 performance decisions. EVD-07 remains open
  pending a clean, uninterrupted six-row run; its six tests no longer skip.
  Of the four risks originally marked "Not verified", three remain open;
  Rust MTP GDN slot placement was verified separately in `7e93c18` below.
- Later fixed-status review reopened SRV-02, KRN-05, PY-01, SCH-01/02/04,
  EVD-01/02/11/12/13 and MNT-04. Reviewed commit `487f8de` repairs SRV-02,
  KRN-05, PY-01, SCH-01/02/04 and EVD-01/02/11. EVD-01 rejects three-run
  or high-spread evidence; EVD-02 reports capacities from allocated FA/GDN
  tensors and fails CLI/log mismatches. EVD-11 scans the full cache trees.
  CPU regressions passed (73 tests and 37 subtests), as did Rust workspace
  tests (56/26/14), formatting, line width, Ruff, Clippy and commit hooks.
  Current GPU suite and full framework measurements remain pending.
- EVD-12 is repaired by `f34b661` plus 9648012: the exact loaded
  `.so` SHA-256 and TVM FFI nvcc path/version are recorded, and a source/flags/
  SM100a/ABI keyed sidecar permits hash-verified reuse if nvcc is unavailable
  in a later process. Both formal collectors fail closed on incomplete identity.
  Independent code review and focused CPU tests passed. A real B200 qk run
  passed 6/6 cases on UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`,
  with provenance and sidecar verified. That run is diagnostic because the
  worktree was dirty; it is not a clean full operator result. EVD-13's source
  finding is closed by `895d57b` + `487f8de`: the TVM FFI cache is now audited
  and a measured-window write regression passes. Current 12-row framework
  acceptance remains pending separately.
- MNT-04 runtime cleanup is committed as `ff1f944` (following the live Abort
  path added by `96e4ecc`): unused scheduler and zeroing-hint state is removed,
  protocol comments and EN/ZH design examples match the wire, grammar masks
  use direct draft IDs, FA/GDN capacities are checked separately, init has one
  deadline, and Ctrl-C/SIGTERM request Python shutdown within a shared
  deadline, forcing exit when it expires. Two real HTTP fixture regressions
  cover active SSE and a stalled reader. A
  focused CPU run passed 46 tests and 23 subtests; Rust workspace passed
  56/26/17, and formatting, line width, Ruff, Clippy and hooks passed.
  Independent review passed. Three unused frozen TileLang wrappers remain
  intentionally because their hashes define the historical comparison; the
  production and formal harness do not call them. No current GPU suite or
  12-row framework run is claimed for this change.
- `6da6427` closes EVD-03/08/10 in reviewed source: `scripts/test.sh` is the
  documented CPU/full pytest entry point with explicit GPU markers; the
  scripted worker tracks FA/GDN ownership, and an HTTP cancellation test
  proves real Rust shared-pool reuse by completing a replacement request
  that occupies all four allocatable blocks across position 784; formal
  fixtures use a local CUDA generator at the old seed 784. `scripts/test.sh cpu`
  passed 143 tests and 67 subtests (112 GPU cases deselected); Rust workspace
  passed 56/26/17, with fmt, line width, Ruff, Clippy and hooks clean. The
  focused B200 RNG test passed 1/1 on UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`; log:
  `/tmp/oh-my-vllm-evd10-fixture-smoke.log`. The owned GPU process exited.
  The full current GPU suite, formal operator matrix and 12-row framework
  collection remain pending.
- `4ed62ab` closes EVD-04/05 in independently reviewed tests. Owned CUDA
  paged GQA decode now has a CPU FP64 oracle across batch 1/2 and 784/785,
  with a reversed physical page table and a negative control proving that a
  missed second page exceeds tolerance. Grouped draft and proposal graph tests
  have independent CPU FP64 references alongside their consistency checks.
  The focused B200 suite passed 9/9 on UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`; the owned process exited.
  `scripts/test.sh cpu` passed 143 tests and 67 subtests (114 GPU cases
  deselected). Rust workspace passed 56/26/17, with fmt, line width, Ruff,
  Clippy and hooks clean. Current full GPU/formal/12-row acceptance remains
  pending.
- `ea0aa4e` closes EVD-09's source gap after independent review: every formal
  case now verifies both backends' returns and written cache/state slots before
  timing, with exact copied values, FP8 scale strides, and per-slot recurrent
  state limits. Mismatch fails collection; timed callables and the default
  fixture API are unchanged. Six adversarial CPU comparator cases and six
  representative B200 fixtures passed on UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`; no owned GPU process remains.
  `scripts/test.sh cpu` passed 149 tests and 67 subtests (120 GPU cases
  deselected); Rust workspace passed 56/26/17, and fmt, line width, Ruff,
  Clippy and hooks passed. A clean full operator matrix and 12-row framework
  collection still remain pending.
- `04540bd` implements the listed MNT-02 direct CUDA FFI contracts after
  independent review. Quantize, gates and recurrence check exact dtype, device,
  shape and stride before launch; fused SiLU accepts only BF16, empty gates
  return without a zero-grid launch, and empty quantize/recurrence reject.
  The fixed `1e-6` RMS epsilon matches the sole supported Qwen checkpoint,
  which the loader validates before loading weights. Recurrent index values
  remain a caller precondition: `starts` covers the token rows from zero,
  `reads` are in the pool, and `writes` are `-1` or in the pool. The CPU batch
  plan checks slot capacity and the runner constructs `starts`; no GPU→CPU
  index readback was added. B200 direct FFI tests passed 45/45, plus a focused
  `writes=-1` case 1/1, on UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`; owned GPU processes exited.
  `scripts/test.sh cpu` passed 153 tests and 67 subtests (166 GPU cases
  deselected); Rust workspace passed 56/26/17, and fmt, line width, Ruff,
  Clippy and hooks passed. The first clean affected-formal attempt at
  `6bd2119` stopped at a KRN-04 false rejection in large fused SiLU quantize;
  its two preceding PASS rows do not make that attempt acceptance evidence.
  Independently reviewed `952a01d` corrects the double-counted packed FP8
  width, accepts exactly `2**31` input elements, and guards that limit at
  direct CUDA FFI before launch. Independently reviewed `312c54b` applies
  the same exact boundary to the standalone `silu_mul` Python wrapper and
  direct CUDA FFI. CPU tests cover below/at/above the limit, the formal shape,
  and fused versus plain width. B200 direct FFI passed 55/55; current
  `scripts/test.sh cpu` passed 156 tests and 67 subtests (175 GPU cases
  deselected), Rust workspace 56/26/17, fmt, line width, Ruff, Clippy and
  hooks passed. On clean `312c54b`, the
  [affected formal subset](../bench/baseline/2026-09-28-audit-mnt02-krn04-operators.json)
  passed all 60 selected `quant`/`silu_quant`/`gates`/`recurrent` cases on
  UUID `GPU-a832d9c1-260f-9e37-0c99-95e62ab16ca5`. Each case passed output
  verification, 3 × 20 alternating pairs and the positive one-sided 95%
  bootstrap bound; the smallest gain bound was 0.00000699734 ms. The first
  clean `312c54b` run passed 59/60; both backends had elevated timing in the
  failed gates case, and the cause remains unconfirmed. That run is rejected
  diagnostic evidence. A separate
  [gates artifact](../bench/baseline/2026-09-28-audit-gates-confirm-operators.json)
  passed 13/13 on UUID `GPU-80cebbaf-a106-2605-b902-f5f9a08645ea`; the
  previously failed shape's lower bound was 0.0000465867 ms. No owned GPU
  process remained. This closes the listed KRN-04/MNT-02 source findings;
  the full current GPU suite, 147-case formal matrix and 12-row framework
  collection remain pending separately.
- Independently reviewed `0394727` closes MNT-03's alignment and dispatch
  contract. Recurrence checks the actual BF16/FP32 eight-element vector
  alignment (16/32 bytes), and alias-span analysis rejects negative strides.
  The documented RMS and gated-RMS dispatch formulas are compared to an FP64
  reference within existing BF16 tolerance; no bitwise batch invariance is
  promised. B200 focused tests passed 6/6, CPU passed 156 tests and 67 subtests
  (181 GPU cases deselected), Rust workspace 56/26/17, and fmt, line width,
  Ruff, Clippy, and hooks passed. Clean `0394727`
  [affected recurrent evidence](../bench/baseline/2026-09-28-audit-mnt03-recurrent-operators.json)
  passed 8/8 selected cases with output checks and positive bootstrap bounds
  on UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`; the smallest lower
  bound was 0.000064624 ms. Owned GPU processes exited. Full 147-case, GPU,
  and 12-row framework acceptance remains pending.
- Independently reviewed `be84176` closes MNT-01's discarded Python tuning
  knobs. CUDA private factories now accept only live settings while frozen
  TileLang signatures and public APIs stay intact. Decode still passes split,
  first-position, grouping, and position-width settings; its block calculation
  remains for a conservative position-width bound, and split buffer sizes are
  checked before launch. The focused CPU mock suite passed 9/9 and the full
  CPU entry point passed 165 tests and 67 subtests (181 GPU cases deselected).
  On B200 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`, public-output
  cases passed 25/25 and the post-guard decode rerun passed 2/2. Rust workspace
  56/26/17, fmt, line width, Ruff, Clippy, and hooks passed; owned GPU
  processes exited. Full current GPU, 147-case formal, and 12-row framework
  acceptance remain pending.
- Independently reviewed `f161b5e` replaces EVD-07's six skipped boundary
  placeholders with real ordinary/MTP4 GPU rows at batch 1/2/4, plus an
  explicit MTP draft gate for the long-context HTTP script. The CPU entry
  passed 201 tests and 67 subtests (181 GPU cases deselected); Rust 56/26/17,
  fmt, line width, Ruff, Clippy and hooks passed. Three clean-source GPU
  collection attempts were rejected by the GPU exclusivity gate: two on
  UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` during another user's
  short GEMM jobs, then one on UUID
  `GPU-a832d9c1-260f-9e37-0c99-95e62ab16ca5` when an external multi-GPU
  job started before MTP4 batch 4. The latter completed five diagnostic rows,
  which are not six-row acceptance. Raw logs remain under
  `/tmp/oh-my-vllm-context-boundary-f161b5e*`; owned GPU processes exited.
  A clean six-row run and the real long-context HTTP smoke remain pending.
- Independently reviewed `5a3b703` closes SCH-07's repeated whole-history
  SHA-256 work. The append-only hash chain matches full recomputation through
  262144 tokens and still yields a real prefix hit after generated output
  fills a block. A 100-iteration CPU microbenchmark measured old 42/334-block
  medians of 98.373/786.007 microseconds versus incremental medians of
  2.313/2.300 microseconds; this is local CPU work, not a framework result.
  Rust workspace 58/29/17, CPU pytest 201 tests and 67 subtests (181 GPU
  cases deselected), fmt, line width, Ruff, Clippy and hooks passed. Current
  12-row framework acceptance remains pending.
- Independently reviewed `c082937` closes SRV-09's repeated whole-call scan
  on the serving thread. The parser retains per-tool byte progress across
  chunks and checks non-string JSON at parameter close or before an
  incomplete length-finished call is discarded; the final `parse_tool`
  validation remains. The actual `Parser::feed`
  path has a bounded-byte-count regression across seven chunk sizes, with
  malformed and incomplete XML/JSON cases. A temporary CPU benchmark used
  16-byte chunks, five warmups, and ten measured repetitions: old `e7e6d42`
  median 151/1,668 microseconds versus new 22/89 microseconds for 8,192/32,768
  parameter bytes. This is parser-only cost. Rust workspace 58/29/21, CPU
  pytest 201 passed and 67 subtests (181 GPU cases deselected), fmt, line
  width, Ruff, Clippy, and hooks passed. Current 12-row framework acceptance
  remains pending.
- Independently reviewed `67deb85` closes SCH-06's priority inversion under
  pool pressure. The old eight-block regression evicted older request 1;
  the new scheduler evicts younger request 2 and retries request 1. A
  separate-pool MTP4 case evicts `[3, 2]` and preserves waiting order
  `[2, 3, 4]`, accepted histories, and cleared drafts. Tests cover the
  self-preemption fallback and same-step preempt/re-admit output; a CPU-fake
  Python worker test verifies state reset before planning, without claiming
  full production-block-size readmission execution. Requirements and
  architecture now state the implemented priority policy in both languages.
  Rust workspace 58/32/21, CPU pytest 202 passed and 67 subtests (181 GPU
  cases deselected), fmt, line width, Ruff, Clippy, and hooks passed. This
  policy repair has no local speed claim; current 12-row framework acceptance
  remains pending.
- Independently reviewed `b1761e9` closes SRV-10's step-counted SSE loss.
  A full 256-event channel pauses only its request and starts a 30-second grace;
  a 1,024-event/8-MiB FIFO preserves pending and final events, and generation
  resumes after the pending queue clears with at least 128 free channel slots.
  A real CPU HTTP test stopped reading, observed more than 256 worker steps
  and a stable pause before 750, completed another request while the stream
  was paused, and then received content sequence 1–750 and `[DONE]` in order.
  Rust tests cover grace timing, low-water behavior, capacity, disconnect,
  final drain, and detached drain task exit. At this checkpoint, an in-flight
  prepare or execute RPC could delay the 30-second check by its own timeout.
  After `efdef09`, background CPU preparation no longer blocks that check;
  an execute round trip or stalled prepare/Abort send still can. Rust workspace
  58/38/26, `scripts/test.sh cpu` 203 passed
  and 67 subtests (181 GPU cases deselected), fmt, line width, Ruff, Clippy,
  and hooks passed. Current full GPU and 12-row framework gates remain pending.
- Independently reviewed `f4aacca` closes KRN-07's silent misaligned FA-cache
  clone by adding a thread-safe per-process fallback count and power-of-two
  warnings with copied-byte size. It preserves unaligned BF16 view support;
  the clone remains, so no speed gain is claimed. A pre-change diagnostic
  clone of 4,383,375,360 bytes took median 2.736096 ms (five warmups, ten
  samples) on B200 UUID `GPU-a832d9c1-260f-9e37-0c99-95e62ab16ca5`, log
  `/tmp/oh-my-vllm-krn07-clone-baseline.log`. Four focused B200 cases passed
  on the same UUID, covering aligned/misaligned numerical output, cumulative
  calls, and warning throttling; no owned GPU process remains. CPU pytest
  passed 203 tests and 67 subtests (182 GPU cases deselected); Rust workspace
  58/38/26, fmt, line width, Ruff, Clippy, and hooks passed. Current full
  GPU/formal operator and 12-row framework gates remain pending.
- Independently reviewed `40e3e57` closes KRN-08's unobservable CUDA
  fast/generic dispatch choices. Thread-safe host submission counters cover
  `norm`, `add_norm`, `gated_norm`, `qk`, `recurrent`, `append`, and
  `convolution`; the formal collector requires one fast and zero generic
  submissions during output verification for its six listed operations.
  Focused B200 tests cover both variants of all seven. Clean
  [affected formal evidence](../bench/baseline/2026-09-28-audit-krn08-operators.json)
  passes 66/66 selected cases (3 × 20 interleaved pairs per case) on UUID
  `GPU-1b174534-ebba-826f-b452-8e7f3c05c301`; each row records fast=1,
  generic=0 and a positive one-sided 95% gain bound. The smallest bound is
  0.000000213335 ms; the maximum three-round spread/median is 1.695%.
  `scripts/test.sh cpu` passed 210 tests and 67 subtests (183 GPU cases
  deselected); Rust workspace, fmt, line width, Ruff, Clippy, and hooks
  passed. The owned GPU process exited. This is an affected subset, not a
  full 147-case result or a measured framework speed gain; the full GPU suite
  and current 12-row framework gates remain pending.
- Independently reviewed `94c3473` closes KRN-09's consuming CUDA launch
  checks and adds eager-only fault diagnostics. The 19 owned launch sites now
  use `cudaPeekAtLastError`; opt-in `OH_MY_VLLM_CUDA_DEBUG_SYNC=1` requires
  `OH_MY_VLLM_ENFORCE_EAGER=1` and checks the same stream before and after each
  launch, refusing graph capture first. Diagnostics label the observation
  phase without claiming an exact faulting instruction. Seven isolated B200
  cases passed on UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`, plus
  five repeated prior-fault subprocesses. CPU pytest passed 211 tests and 67
  subtests (190 GPU cases deselected); Rust workspace, fmt, line width, Ruff,
  Clippy, and hooks passed. The owned GPU process exited. Three same-GPU
  old/new eager RMS host-call diagnostics had near-equal medians but isolated
  outliers exceeded the 10% spread rule; their cause is unproven, so no speed
  claim is made. The logs and raw samples remain outside Git under
  `/tmp/oh-my-vllm-krn09-*`. Current full GPU/formal and 12-row framework
  gates remain pending.
- Independently reviewed `37f8cc9` implements KRN-10's per-specialization
  `cudaFuncSetAttribute` setup and per-thread, per-BK-tile caching of one or
  two FA page-table entries. The scope assumes one B200 CUDA context for the
  process. The clean `6ba9046`
  [before summary](../bench/baseline/2026-09-28-audit-krn10-attention-before.json)
  passed 16/16 selected attention cases on UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`; it does not validate the
  new source. Post-change focused B200 attention tests passed 29 existing
  and four new direct-FFI boundary/Position/grouped/graph cases. Same-GPU
  eager FFI A/B measured old/new medians 3.4138/3.0865 microseconds per call
  with 3.51%/5.33% spread, a host diagnostic only. Three full after16 formal
  attempts were rejected by external GPU processes (PIDs 849461, 897317,
  913070); none is accepted. `scripts/test.sh cpu` passed 211 tests and 67
  subtests (194 GPU deselected), plus Rust workspace, fmt, line width, Ruff,
  Clippy, and hooks. Owned GPU processes exited. KRN-10 remains open pending
  clean after16, current full147/GPU suite, and 12-row framework acceptance.
- Independently reviewed `17413e5` is the PY-05 candidate: MTP eager/draft/
  proposal metadata uses active extent width, and target decode keeps its
  existing extent width while uploading one page row per request. Device
  `index_select` expands rows for token-level attention. Target and draft
  graph replay still copy changed tables; regressions cover 783/784/262143
  boundaries and changed request grouping. At max context 262144, extent
  36864 and four 42-page requests with five token rows each, draft tables
  shrink from 20×335 to 20×48. Same-GPU host component A/B (5 warmups,
  20×256 calls) yielded median microseconds of 259.69→32.13 for draft graph,
  265.86→32.22 eager, 57.39→15.52 proposal, and 46.51→32.06 target. Some
  draft samples exceeded 10% spread; these are diagnostic component timings,
  not framework throughput evidence. Raw traces are under
  `/tmp/oh-my-vllm-py05-*`. `scripts/test.sh cpu` passed 220 tests and 67
  subtests (196 GPU deselected), focused B200 passed 5/5 on UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`, and Rust workspace
  58/38/26, fmt, line width, Ruff, Clippy and hooks passed. CPU and B200
  logs are `/tmp/oh-my-vllm-py05-cpu-17413e5.log` and
  `/tmp/oh-my-vllm-py05-gpu-focused-17413e5.log`. The owned GPU
  process exited. PY-05 remains open pending the full current GPU suite and
  12 framework rows, especially MTP batch 4.
- Independently reviewed `503ea27` is the PY-07 candidate: sampler prompt
  counts are computed once on the GPU at registration, generated counts are
  updated only on commit, and speculative drafts use a temporary count copy.
  With `top_k=50` plus `top_p=0.9`, a unique kth score allows a local stable
  sort; kth ties fall back to the full stable sort. That tie check adds one
  CUDA host synchronization. On the same idle B200 UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`, a diagnostic interleaved
  5-warmup, 20×8-call A/B at 262144 prompt tokens and vocabulary 248320
  measured median ms of 18.647→0.132 for penalty, 0.451→0.361 for top-k/p,
  and 18.870→0.418 for both; second-run spreads were below 10%. One-time
  registration grew from about 0.063→2.314 ms at 32768 prompt tokens and
  0.662→18.520 ms at 262144; the 32768 old and 262144 candidate samples
  had 10.65% and 15.56% spread, respectively. These are
  dirty-tree component observations, not framework gates. Logs:
  `/tmp/oh-my-vllm-py07-ab-gpu-dirty-r2.log`,
  `/tmp/oh-my-vllm-py07-register-gpu-dirty.log`,
  `/tmp/oh-my-vllm-py07-cpu-503ea27.log`, and
  `/tmp/oh-my-vllm-py07-gpu-focused-503ea27.log`. CPU 226 tests and 67
  subtests (199 GPU deselected), B200 sampler 3/3, Rust workspace 58/38/26,
  fmt, line width, Ruff, Clippy and hooks passed. Owned GPU processes exited.
  PY-07 stays open until the full current GPU suite and 12 framework rows
  establish that throughput and TTFT still meet their gates.
- Independently reviewed `efdef09` is the SRV-08 candidate. Rust overlaps one
  in-flight prepare with execute steps, buffers out-of-order Prepared replies,
  and treats unexpected/duplicate protocol replies as engine-fatal. Python
  builds template/tokenizer/grammar inputs on one daemon CPU thread; the DEALER
  owner receives a pipe wakeup and alone installs live request state. Real
  bridge and isolated HTTP tests cover no-followup wakeup, active SSE progress,
  both reply orders, timeout/late reply, cancellation, register overlap,
  malformed/duplicate reply and bounded shutdown. The Qwen tokenizer/XGrammar
  CPU concurrency and cross-thread matcher handoff regression passed. With one
  warmup plus five measured runs of each CPU fixture on the same current Rust
  binary, the median maximum active SSE event gap changed from 0.781507 s
  (synchronous `f194127` fixture) to 0.081265 s (deferred fixture); the
  respective spread/median values were 0.047% and 0.157%. Temporary harness
  and raw log: `/tmp/oh-my-vllm-srv08-ab.py` and
  `/tmp/oh-my-vllm-srv08-ab-efdef09.log`. `scripts/test.sh cpu` passed 236 tests and 70
  subtests (199 GPU deselected), with log
  `/tmp/oh-my-vllm-srv08-cpu-final.log`. Rust workspace 58/38/26, fmt, line
  width, Ruff, Clippy and hooks passed; task-owned CPU servers/workers exited.
  This CPU fixture measurement is not framework throughput evidence. Main
  thread sampler registration still includes PY-07 GPU prompt-count work;
  Formal B200 active-stream latency, the full current GPU suite and all 12
  framework rows remain pending, so SRV-08 stays open.
- A real-model B200 SRV-08 diagnostic with HEAD at `ed8c18c` ran on UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`: the first request emitted
  2,051 SSE data events, including 20 while a second 3,016-input/16-output
  request completed in 0.582 s. Logs are
  `/tmp/oh-my-vllm-srv08-b200-smoke.log` and
  `/tmp/oh-my-vllm-srv08-b200-service.log`; the service exited with no owned
  GPU process. This single run is diagnostic, not the 12-row gate.
- Independently reviewed `7e93c18` closes the narrow Rust MTP GDN slot
  placement risk. Real scheduler output covers 783/784/785/1568-token
  prompts and chunked prefill; a CPU HTTP/ZMQ fixture passes the actual Rust
  execute frame to Python `plan_request`/`commit` and verifies the 783→784
  source, five candidate writes and checkpoint copy. `scripts/test.sh cpu`
  passed 237 tests and 70 subtests (199 GPU deselected), logged at
  `/tmp/oh-my-vllm-mtp-slot-cpu.log`; Rust workspace 58/39/26, fmt, line
  width, Ruff, Clippy and hooks passed. This is not GPU numerical or current
  MTP framework acceptance.

Remaining work: PY-03/04/06, SRV-08, KRN-06, PY-05/07 and KRN-10 acceptance,
EVD-07, the three remaining unverified risks, and current
full operator/GPU suite and 12-row framework acceptance. See the [audit index](audit-2026-09-23.md)
for individual status and evidence limits.

## Historical 2026-09-24 remediation checkpoint

The audit of `030f60f` is recorded in
[audit-2026-09-23.md](audit-2026-09-23.md). It contains 51 findings across 6 batches.
Remediation started 2026-09-24 and is in progress.

### Completed batches

**Batch 1 — Serving availability** (`d0d286a`, 2026-09-24): SRV-01/02/03/04/06 fixed.
- SRV-01: temperature < 1e-5 clamped to 0 in both `request.rs` and `sampling.py`.
- SRV-02: `prctl(PR_SET_PDEATHSIG, SIGKILL)` in child pre-exec; Python watchdog polls
  parent PID every 5 s and exits if it is gone; `sock.recv` gets 5 s timeout.
- SRV-03: HTTP ready channel carries `(StatusCode, String)`; non-200 errors propagate.
- SRV-04: `Parser::feed` flushes pending bytes on `LengthFinish`; test added.
- SRV-06: `register_request`/`abort_request` demoted to fire-and-forget `()`.

**Batch 3 — Kernel/worker contracts, host-side** (`eb5ff62`, 2026-09-24):
  KRN-01/04/05, PY-01/02 fixed; MNT-02 partially addressed. KRN-02/03 were
  deferred at this checkpoint (device code required a GPU re-run).
- KRN-01: `decode_attention.py` asserts `group_size ≤ 5` when `starts` is supplied.
- KRN-04: `elementwise.py` and `fp8.py` guard against int32 overflow.
- KRN-05: `gdn.py` moves dtype/shape/stride validation into `normalize_qk`.
- PY-01: `batch_plan.py` debug-asserts FA tail pages are disjoint across requests.
- PY-02: `qwen.py` asserts `w.shape[0] % 128 == 0` for every FP8-scaled projection.
- MNT-02: `elementwise.py` early-returns on empty tensors in `silu_mul`/`delta_gates`;
  catch-all dtype dispatch and hard-coded epsilon remain open.
- Tests added: `tests/test_validation_guards.py`, `tests/test_batch_plan.py`,
  `tests/test_independent_decode_attention.py`.

**Batch 2 — Evidence integrity** (`895d57b`, 2026-09-24): EVD-06/11/12/13
fixed; EVD-01/02 partially addressed.
- EVD-01: `compare_vllm.py` enforces reps=5 and warmup=2; high spread only
  emits a warning, so the spread gate remains open.
- EVD-02: `ttft.py` parses some effective BENCH_CONFIG fields; FA/GDN pool
  capacities still come from CLI arithmetic, and mismatch regression is absent.
- EVD-02: `ttft.py` reads the actual BENCH_CONFIG log line from the worker via
  `_parse_bench_config()`; `zmq-worker/src/main.rs` emits `info!("BENCH_CONFIG", …)` at startup.
- EVD-06: `test_kernel_reference.py` asserts the exact expected file set in the
  frozen reference JSON.
- EVD-07: `tests/test_context_boundary.py` added six skip placeholders for
  REQ-CONTEXT-001 boundary runs; this did not close the finding.
- EVD-11: `measurement.py` appends to `problems` when a cache root is missing.
- EVD-12: `cuda_backend/__init__.py` gains `provenance()` returning nvcc version
  and the compiled `.so` SHA-256.
- EVD-13: `TVM_FFI_CACHE_DIR` added as a fourth cache root in the steady-state audit.

**Batch 4 — Scheduler hardening** (`42197f7`, 2026-09-24): SCH-01/02/03/04/05 fixed.
- SCH-01: `update()` validates token counts and the `is_final_chunk` flag before
  mutating state.
- SCH-02: `debug_assert!` → `assert!` for refcount/free-list invariants in `pool.rs`.
- SCH-03: `aligned_prefill` clamps to at least 1 when `count > 0`.
- SCH-04: `add_request()` returns `bool`; rejects over-capacity requests at admission
  (HTTP 413 from the serving layer).
- SCH-05: `remove_blocks_in_range` uses `continue` instead of `break` on NULL_BLOCK_ID.
- SRV-05/07 were still open at this checkpoint; `96e4ecc` later repaired them.

**Batch 6 partial — Cleanup** (`62101a3`, 2026-09-24): MNT-04 was marked
fixed at that checkpoint, but later review reopened it. That commit removed the
unused `abort_request()` method and updated part of the protocol comment;
`96e4ecc` and `ff1f944` completed the live Abort path and remaining cleanup.

### Findings still open at that checkpoint

The following findings were either deferred (device-code changes require a GPU
re-run) or are P2/P3 work accepted for later:

- **KRN-02/03** (P1): device-code fixes require re-running formal operator cases on B200.
- **SRV-05/07** (P1): RPC cancellation safety and `serving_outputs` side-channel.
- **P2 performance findings:** PY-03/04/05/06/07, SRV-08/09/10, SCH-06/07,
  KRN-06/07/08/09/10. Each requires measuring before keeping.
- **P3 cleanup/test findings:** EVD-03/04/05/08/09/10, MNT-01/03.

Two rules apply to every batch:

- A change to `kernels.cu` device code requires re-running its formal operator cases.
- A change to the hot path requires re-checking the framework rows before claiming
  the 12-row result still holds.

## Documentation cleanup (this change)

- Added the audit document.
- Removed superseded milestone narratives from `acceptance.md`,
  `cuda-development.md`, `plan.md` and `bench/baseline/README.md`. The raw
  artifacts remain, and a history table in `acceptance.md` indexes them.
- Deleted `performance-gap-2026-09-22.md`. It described an MTP gap that later
  work closed, and its summary data remains in
  `bench/baseline/2026-09-22-performance-gap.json`.
- Corrected stale content:
  - Commands now name `ttft.py` and the full pytest suite.
  - The protocol and preemption descriptions match the current code.
  - ADR statuses are current.
- Updated Chinese companions to match.

## Operating rules

- **Environment:** use `scripts/with-env.sh`.
- **GPU selection:** pick an idle GPU by UUID with `scripts/with-gpu.sh`.
- **Git:**
  - Develop and commit only on `main`.
  - Stage only the files you intended to change.
  - Keep hooks enabled.
  - Maintain the Chinese Markdown companions.
  - Include the required commit attribution.
- **GPU cleanup:** stop every GPU program you own promptly, and verify ports and
  GPU memory are released.
- **What stays out of the repository:** temporary tuning and raw GPU traces.
  Formal summarized evidence goes in `bench/baseline`.
