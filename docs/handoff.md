# Handoff — 2026-09-29

## Semantic IR accepted checkpoint (2026-09-29)

REQ-IR-001 now routes the Qwen/MTP model's owned CUDA and key FlashInfer calls
through Python `torch.library` semantic operators. Production uses fullgraph
`torch.compile` for prefill, target decode, MTP draft and proposal, with the
existing manual CUDA Graphs around the latter three. Static metadata selects
providers; cache mutations have explicit schemas and the PyTorch reference
requires an explicit debug selection. The checked 18-site inventory and a
conservative BF16 SiLU/FP8 graph rewrite are included. Eager/compiled 27B
tests compare outputs and written caches across prefill-to-decode and changed
MTP replay metadata. There is no automatic eager/reference fallback.

Clean commit `619c9d9` passed `scripts/test.sh full`: **544 passed, 70
subtests**, including all six 258048+4096 boundary runs, on B200 UUID
`GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` (log
`/tmp/oh-my-vllm-ir-full-gpu-green.log`). `scripts/test.sh cpu` passed
**303 tests, 70 subtests** (`/tmp/oh-my-vllm-ir-cpu-verified.log`). Ruff,
Rust workspace tests, fmt, line width and Clippy passed. A read-only
sub-agent review found no remaining P1/P2 correctness issue. The first full
GPU attempt was invalidated by an unrelated process on the selected GPU;
the second passed 543/544 and exposed an old mock returning a non-Tensor;
the fixed test and the final full suite pass. Owned GPU processes exited.

[Formal operator evidence](../bench/baseline/2026-09-29-ir-operators.json)
passes all 147 CUDA/frozen-TileLang cases on clean `619c9d9`; the smallest
positive one-sided 95% gain bound is `0.0000036639670530955112 ms`.
[Framework evidence](../bench/baseline/2026-09-29-ir-framework.json) accepts
all twelve rows from the same source: throughput is 99.27–125.63% and TTFT
76.92–95.91% of the frozen baseline. Every accepted row has two warmups, five
measurements, at most 10% spread, no preemption and a passing steady-state
log/cache audit. Nine rows used one B200 UUID and three prefix rows used
another; each row was single-GPU. Three earlier complete prefix batch-1
attempts were rejected for 17.378%, 20.976% and 20.177% TTFT spread. Six
other attempts were interrupted by unrelated GPU processes. Raw identities,
hashes and rejection records are preserved in the framework summary. The
isolated TTFT spike cause remains unproven; it predates IR. Current audit
found no logged or disk-cache evidence of measured-window compilation/capture,
but cannot rule out silent in-memory recompilation. A compilation-count
boundary snapshot is a future audit improvement.

REQ-IR-001 is accepted for the tested workset. The 4096 Dynamo recompile
limits and per-unit compilation warnings do not prove bounded long-lived
shape churn; eviction/recapture pressure beyond the tested workset remains an
open capacity limit. No activation donation is enabled without a proven
temporary destination. No vLLM runtime dependency was added.

## Previous accepted state

The CUDA migration's original `c36d1c9` acceptance is historical. The
whole-repository audit remediation source at clean `e3c42e0` now passes the
full B200 suite, 147-case formal operator matrix, and 12 accepted framework
rows with one documented high-spread row repeat. PY-04, PY-06 and KRN-06
retain the scoped open limits below. The dated checkpoints after the latest
one preserve what had been pending at those earlier commits.

- **Kernel backend:** CUDA is the default on B200. Setting
  `OH_MY_VLLM_KERNEL_BACKEND=tilelang` before Python starts selects the frozen
  TileLang comparison instead. Runtime identity reports the selection actually
  used, and nothing falls back silently to TileLang.
- **Ownership:** Rust owns serving, scheduling and the logical KV cache. Python
  owns GPU computation. No vLLM runtime, source or environment dependency is used.
- **Current operator cases (clean `e3c42e0`, explicit CUDA):** all 147 formal
  cases pass three rounds of 20 interleaved pairs, output checks and the
  frozen TileLang comparison; the smallest positive one-sided 95% lower bound
  is `0.000005722664296627035 ms`.
- **Current framework rows (same source):** 11 passed in the first complete
  collection; prefix batch 1 was rejected for 23.924% TTFT spread and passed
  a separate same-GPU repeat at 7.825%. The resulting 12 accepted rows have
  throughput 99.43–124.58% and TTFT 75.70–96.40% of the frozen baseline.
  Every accepted row passed the 10% spread and steady-state audit gates.
- **Default-selection change (`030f60f`):** only changes backend selection and
  identity. The 174-test plus 31-subtest suite and the MTP4 service constraint
  and lifecycle checks passed on the working tree after `ef07b2b` that became
  this commit (CUDA source unchanged), not on a separate run of `030f60f`.
- **Other passing checks:**
  - Six 258048+4096 boundary runs finish without OOM or preemption.
  - Real-text preemption and prefix checks.
  - Both oh-my-pi APIs.

Current evidence is in [acceptance.md](acceptance.md) and the
`bench/baseline/2026-09-29-audit-final-{operators,framework}.json` summaries.
The `2026-09-22-cuda-{operators,framework,features}.json` artifacts remain history.

## Latest acceptance checkpoint (2026-09-29)

- Clean `e3c42e0` B200 full pytest: **491 passed, 70 subtests**, 42 third-party
  warning/compile-hint messages; log
  `/tmp/oh-my-vllm-final-full-gpu-e3c42e0.log`.
- [Full formal summary](../bench/baseline/2026-09-29-audit-final-operators.json):
  147/147 passed, UUID `GPU-a832d9c1-260f-9e37-0c99-95e62ab16ca5`,
  no interference, clean source, loaded CUDA module SHA-256
  `24bec3a7208ac09629d0a594940266dec2dfbb272c87e02e27c6a8196cd84318`.
  Raw `/tmp/oh-my-vllm-final-formal-e3c42e0.json` SHA-256
  `526c99df52ded4cf73630d08ed5f0861a383da5644bc937f55bb3a4725989c33`.
- [Framework summary](../bench/baseline/2026-09-29-audit-final-framework.json):
  same source and UUID, release binary SHA-256
  `90038fc7f9598e1e42f7beb8f04370463f3d1a33fb83b33a91f6698e3f0af888`.
  First owner `/tmp/oh-my-vllm-final-framework-e3c42e0.log` produced 12
  structurally valid row artifacts but accepted only 11: prefix batch 1 had
  23.924% TTFT spread, so the owner exited 1;
  separate owner `/tmp/oh-my-vllm-final-framework-prefix1-retry-e3c42e0.log`
  exited 0 with 7.825% spread. The accepted 11+1 rows each had two warmups,
  five measurements, observed pool capacities, clean steady-state audit,
  throughput at least 99.426%, and TTFT at most 96.402% of baseline.
- Post-run `scripts/test.sh cpu` passed 266 tests, 225 GPU deselections and
  70 subtests (`/tmp/oh-my-vllm-final-cpu-e3c42e0.log`). Rust workspace,
  fmt, line-width, Ruff format/check and Clippy all passed. No owned
  server/worker/TTFT process or service listener remained; selected GPU was
  0 MiB/0% with no compute app. Thirteen run-owned unbound IPC socket files
  were removed. The worktree is changing only for these reviewed docs/evidence.
- PY-03's bounded graph policy is closed for tested Serve workset shifts and
  framework rows. Six 258048+4096 boundary cases completed on their earlier
  clean source; they do not test repeated shape churn at 262144. A separate
  physical <4-GiB first-miss probe passes. A later resident 36k→40k
  transition also passed below 4 GiB, but PY-04 is **partial/open** because
  cross-shape 262k eviction and recapture remain untested;
  PY-06 is **partial/open** because whole-step host/GPU overlap benefit is
  unverified after a pinned-DtoH diagnostic with no material full-call gain;
  KRN-06 remains **open/no-go** after its safe candidate had no
  stable full-call gain. See the current [audit index](audit-2026-09-23.md).

## Physical low-headroom checkpoint (2026-09-29)

- An external-only, independently reviewed shim ran on clean HEAD `8111c19`
  and B200 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`, using the
  unchanged release binary SHA-256
  `90038fc7f9598e1e42f7beb8f04370463f3d1a33fb83b33a91f6698e3f0af888`.
  The CLI FA capacity 7000 mapped to 2333 actual FA blocks; GDN capacity was
  128. After cache initialization, 68 retained allocations of at most 256 MiB
  reduced real free memory from 22,479,568,896 to 4,225,957,888 bytes
  (3.936 GiB) for an MTP4 batch-1, 32-input, 32-output bench. Target, draft
  and proposal `headroom_eager` counts were 26/110/28, with zero captures.
  The run completed 32 output tokens, proposed/accepted 56/18 MTP drafts,
  and reported zero preemptions.
- Raw `/tmp/oh-my-vllm-py04-physical/probe-141183.log` has SHA-256
  `09c2513114c4f2a5f7700023d2d49e07ca236bfbc2a4856a880110b2d810d17a`;
  result `/tmp/oh-my-vllm-py04-physical/probe-result-141183.json` has SHA-256
  `ce0fb9ed75ddad08b5b4647f3a41796eb41c75bbba73263e4ae68a835ecb2a49`.
  Owner PID 141183, child PGID 141218 and worker PID 141220 exited; selected
  GPU returned to 0 MiB/0%, and the owned IPC socket/listener was absent.
  The first attempt failed closed before reservation because its 20 GiB
  precondition was lower than actual 20.94 GiB free. This diagnostic proves
  fresh graph misses become eager under the physical guard for this one
  configuration. Existing-graph late shape changes and repeated 262144-token
  churn were unverified at this checkpoint. The later resident-graph
  diagnostic below narrows the first limit; PY-04 remains **partial/open**.

## Resident graph and repeated 262k checkpoint (2026-09-29)

- Independently reviewed external-only probes on clean HEAD `70a5e5e`
  used B200 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` and the
  unchanged release binary SHA-256
  `90038fc7f9598e1e42f7beb8f04370463f3d1a33fb83b33a91f6698e3f0af888`.
  Production source was not changed. Both used zero warmups and are behavior
  diagnostics, not formal performance measurements.
- With FA/GDN 2333/128, an MTP4 batch-1 36832+256 run first established
  exact resident target/draft/proposal extent-36864 graphs and no extent-40960
  graphs. Sixty-seven retained allocations of at most 256 MiB lowered actual
  free memory from 22,055,944,192 to 4,070,768,640 bytes. The first 40960
  misses came draft→proposal→target: each had free bytes below 4 GiB,
  incremented `headroom_eager` once, retained the old graph and did not add
  a capture. Final capture counts remained 2/2/1, no eviction. Output was
  256 tokens, MTP proposed/accepted 225/199, preemptions 0. Raw
  `/tmp/oh-my-vllm-py04-resident/probe-283459.log` SHA-256 is
  `9f5113d42622635639dbb1e278f0b68fc8c1681e4233939ff115de9ea3a8f847`;
  result `/tmp/oh-my-vllm-py04-resident/probe-result-283459.json` SHA-256 is
  `f5fae3f04eb916ce6f358c0a15d74af50f573528edf3559b8925760b3ebbbc7d`.
- With FA/GDN 1400/128, a separate MTP4 batch-4 worker completed two
  258048+4096 rounds in one process. Both produced 16,384 tokens with zero
  prefix hits or preemptions and MTP proposed/accepted 16,219/12,318. The
  first round captured 45 graphs, the second 0; final residents were
  target/draft/proposal 13/28/4, while eviction, recent recapture and churn
  cooldown stayed 0. Thus same-shape 262144-token repetition completed,
  while `repeated_churn_verified=false`. Raw
  `/tmp/oh-my-vllm-py04-262k/probe-308672.log` SHA-256 is
  `10ab638c78e4ac8ca91c45af1d279bba211a20594b717b799efb844fc5514f61`;
  result `/tmp/oh-my-vllm-py04-262k/probe-result-308672.json` SHA-256 is
  `1df3e7f27be02e8b5326ce9ca0b32dc40ed171aa193e8f28bbc4d4db08507bea`.
- Both owners exited with no surviving worker/group or IPC listener; each
  selected GPU returned to 0 MiB with no compute PID, and source identity
  stayed clean. The Rust Bench command holds one shape across repetitions.
  A real 262k cross-shape eviction/recapture test needs a separately reviewed
  bounded mixed-shape Serve session with exact token lengths and per-wave
  graph-key/counter records. That path remains unverified; PY-04 remains
  **partial/open**. PY-06 and KRN-06 retain their preceding statuses.

## Whole-step overlap checkpoint (2026-09-29)

- Clean HEAD `fa3b8e6` used the unchanged release binary SHA-256
  `90038fc7f9598e1e42f7beb8f04370463f3d1a33fb83b33a91f6698e3f0af888`
  and B200 UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`. An external
  profiler captured one warmed MTP4 batch-4 decode step at 32768 input/256
  output: CPU step 21.891 ms, 1251 kernels totaling 13.624 ms, merged GPU
  work 13.535 ms. The target greedy readback's host `cudaMemcpyAsync` lasted
  15.044 ms while its 320-byte GPU DtoH lasted about 3.4 microseconds. The
  target token is needed for verification, commit and MTP; proposal tokens
  are needed by Rust's next scheduler update. The 4.417 ms gap after target
  DtoH cannot be attributed from one profiler trace. Raw trace
  `/tmp/oh-my-vllm-py06-profile/trace-173909.json` SHA-256 is
  `00767f55e0b04e12886385c703ee0d5d0b7e714addd4beb7b7e0eb5dc694357c`;
  owner result `/tmp/oh-my-vllm-py06-profile/result-173804.json` SHA-256 is
  `6a0bdc5e05a21d42d6c90383b6019f3ddb910a8c5e157e23bec3787980e5473d`.
  Its 157 ms instrumented wrapper wall time includes profiler setup/exit and
  is not a throughput measurement.
- An independently reviewed external pinned-D2H shim passed 12 focused
  checks: nine CUDA exact-output cases for FP16/BF16/FP32, ties and nonfinite
  values, one CPU fallback and two CUDA invalid-input rejections.
  `/tmp/oh-my-vllm-py06-ab/equivalence.log` SHA-256 is
  `a7283cce1be0067b1ad6527d9855409143a05f34c0fe0217c29b8e65d3a8745a`.
  A profiler-free ABBA comparison used four serial workers on that UUID,
  one warmup and three measured runs per worker, with the same MTP4 batch-4
  32768→256 workload. Baseline/pinned medians across six samples each were
  137.031/137.328 token/s (+0.216%); maximum median deviations were
  0.885/0.314%. The two baseline groups drifted about 0.287%, exceeding
  the candidate gain. Every measured row had 1024 outputs, 69 steps,
  961/783 proposed/accepted drafts and no preemption. Result
  `/tmp/oh-my-vllm-py06-ab/result-211290.json` SHA-256 is
  `bedccc3e3c2ef52c545755249250810d50dac7a4994a832905e718704e949f03`;
  it records all four raw log hashes and cleanup. This 1+3 diagnostic
  is not the formal 2+5 framework protocol and has no token-sequence hash.
  Pinned DtoH was not added to production. The prior full GPU, 147 operator
  and 12 framework gates remain valid because no repository source changed.
  PY-06 remains **partial/open**. Owner PID 173804 and its worker/IPC from
  tracing, ABBA owner PID 211290 and all four worker groups/IPC, and the
  focused test process exited; the selected GPU returned to 0 MiB/0%.

## Earlier remediation checkpoint (2026-09-28)

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
  Re-run all 12 rows after P2 performance decisions. EVD-07's clean six-row
  boundary and real MTP HTTP smoke passed on `bd8e21e` and `8dfc97b`.
  The four risks originally marked "Not verified" now have scoped evidence:
  Rust MTP GDN placement in `7e93c18`, tested FlashInfer workspace/stale-tail
  paths in `6b378d5`, and the BF16 reciprocal bound in `32cdc2b` below.
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
  The later clean `bd8e21e`
  [six-row artifact](../bench/baseline/2026-09-28-audit-evd07-context-boundary.json)
  passed all ordinary/MTP4 batch 1/2/4 rows at 258048+4096 on pinned UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`. Every row generated
  `batch_size * 4096` output tokens, with zero preemptions and no OOM.
  MTP4 proposed/accepted drafts were 5010/2842, 9075/5917, 16219/12318;
  largest worker peak reserved memory was 132441440256 bytes. Start/end source,
  release binary hash and row UUIDs matched, with clean Git status. Raw logs
  are under `/tmp/oh-my-vllm-evd07-context-bd8e21e*`. All owned GPU workers
  exited and `nvidia-smi` reported no compute processes afterward. The real
  131072-token MTP HTTP smoke then passed on clean `8dfc97b`:
  [summarized evidence](../bench/baseline/2026-09-28-audit-evd07-long-context-http.json)
  records chat/completions and responses, first/repeat each, four validated
  strict-JSON answers, 131099 prompt tokens per request, 130928 cached tokens
  on repeats and 20 MTP draft proposals per request. Start/end clean source,
  binary SHA-256 and UUID match; dedicated server exit=0, listener/IPC closed,
  selected GPU compute apps empty. Logs/provenance:
  `/tmp/oh-my-vllm-evd07-http*`. EVD-07 is fixed for the requested evidence;
  full current GPU, 147-case formal and 12-row framework gates remain pending.
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
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`. Clean `231f066`
  [after summary](../bench/baseline/2026-09-28-audit-krn10-attention-after.json)
  passed the same selected 16/16 cases on the same UUID: clean source,
  no interference, all output and frozen TileLang comparisons passed,
  minimum one-sided 95% lower gain +0.0028209709 ms, and maximum three-round
  spread/median 0.173%. `selected_passed=true`; top-level `passed=false`
  denotes the unselected 131 other formal cases. All 16 CUDA three-round
  medians improved across the two source commits, median 7.26%, but this
  does not isolate changes or establish framework speed. Post-change focused
  B200 attention tests also passed 29 existing and four new direct-FFI
  boundary/position/grouped/graph cases. Same-GPU
  eager FFI A/B measured old/new medians 3.4138/3.0865 microseconds per call
  with 3.51%/5.33% spread, a host diagnostic only. Three earlier full after16
  formal attempts were rejected by external GPU processes (PIDs 849461, 897317,
  913070); none is accepted. `scripts/test.sh cpu` passed 211 tests and 67
  subtests (194 GPU deselected), plus Rust workspace, fmt, line width, Ruff,
  Clippy, and hooks. Owned GPU processes exited after clean after16. The
  affected subset is verified; final acceptance remains pending on current
  full147/GPU suite and 12-row framework gates.
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
- Independently reviewed `6b378d5` verifies the two FlashInfer risks for
  version 0.6.18.post1 on B200's actual TRT-LLM gen decode and context
  prefill paths. Each zeroed or `0xA5` workspace was set on a separate fresh
  plan before its first backend call, crossed with zero/128 stale KV tails;
  a backend spy saw eight actual calls per case. Decode covers nine short
  and page-boundary lengths, batch 4 at 32769, grouped five drafts at 785,
  and single 131073; prefill covers 4097/4237/4703. All 15 focused GPU
  cases passed with zero-tolerance numerical equality across contamination
  variants; the small decode cases
  also matched CPU FP64. UUID:
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`; raw log:
  `/tmp/oh-my-vllm-flashinfer-poison-tests-r2.log`. `scripts/test.sh cpu`
  passed 237 tests and 70 subtests (214 GPU deselected), logged at
  `/tmp/oh-my-vllm-flashinfer-cpu.log`; Rust workspace 58/39/26, fmt,
  line width, Ruff, Clippy and hooks passed. The owned GPU and CPU workers
  exited. FlashInfer's documented zeroing requirement applies to XQA, which
  production does not call. The result does not cover XQA, FA2, or all
  262144-token layouts; full current GPU and framework gates remain pending.
- Independently reviewed `32cdc2b` verifies the finite BF16 reciprocal
  assumption from NVIDIA PTX ISA 9.4 and the source operand range. The ISA
  guarantees at most 1 ulp for `rcp.approx.f32`; finite BF16 scale/inverse
  remain FP32 normal, so `.ftz` does not change them. Nonfinite maxima and
  FP16/FP32 use exact division. A B200 regression at commit `32cdc2b`
  checks production CUDA FP8 bytes and scales against a CPU IEEE FP32 RN
  reference for 65,280 finite BF16 encodings, 32,640 finite maxima with
  signed midpoint-derived samples, all 126 exact positive/negative FP8
  midpoints at maximum 448, and nonfinite fallback. Focused 3/3 passed on
  UUID `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`; log:
  `/tmp/oh-my-vllm-rcp-focused-32cdc2b.log`. The CPU suite at test commit
  `c0ff543`, before the comment-only source correction, passed 237 plus 70
  subtests (217 GPU deselected), log:
  `/tmp/oh-my-vllm-rcp-cpu-final.log`; Rust workspace 58/39/26, fmt,
  line width, Ruff, Clippy and hooks passed. All owned processes exited.
  This bounded hardware test does not exhaust all maximum/input pairs or
  prove bitwise equality of intermediate FP32 quotients. The full current
  GPU suite, 147-case formal matrix, six context rows and 12 framework rows
  remain pending.
- Independently reviewed `b464634` is a PY-04 shared-pool candidate. Target,
  draft and proposal graphs each share a separate family pool; draft hidden is
  copied to the next graph's persistent input on the same stream before replay.
  At source commit `b464634`, B200 graph tests passed 8/8 on UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`, log
  `/tmp/oh-my-vllm-py04-gpu-b464634.log`. A paired dirty-candidate diagnostic
  on the same UUID used HEAD `a02812d` plus identical five-file patch SHA-256
  `7c531868cda9e65590ecdf08e1ef467f51c4195ea8b6c371e92fa36db752b8fe`
  and release binary SHA-256
  `90038fc7f9598e1e42f7beb8f04370463f3d1a33fb83b33a91f6698e3f0af888`;
  both logs include per-file hashes: `/tmp/oh-my-vllm-py04-{private,shared}-r2.log`.
  With FA 1400/GDN 128 blocks, MTP4 batch 4, 32768 input/512 output, two
  warmups and five measured runs, private→shared peak reserved memory fell
  126565220352→125986406400 bytes (578813952 bytes, about 552 MiB). Median
  throughput was 243.936→244.011 token/s and maximum TTFT 6.55794→6.55717 s;
  both spreads were below 0.55%, with zero preemptions and 1562 accepted drafts
  per run. A separate **clean `b464634`** run with FA 2333/GDN 128 crossed
  extent 36864→40960, captured new target/draft/proposal shapes, and completed
  without OOM or preemption at 176815079424 peak reserved bytes; it had one
  warmup and one measured run, log `/tmp/oh-my-vllm-py04-late-b464634.log`.
  Candidate-tree `scripts/test.sh cpu` passed 237 tests/70 subtests (220 GPU
  deselected), log `/tmp/oh-my-vllm-py04-cpu.log`; Rust workspace 58/39/26,
  fmt, line width, Ruff, Clippy and hooks passed. Owned GPU/worker processes and
  sockets were cleared. PY-04 remains **open**: caches still precede capture,
  one near-full shape does not bound all later captures, and the current full
  GPU, 147-operator, six-boundary and 12-row framework gates remain pending.
- Independently reviewed `5772834` is a PY-03 bounded graph-admission
  **candidate**, not a performance acceptance. Target owns 32 slots; draft
  and proposal share 32 with 16/4 eviction floors. Four observations and
  decayed frequency above twice the coldest eligible resident are required
  for replacement. Metadata is bounded, failed captures keep the old graph,
  and a first failed capture discards its unowned family pool handle. New
  captures defer to eager below 4 GiB CUDA free memory; CUDA OOM remains fatal.
  Clean `5772834` on B200 UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad` passed 3/3 focused real
  graph tests, including nonempty capture failure then new-handle retry;
  log `/tmp/oh-my-vllm-py03-clean-focused-5772834.log`. Candidate
  `scripts/test.sh cpu` passed 255 tests/70 subtests (223 GPU deselected),
  log `/tmp/oh-my-vllm-py03-cpu-full-candidate.log`; Rust workspace, fmt,
  line width, Ruff, Clippy and hooks passed. With the same clean source,
  external first-come shim versus LFU, FA 1400/GDN 128 and MTP4 batch 4,
  input 36832/output 512, two warmups and five measured runs gave median
  throughput 213.797→214.324 token/s (spreads 0.324/0.361%) and maximum
  TTFT 7.50258→7.50947 s (spreads 0.390/0.693%); no stable speed gain is
  claimed. Draft/proposal budget-eager calls fell 182/35→63/20, while
  captures rose 25/7→30/8, with 3/3 evictions and zero recent recaptures.
  Logs: `/tmp/oh-my-vllm-py03-{oldmode,lfu}-clean-5772834.log`.
  Exact LRU was rejected after 208.301 token/s in an earlier dirty-candidate
  diagnostic. A full-model FA 2333/GDN 128 diagnostic held 32 MTP graphs at
  179080003584 peak reserved bytes, but minimum observed pre-capture free
  memory was 11524046848 bytes (10.73 GiB), so the physical 4 GiB gate was
  not triggered;
  log `/tmp/oh-my-vllm-py03-nearfull-final-candidate.log`. All owned GPU
  processes exited. PY-03 remains **open/candidate** pending shape-churn,
  262k, low-headroom and 12-row framework evidence.
- Reviewed `2837c49` adds a shared MTP graph-capture cooldown to both Serve and
  Bench after a recent draft/proposal recapture; resident graph hits continue,
  new misses run eagerly for 32768 decisions, then admission resumes. CPU
  regressions cover boundaries, both family floors and hot-workset recovery.
  `scripts/test.sh cpu` passed 266 tests/70 subtests (225 GPU deselected), log
  `/tmp/oh-my-vllm-py03-churn-cpu-full.log`; B200 focused GraphCache passed
  4/4 on `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`, log
  `/tmp/oh-my-vllm-py03-churn-graph-gpu4.log`. Rust workspace, fmt, line width,
  Ruff and Clippy passed before the identical source/test commit; hooks passed
  at commit. Independent final source and Serve evidence review found no
  correctness blocker.
  The dirty-candidate MTP batch-4 row passed its local capture and comparison
  audit at 99.60% baseline throughput and 96.09% baseline maximum TTFT; raw
  `/tmp/oh-my-vllm-py03-churn-bs4-candidate.json`. Same-B200 real Serve A/B
  ran 12 workload waves/36 requests/36864 output tokens per side against
  clean `ed63177`, with identical response hashes; candidate versus old
  draft/proposal captures were 31/10 versus 40/11. Excluding the cold first
  wave, summed wave-maximum latency was 33.409 versus 33.483 s, a single
  diagnostic pair with no stable speed conclusion. The original A/B owner is
  `/tmp/oh-my-vllm-py03-serve-ab-original.py` (SHA-256 `7d23377d282b0323d6eaf5ecde99c5d95a8793f35e25a0e2613a0757a41ca971`);
  A/B artifacts/logs are `/tmp/oh-my-vllm-py03-serve-{candidate,old}-shift.*`.
  A separate candidate Serve run completed 88 waves/264 requests/270336 output
  tokens, observed two cooldowns, captured new draft/proposal shapes after
  expiry, and retained 20/12 family graphs; artifact/log
  `/tmp/oh-my-vllm-py03-serve-candidate-recovery.*`. All owned workers, GPU
  processes and IPC listeners exited. Those Serve inputs reached about 21k
  tokens with 1024 output, and the long run has no old comparison. PY-03 stays
  **open/candidate** until the clean final 12-row gate; 262k and physical
  low-headroom capture remain separate limits.
- Independently reviewed `3bbf926` is a PY-06 non-greedy readback
  **candidate**. The runner draws per request on the GPU, then combines
  rows for one final host transfer before request-local verify/commit;
  duplicate request IDs fail before forward or RNG draw. The pure-greedy
  path is unchanged. Clean `3bbf926` on B200 UUID
  `GPU-a4b4fc91-7347-839a-dd09-b1f0818ef5ad`, four requests × five
  rows × 248320 logits, temperature 1 without penalties/grammar, five
  warmups plus 20 measured pairs: per-request 0.740048 ms versus combined
  0.674659 ms median (spreads 3.282/2.047%). Raw log:
  `/tmp/oh-my-vllm-py06-clean-ab-3bbf926.log`. Pinned H2D staging had
  no stable material benefit across 64–32768 int64 values, so it was not
  changed; raw log `/tmp/oh-my-vllm-py06-h2d.log`. MTP eager fallback and
  output-dependent next-step scheduling remain serialized, and top-k/top-p
  can synchronize during its tie check. Focused B200 tests passed 32/32,
  log `/tmp/oh-my-vllm-py06-gpu-focused-r2.log`; `scripts/test.sh cpu`
  passed 259 tests/70 subtests (224 GPU deselected), log
  `/tmp/oh-my-vllm-py06-cpu-full.log`. Rust workspace, fmt, line width,
  Ruff, Clippy and hooks passed. Owned GPU processes exited. PY-06 remains
  **open/candidate** pending current full GPU and 12-row framework gates
  plus representative whole-step overlap evidence.
- KRN-06 was investigated but its proposed optimization is **open/no-go**;
  the original streaming-store `"memory"` clobber remains. CUDA 13.1's
  `__stcs(uint2*)` still has that compiler clobber, so the audit's suggested
  substitution would not remove it. A reviewed no-clobber trial needed
  shape/dtype/layout/device and disjoint-output FFI guards to remain safe.
  The trial's 7/7 direct-FFI rejection and 4/4 BF16 exact/CPU FP64 tests
  passed on B200, but these are **rejected-candidate tests**, absent from the
  retained source. Clean `2c85dfa` norm/add_norm before was 25/26 formal;
  the affected 2496-row case passed, while unrelated 1248-row add_norm failed
  by a near-zero bound. The dirty candidate was also 25/26, with the same
  failure; isolated 1248-row 3×20 repeated as PASS, exposing cross-run
  variance. Raw reports: `/tmp/oh-my-vllm-krn06-before-2c85dfa.json`,
  `/tmp/oh-my-vllm-krn06-after-dirty.json`, and
  `/tmp/oh-my-vllm-krn06-1248-repeat-dirty.log`. A same-GPU, single-CPU
  5+100 interleaved old/no-guard/guard diagnostic gave direct FFI total
  medians 21.980/21.971/22.302 microseconds and output-allocation call
  medians 33.070/33.143/33.528 microseconds; per-call spreads exceeded
  10%, so no stable full-call gain is claimed. Logs:
  `/tmp/oh-my-vllm-krn06-host-trio-dirty.log` and
  `/tmp/oh-my-vllm-krn06-wrapper-trio-dirty.log`. The patch was reverted,
  no source/test commit was made, and all owned B200 processes exited.

At this 2026-09-28 checkpoint, PY-03/04/06, SRV-08, PY-05/07, KRN-10 and
the full operator/GPU/12-row gates were still pending. The 2026-09-29
checkpoint above supersedes that queue. KRN-06 remains open/no-go until a
different safe optimization demonstrates whole-call benefit. See the
[current audit index](audit-2026-09-23.md) for each status and limit.

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
