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
  because its six boundary tests still skip. The four audit risks marked
  "Not verified" have not yet been closed.
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

Remaining work: PY-03..07, SRV-08..10, SCH-06/07, KRN-06..10,
EVD-07/09, MNT-01..03, the four unverified risks, and current
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
