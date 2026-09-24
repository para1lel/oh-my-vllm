# Handoff — 2026-09-24

## Current state

The CUDA migration is complete and accepted. The next task is to fix the findings of
the whole-repository code audit.

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

## Open work: code audit remediation

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
  KRN-01/04/05, PY-01/02, MNT-02 fixed. KRN-02/03 deferred (device code; require GPU re-run).
- KRN-01: `decode_attention.py` asserts `group_size ≤ 5` when `starts` is supplied.
- KRN-04: `elementwise.py` and `fp8.py` guard against int32 overflow.
- KRN-05: `gdn.py` moves dtype/shape/stride validation into `normalize_qk`.
- PY-01: `batch_plan.py` debug-asserts FA tail pages are disjoint across requests.
- PY-02: `qwen.py` asserts `w.shape[0] % 128 == 0` for every FP8-scaled projection.
- MNT-02: `elementwise.py` early-returns on empty tensors in `silu_mul`/`delta_gates`.
- Tests added: `tests/test_validation_guards.py`, `tests/test_batch_plan.py`,
  `tests/test_independent_decode_attention.py`.

**Batch 2 — Evidence integrity** (`895d57b`, 2026-09-24): EVD-01/02/06/07/11/12/13 fixed.
- EVD-01: `compare_vllm.py` defaults to reps=5, warmup=2; emits a spread warning.
- EVD-02: `ttft.py` reads the actual BENCH_CONFIG log line from the worker via
  `_parse_bench_config()`; `zmq-worker/src/main.rs` emits `info!("BENCH_CONFIG", …)` at startup.
- EVD-06: `test_kernel_reference.py` asserts the exact expected file set in the
  frozen reference JSON.
- EVD-07: `tests/test_context_boundary.py` (six skip-placeholder tests for
  REQ-CONTEXT-001 boundary runs).
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
- SRV-05/07 remain open (cancellation safety and side-channel overwrite require
  larger changes; tracked in audit).

**Batch 6 partial — Cleanup** (`62101a3`, 2026-09-24): MNT-04 fixed.
- Removed dead `abort_request()` method and `AbortMsg` import from `client.rs`.
- Updated protocol doc comment in `lib.rs` to match the live wire format.

### Remaining open findings

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
