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

### In progress — uncommitted working-tree changes

The following files have partial Batch 2/3 fixes applied by subagents that were
stopped mid-task. They have been verified to compile and pass ruff, but have NOT
been committed. The new session must verify, complete, and commit them.

**Batch 2 — Evidence integrity (partial):**
- `benchmarks/compare_vllm.py` — EVD-01: default reps→5, warmup→2, spread warning.
- `benchmarks/measurement.py` — EVD-11: missing cache root now appends to `problems`.
- `benchmarks/ttft.py` — EVD-02: `_parse_bench_config()` reads BENCH_CONFIG log line;
  EVD-13: `TVM_FFI_CACHE_DIR` added as fourth cache root.
- `crates/zmq-worker/src/main.rs` — EVD-02: `info!("BENCH_CONFIG", …)` at startup.
- `tests/test_kernel_reference.py` — EVD-06: test asserts fixed expected file set.
- `tests/test_context_boundary.py` (untracked) — EVD-07: six skip-placeholder tests.

  **Blocker:** `tilelang-reference.json` is missing the `__init__.py` entry
  (SHA256: `601991901a71a8a9d7d4f4e62a916c83b46e7cb4a79a1b2cd4469ace63b461e3`).
  Add it before the EVD-06 test will pass.

  **Incomplete:** EVD-12 not started.

**Batch 3 — Kernel/worker contracts (host-only, KRN-01/04/05 PY-01/02 MNT-02):**
- `python/oh_my_vllm/kernels/decode_attention.py` — KRN-01: assert group_size ≤ 5
  in `decode()` when `starts` is supplied.
- `python/oh_my_vllm/kernels/elementwise.py` — KRN-04: int32 overflow guard in
  `silu_mul`; MNT-02: empty-tensor early return in `silu_mul` and `delta_gates`.
- `python/oh_my_vllm/kernels/fp8.py` — KRN-04: int32 overflow guard in `quantize`.
- `python/oh_my_vllm/kernels/gdn.py` — KRN-05: dtype, shape, and stride validation
  moved into `normalize_qk` (previously only in `prefill`).
- `python/oh_my_vllm/models/qwen.py` — PY-02: assert `w.shape[0] % 128 == 0` for
  every FP8-scaled projection part in `Checkpoint.linear`.
- `python/oh_my_vllm/worker/batch_plan.py` — PY-01: debug-mode assertion that
  writable FA tail pages are disjoint across requests in `validate_batch`.
- `tests/test_batch_plan.py` — tests for KRN-01 host guard and PY-01 FA sharing.
- `tests/test_validation_guards.py` — standalone guard tests for KRN-04/05 PY-02
  MNT-02; 23 tests pass, ruff clean.
- KRN-02/03 deferred (device code; require GPU re-run).

### Remaining batches (not started)

**Batch 4 — Scheduler hardening:** SCH-01/02/03/04/05, SRV-05/07.
**Batch 5 — Measured performance:** PY-03/04/05/06/07, SRV-08/09/10, SCH-06/07,
  KRN-06/07/08/09/10. Measure before keeping each.
**Batch 6 — Cleanup:** EVD-04/05/08/09/10, MNT-01/03/04.

After all batches: update `audit-2026-09-23.md` (and `.zh.md`) status fields to
`fixed <sha>`, then `git push origin main`.

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
