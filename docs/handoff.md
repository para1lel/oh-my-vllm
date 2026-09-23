# Handoff — 2026-09-23

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
[audit-2026-09-23.md](audit-2026-09-23.md). It contains:

- 4 P0 findings, all in serving availability.
- 17 P1 latent silent-wrong or unsafe contracts.
- 16 P2 performance, robustness or evidence findings.
- 14 P3 findings in tests, evidence and maintainability.

The audit found no wrong-token defect on the default path, and the accepted
evidence above stands.

Work through the batches in the order given in the audit, and mark each finding's
status in its index:

1. Serving availability
2. Evidence integrity
3. Kernel and worker contracts
4. Scheduler hardening
5. Measured performance
6. Cleanup

Two rules apply to every batch:

- A change to `kernels.cu` device code requires re-running its formal operator
  cases.
- A change to the hot path requires re-checking the framework rows before
  claiming the 12-row result still holds.

Known issues that affect operators today (details in the audit):

- A single request's sampling, grammar or prepare error stops the serving
  engine. Every later request then gets 503 until restart (SRV-01).
- Killing the Rust process with SIGTERM or SIGKILL orphans the Python worker. The
  orphan keeps holding the GPU and its `with-gpu.sh` lock (SRV-02). After an
  abnormal exit, check `nvidia-smi` and stop only the processes you own.
- `benchmarks/compare_vllm.py` is a historical nine-row tool. It does not enforce
  current gates, so use `benchmarks/ttft.py` (EVD-01).

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
