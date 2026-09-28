# Benchmark evidence

This directory holds formal, summarized acceptance evidence. Do not add GPU profiling
traces or temporary tuning records. See [docs/acceptance.md](../../docs/acceptance.md)
for interpretation and a dated index of every artifact.

## Current

- `2026-09-29-audit-final-operators.json` summarizes clean `e3c42e0` full
  147/147 formal output and frozen TileLang decisions on a pinned B200. It
  records source/compiler/module provenance, all case decisions, no GPU
  interference, and the SHA-256 of the raw `/tmp` result; raw pairs stay out
  of this repository.
- `2026-09-29-audit-final-framework.json` summarizes the same clean source's
  12 accepted framework rows against the frozen 2026-09-22 EngineCore baseline.
  It preserves 2 warmups, 5 measurements, per-row TPS/TTFT/spread and steady
  audit, plus the rejected high-spread prefix batch-1 attempt and its passing
  same-GPU repeat. The initial all-row owner exited with failure; the summary
  combines its 11 accepted rows with the explicit one-row repeat.

The following earlier artifacts retain their status at their own source commits;
their statements that later gates were pending are historical.

- `2026-09-28-audit-evd07-context-boundary.json` records six clean `bd8e21e`
  ordinary/MTP4 258048+4096 boundary runs at batch 1/2/4 on one B200.
  All finish without OOM or preemption; MTP4 proposals are nonzero.
- `2026-09-28-audit-evd07-long-context-http.json` records clean `8dfc97b`
  four-request MTP4 strict-JSON service smoke beyond 131072 prompt tokens,
  including prefix reuse and listener/worker cleanup. The current full
  GPU/147-case/12-row gates remain pending.
- `2026-09-28-audit-krn10-attention-before.json` and
  `2026-09-28-audit-krn10-attention-after.json` summarize the 16 selected
  attention cases on clean `6ba9046` and `231f066`, respectively. All selected
  after cases pass output and frozen TileLang comparison. The full 147-case
  matrix and current 12-row framework gates remain pending.
- `2026-09-28-audit-krn08-operators.json` records clean `40e3e57` affected
  formal evidence: all 66 selected normalization, Q/K, recurrent, and
  convolution cases pass the frozen TileLang comparison and record one fast,
  zero generic CUDA host dispatches. It is not a new 147-case decision.
- `2026-09-28-audit-mnt03-recurrent-operators.json` contains clean `0394727`
  affected formal evidence: all 8 selected recurrent cases pass output and
  frozen TileLang gates on a pinned B200. It is not a new 147-case decision.
- `2026-09-28-audit-mnt02-krn04-operators.json` contains clean `312c54b`
  formal evidence for all 60 selected `quant`, `silu_quant`, `gates`, and
  `recurrent` cases. Every selected case passes output verification and the
  frozen TileLang comparison. It is a subset, not a new 147-case decision.
- `2026-09-28-audit-gates-confirm-operators.json` confirms all 13 `gates`
  cases on a second pinned B200 after one rejected 59/60 subset attempt. The
  rejected attempt is described in the acceptance index; its cause is unknown.
- `2026-09-28-audit-p1-operators.json` contains the clean `96e4ecc` affected
  formal subset: 29/29 selected `prepare_attention` and `attention` cases
  pass on a pinned B200. It is not a new full 147-case decision. Current
  12-row framework performance remains unmeasured after hot-path repairs.
- `2026-09-22-refreshed-enginecore.json` is the frozen 12-row official vLLM
  EngineCore baseline, collected at upstream `e9f169d16b9408bb9ae44f75072b91a5521d733c`.
  - Each `rows[].artifact` is the exact input `benchmarks/ttft.py` expects.
  - Serialize a row as `json.dumps(artifact, indent=2)` plus a newline to
    reproduce its `rows[].sha256`.
  - Do not regenerate this file without explicit user authorization.
- `2026-09-22-cuda-framework.json` contains the 12 accepted CUDA rows from clean
  `c36d1c9`.
  - Every row includes the candidate's raw repetitions and a `baseline_sha256`
    linking it to the frozen row.
  - It also keeps all prior, failed and excluded attempts.
- `2026-09-22-cuda-operators.json` contains all 147 formal operator decisions.
- `2026-09-22-cuda-features.json` contains:
  - Default-selection correctness
  - Boundary runs
  - Service constraints and lifecycle checks
  - Agentic readback, with generated-answer limitations
  - The cleanup snapshot

## Historical

The remaining JSON files record earlier stages, listed in the acceptance index:

- The 2026-09-19 adapter
- The 2026-09-21 serving, V2 and independent-runtime stages
- The 2026-09-22 TTFT, performance-gap and TileLang stages
- The CUDA milestone subsets

These results were correct for the implementation measured at the time.
Pre-2026-09-22 rows use the original baseline and a 3-repetition protocol, so
they are not current acceptance.

The `baseline_*`, `clean_*` and `mtp_*` files and `logs/` are early captures from
the vLLM HTTP serving endpoint. Their settings, inputs and timing boundary differ
from the EngineCore protocol. Never use them as a denominator.

Some older artifacts use the erroneous "Qwen3.5" label or the
`qwen3.5-27b-fp8` service ID. They are kept as-is because they record the actual
executions. The model is Qwen3.8-27B-FP8.
