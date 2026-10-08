# Current state and open work

Updated: 2026-10-08.

## Current implementation

The independent runtime supplies Rust service/scheduling/logical KV and Python GPU execution for the target model on one B200.
CUDA is the owned default backend. TileLang stays pinned for comparison.
Semantic IR compiles prefill, target decode, MTP draft, and four-step proposal inside manual graph lifecycles.
Block 784, ordinary FP32 state, MTP BF16 state, and existing features stay active.

The twelve-chapter Twine tutorial includes current core source and keeps the user-requested preview on port 18084.
Its address, PID, and maintenance commands are in ignored `LOCAL.md`.

The [acceptance index](acceptance.md) binds each historical measurement to its source.
The latest full 147-case/twelve-row performance result applies to `619c9d9`.
Later IR correctness changes passed 549 GPU tests, 70 subtests, and 308 CPU tests without a new full performance collection.

## Document and portability refactor

Project English Markdown now uses ASD-STE100 Issue 9 rules with a project technical glossary and Chinese translations.
Architecture includes design. This file includes the former plan.
Kernel development combines CUDA/TileLang guides.
The audit keeps open problems and a 48-finding fix index.

The acceptance document is the sole evidence index.
Key ADR decisions and replacement relationships stay.
Daily repair narratives stay in Git history.

Environment selection uses explicit prefix, active virtual/conda environment, or repository `.venv`.
Model selection uses `OH_MY_VLLM_MODEL` or explicit `--model` consistently across execution tools.
Rust library launch rejects an empty model before it starts a child.
Full GPU tests reject missing model configuration before GPU selection.
CPU model integrations skip explicitly. Pure validation tests stay active.

All 66 initial tracked evidence files have byte-identical ignored local copies with SHA-256 comparisons.
Tracked derived summaries keep numeric measurements, versions, and original hashes and remove host identities and raw payloads.
Formal comparators reject derived envelopes and keep the current gates.
New servers must measure their own matching baseline.
Local configuration and original evidence are ignored by Git.

## Verification for this change

Rust workspace tests passed 124 cases. Formatting, the 100-character limit, Clippy, and the release build passed.
Ruff formatting/checks passed for Python, scripts, benchmarks, tests, and development tools.
The final CPU suite passed 329 tests and 70 subtests, with 241 GPU tests deselected.
After the last checker changes, its 13 focused tests passed again.

With the model unset, six selected tests and two subtests passed. Thirty-three checkpoint integrations skipped explicitly.
The two GPU smoke tests each generated 64 tokens from the target checkpoint.
Ordinary execution selected the model from the environment.
MTP4 used explicit CLI selection with a deliberately invalid model environment value.
These runs show the intended selection precedence through model execution.

The tutorial rebuild, Node excerpt tests, eight Playwright tests, and Rust trace test passed.
Trace formatting and Clippy passed.
Browser plugin not available: the configured Playwright Chromium supplied browser verification.
Desktop/mobile screenshots, the two themes, source links, navigation, formulas, fields, and the scheduler experiment passed examination.
The user-requested static preview stays on port 18084. All smoke workers exited and released GPU resources.

Reviews by other agents included all 27 English/Chinese document pairs, dictionary meanings, technical parts of speech, and implementation contracts.
They also compared all 66 archived files and 64 derived JSON summaries with source bytes and hashes.
The baseline extraction reproduced all twelve row hashes.
The hook checks pairing, protected commands, numbers, glossary definitions, links, mechanical STE rules, and host information.
Approved meanings and full translation fidelity still depend on review, as [writing rules](writing.md) specify.

The full GPU regression suite, six maximum-context runs, 147-case operator collection, and twelve-row performance collection were not repeated.
Inference computation did not change. This refactor changes documents, deployment selection, test setup, and evidence handling.
Current-source performance still depends on a new full collection. Historical acceptance keeps its measured source and configuration.

## Open work

- Complete user review of the expanded tutorial and address specific teaching gaps.
- Test cross-shape 262k graph eviction/recapture with measured memory pressure, as `PY-04` specifies in [audit](audit.md).
- Show bounded compiler storage with sustained shape changes before a broader long-lived-worker claim.
- Keep `PY-06` host-overlap work measurement-driven. The pinned readback experiment showed no material full-call gain.
- Keep the rejected `KRN-06` candidate withdrawn until new full-operation evidence shows a benefit.
- Design performance thresholds from roofline analysis instead of vLLM measurements in a new requirement change.

The existing 95% throughput, 110% TTFT, and 10% spread gates stay active.
Read [requirements](requirements.md), [testing](testing.md), and [audit](audit.md) before the next implementation task.
