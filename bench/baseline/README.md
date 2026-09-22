# Benchmark evidence

`2026-09-22-tilelang-acceptance.json` preserves all twelve final raw candidate
and frozen baseline sets, prior failed attempts and interference exclusions.
The clean measured implementation is da75c02. On user instruction, stability
changed from5% to10% for both metrics and both engines. Original embedded
`candidate.comparison` decisions remain unchanged; use each row
`comparison_under_current_policy` for the new verdict. All twelve pass.
`2026-09-22-tilelang-correctness.json` records the unchanged full GPU suite,
actual-model probes, six boundaries, text/preemption and real MTP service/agentic
evidence with generated-answer limitations. Temporary operator records are omitted.

The historical baseline_*, clean_* and mtp_* JSON files are existing vLLM serving
captures (OpenAI endpoint). Their settings, input data, timing boundary and MTP
configuration differ from the current matched EngineCore protocol. Preserve them
as historical data; do not use them as the denominator for current acceptance.

2026-09-19-paired-investigation.json preserves completed matched pairs from this
session, including failed ratios, actual repetitions and executable/source identity.
It is explicitly not a final acceptance matrix. At that historical stage ordinary
bs1 needed a variance repeat and several modes/batches were missing or below95%.
The complete acceptance artifact below supersedes that status. Some early rows lack a
captured CPU affinity mask, represented by null. Later contaminated/incomplete
pairs are excluded rather than assigned performance numbers.

Use benchmarks/compare_vllm.py and docs/testing.md for new measurements. No GPU
profiling traces should be added here. docs/handoff.md tracks the current work.

2026-09-19-acceptance.json is the complete nine-row matrix on clean 12d227d:
two warmups, three measurements, identical single-core affinity within each
pair. All rows pass. It includes exact commands, protocol settings and original
result rows. The MTP batch-2 variance repeat is preserved separately rather than
replacing its original measurements. See docs/acceptance.md for interpretation.

2026-09-19-batch-text.json preserves full generated text and observed counters
for ordinary and MTP/prefix two-request preemption checks. Its dirty build
identity is explicit; these are semantic smoke results, not throughput data.

`2026-09-21-independent-stage1.json` records the first independent-runtime
milestone: new-environment operator/CPU tests, short eager model output and the
transitional adapter's MTP4 service checks. It is not the final migration
acceptance or a replacement performance baseline.

`2026-09-21-independent-stage2.json` records independent Worker/MTP/graph
correctness, the FP8 backend defect and fix, final real service/probe results and
one explicitly non-acceptance throughput diagnostic. The default adapter removal,
real independent oh-my-pi tasks and final monitored nine-row matrix remain pending.

`2026-09-21-independent-stage3.json` records removal of legacy runtime/test
dependencies, full target-environment GPU/CPU tests and real file/module/library
independence audits. Final workload acceptance is recorded separately.

Model naming correction (2026-09-22): the target is **Qwen3.8-27B-FP8**.
Older raw logs and frozen JSON may contain the erroneous Qwen3.5 label or the
previous `qwen3.5-27b-fp8` service ID. Those are preserved as records of the actual
historical executions, not current model naming. Current code/docs/service IDs use
Qwen3.8. New measurements must use the corrected name.
