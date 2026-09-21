# Benchmark evidence

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
