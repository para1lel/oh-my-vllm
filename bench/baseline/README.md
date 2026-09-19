# Benchmark evidence

The historical baseline_*, clean_* and mtp_* JSON files are existing vLLM serving
captures (OpenAI endpoint). Their settings, input data, timing boundary and MTP
configuration differ from the current matched EngineCore protocol. Preserve them
as historical data; do not use them as the denominator for current acceptance.

2026-09-19-paired-investigation.json preserves completed matched pairs from this
session, including failed ratios, actual repetitions and executable/source identity.
It is explicitly not a final acceptance matrix. Ordinary bs1 needs a variance
repeat; several modes/batches remain missing or below95%. Some early rows lack a
captured CPU affinity mask, represented by null. Later contaminated/incomplete
pairs are excluded rather than assigned performance numbers.

Use benchmarks/compare_vllm.py and docs/testing.md for new measurements. No GPU
profiling traces should be added here. docs/handoff.md tracks the current work.
