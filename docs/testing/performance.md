# Operator and framework measurements

## Release binary for acceptance

Compare the current runtime with the reviewed semantic contract on CPU before GPU collection:

```bash
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/tooling/test_performance_metadata.py::test_current_runtime_contract_matches_semantic_review -q
```

This checks source hashes, providers, headers, compiler flags, and version pins without CUDA initialization.
Change `performance/contract.json` only after semantic review. Keep the source and provider equality checks active.

The phase-latency collector reads repository `target/release/oh-my-vllm-zmq-worker`.
Set this directory explicitly. The harness must use the candidate binary from this build:

```bash
export CARGO_TARGET_DIR="$PWD/target"
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
```

Use this build before framework, context, or service acceptance.
Set `EVIDENCE_DIR` to a writable external directory for full raw records and logs.

## Formal operator gate

```bash
scripts/with-env.sh python benchmarks/kernels.py --list
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_KERNEL_BACKEND=cuda python benchmarks/kernels.py --numerical-only --output "$EVIDENCE_DIR/operators.json"
```

Use a new external output path. [Static cases](../../development/kernels/cases.py) define the matrix.
All 539 cases must pass independent numerical, rounding, mutation, and state contracts.
The original 317 case IDs stay. The supplements add 200 legal production cases and 22 attention-gate cases.

Partial selection reports `coverage.complete=false` and cannot prove full-matrix acceptance alone.
Use `--operations NAME` or repeated `--case-id ID` for diagnosis.

Omit `--numerical-only` for paired tuning measurements after numerical verification.
Each case then has three rounds of twenty interleaved pairs, with one hundred graph repetitions per sample.
The hierarchical-bootstrap one-sided 95% bound gives time saved against the repository CUDA baseline.
Operator timing is diagnostic. Complete framework phase latency supplies the performance gate.

Restore persistent state between cases and rounds.
Normalization, Q/K, recurrent, and convolution verification must have one fast and zero generic host dispatches.
Graph replay counts cannot replace host dispatch counts.
The complete GPU matrix validates numerical-only collection. Separate selected timings validate candidate comparisons.
Use [kernel development](../kernels.md) for provenance and profiling separation.

## Hardware-counter diagnosis

Use the [profiler worker](../../development/kernels/profile_worker.py) to select one complete formal operation after warmup.
The following command was tested with Nsight Compute 2026.1.1:

```bash
scripts/with-gpu.sh scripts/with-env.sh ncu --target-processes all --nvtx --nvtx-include 'profile-operation/' --kernel-name 'regex:scan_kernel' --launch-count 1 --cache-control none --clock-control none --section SpeedOfLight --section Occupancy --section WarpStateStats --section MemoryWorkloadAnalysis --export "$EVIDENCE_DIR/scan" python -m development.kernels.profile_worker --case gdn_prefill-aaae0f6f08 --backend cuda
ncu --import "$EVIDENCE_DIR/scan.ncu-rep" --page raw --csv --log-file "$EVIDENCE_DIR/scan.csv"
```

Use a new external report path. The selected kernel is only part of the operation.
This command leaves clock and cache state uncontrolled. Its replay duration is diagnostic, not acceptance timing.
CSV output includes a unit row. Keep that row when reading counter values.

Use [measurement guidance](../profiling/measurement.md) for concurrency, replay, and counter interpretation.
Release owned processes after collection. Do not run another CUPTI profiler concurrently.

## Framework performance

Build the current release binary and set a new external output path for each attempt.
Select one of the fifteen rows in [requirements](../requirements/performance.md#req-perf-001-phase-latency).

```bash
scripts/with-gpu.sh scripts/with-env.sh python -m benchmarks.framework --mode ordinary --batch-size 1 --input-len 32768 --output "$EVIDENCE_DIR/ordinary-32768-1.json"
```

Use modes `ordinary`, `mtp4`, `prefix`, and `dspark` with input 32768 and batches 1/2/4.
Use MTP4 input 131072 with the same batches. Mode `prefix` uses MTP4.
Set `OH_MY_VLLM_DRAFT_MODEL` or `--draft-model` for DSpark.
Output count is 4096. Each attempt has two full warmups and five measured repetitions.

Prefix rows must hit 32144 tokens per request.
Other rows reset prefix reuse before each repetition.

The collector records each request's submission, first token, last token, and step boundaries.
It calculates prefill and decode independently from those boundaries.
Each repetition uses the maximum request interval for each phase.
Cleanup follows the last token and has a different timing field.

The worker buffers effective shapes, state access, draft operations, and compiled operation identities.
It writes the trace after measured request execution.
Formal analysis uses the canonical model and official peaks.
Unidentified operations block acceptance.
GPU event intervals are diagnostic stream intervals. They can include host submission gaps.

Except for prefix-hit prefill, each phase must have a wall-time median at most two times its theoretical lower-bound median.
Its wall-time spread must be at most 10%.

Record the prefix-hit prefill latency ratio without a two-times gate.
Its spread check stays active. Prefix-hit decode keeps the two-times gate.
Keep complete failures and interrupted attempts. Investigate a failure. Run that workload again with two full warmups and five measurements.

Connect source, binary, checkpoint bytes, CUDA module, hardware, capacities, and runtime packages to the attempt.
Keep source unchanged during collection.
Compile/cache audit must show steady-state execution.
Available logs cannot exclude all silent in-memory compilation.
Keep raw logs and traces in external storage.

Use portable summaries for review. Formal verification uses full original records.

Use `--diagnostic` for uncommitted-source experiments.
Diagnostic records do not supply formal acceptance.
Profile independently from measured acceptance.

## DSpark threshold and draft statistics

`--dspark-confidence-threshold` selects fixed or shorter proposals and is recorded with the observed worker configuration.
Its initial setting is `0.2`. Use `0.0` for fixed-count proposals up to seven.
Output, context, and grammar limits can decrease the proposal count.
Compare each setting through its scheduled/accepted draft counts and measured performance.
Acceptance rate is total accepted drafts divided by total scheduled drafts in measured work.

Use `bench --prompt-file PATH` for experiments with text or tool histories different from formal inputs.
The file has one request per line, with whitespace-separated token IDs.
Its row count must equal `--batch-size`. Each row length must equal `--input-len`.
The command checks these constraints before model or GPU loading.
Keep text sources, token hashes, templates, and sampling settings with the experiment records.

Compare loaded draft configuration and weight hashes with the requested checkpoint bytes. They must agree.
Verified drafts are scheduled candidates. Returned next-step proposals have a different diagnostic counter.
Portable summaries cannot replace original evidence for formal verification.

Use the [acceptance index](../acceptance.md) for source identities and measured scope.
