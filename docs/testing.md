# Tests and acceptance procedure

Use the environment in [development](development.md).
Set `OH_MY_VLLM_MODEL` for tokenizer or target-model integration tests.
Use [requirements](requirements.md) for active gates and [acceptance](acceptance.md) for historical results.

## CPU and GPU suites

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python scripts/format_cuda.py --check
scripts/with-env.sh python scripts/check_docs.py
scripts/test.sh cpu
scripts/test.sh full
```

`scripts/test.sh` builds the test binary and supplies `OH_MY_VLLM_TEST_BINARY`.
CPU mode selects `not gpu` with GPU visibility empty.
Full mode must use a model location before selection of an idle UUID-pinned B200.
CPU tokenizer/HTTP integrations skip explicitly when the model is unset.
Pure validation tests stay active without a checkpoint.

Selected pytest arguments follow `cpu` or `full`.

Tests include scheduler transactions, cache ownership, cancellation, RPC framing, serving, inference-path FP64 references, and compiled model units.
GPU coverage includes all six maximum-context cases and changed-metadata graph replay.
Full-model drift tests use their documented model-level bounds in addition to strict operator-level tolerances.
Use scripted workers for protocol regressions and the target checkpoint for GPU/agentic acceptance.
Record warnings and skipped tests with their reasons.

## Accuracy

Use independent references with CPU FP64 computation on rounded inference inputs.
Tests must include prefill, decode, boundary writes, strided views, grouped MTP verification, and speculative rollback.
BF16 operator output uses `atol=rtol=0.03`.
Recurrent state must have NRMSE at most 1% and maximum absolute error at most 2% of reference peak.
Examine each written slot and unchanged protected state.

For fused chains, include all rounding, scale, copy, and reduction behavior.
See `tests/probe_worker.py` and the applicable kernel tests.

## Offline execution and numerical probes

Build the release binary before these commands. Set the environment and model from [development](development.md).

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/text-check.ipc --max-tokens 64
```

Use `--num-speculative-tokens 4` for MTP4.
Use `--context-repeats 100 --prefix-hit` to cross a 784-token boundary. The second execution must have a prefix hit.
Use `--binary` to select a different build.
Offline fixed output ignores EOS.

`run --prompt-file PATH` reads whitespace-delimited token IDs, one request per line, with at most 32 requests.
`--arrival-interval N` gives an interval of N scheduling steps between admissions.
`--prefix-hit` first seeds the prompts and then must have positive cache hits.
The global `--scheduler-blocks` limits the Rust pool. Its value must be at most the reported worker capacity.
The `bench` subcommand also has `--warmup` and `--repetitions`.

These two-request probes use 2048 input and 1024 output tokens per request:

```bash
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-text.ipc --scheduler-blocks 10
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-batch.py --socket /tmp/batch-mtp-text.ipc --scheduler-blocks 18 --num-speculative-tokens 4 --prefix-hit
```

They must have positive preemption counts and the correct city at the start and end of each output.
MTP4 must accept drafts. Prefix mode must have positive initial prefix hits.
Examine the printed text as a semantic smoke test.
A small pool alone does not show preemption.

For inference-path numerical diagnosis, use the executable probe as the worker interpreter:

```bash
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON="$PWD/tests/probe_worker.py" python scripts/smoke-text.py --socket /tmp/probe-text.ipc --max-tokens 32
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_WORKER_PYTHON="$PWD/tests/probe_worker.py" OH_MY_VLLM_PROBE_MTP=1 python scripts/smoke-text.py --socket /tmp/probe-mtp.ipc --max-tokens 32 --num-speculative-tokens 4
scripts/with-gpu.sh scripts/with-env.sh python scripts/probe-independent-model.py --max-tokens 32
```

The worker probe instruments model GQA, GDN prefill/recurrent verification, and MTP attention calls without replacement kernels.
It compares rounded inference inputs with independent references through CPU FP64 computation.
Eager mode disables capture so diagnostic copies observe inference requests.
The direct model probe supplies a short eager model diagnosis without Rust scheduling.
Keep these probes and eager mode disabled for performance acceptance.

## Release binary for acceptance

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
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_KERNEL_BACKEND=cuda python benchmarks/kernels.py --output "$EVIDENCE_DIR/operators.json"
```

Set `EVIDENCE_DIR` to a writable evidence directory in an external directory.
The harness uses static cases from `development/kernels/cases.py`.

Use `--operations NAME` to select an operation family.
Give `--case-id ID` for each selected case ID.
A selection with less than 317 case IDs reports `coverage.complete=false` and keeps the full per-case protocol.
All 317 case IDs must pass before matrix acceptance.

It verifies full output and written caches before timing.
Each case must have three rounds of twenty interleaved pairs and faster CUDA medians in all rounds.
The hierarchical-bootstrap one-sided 95% bound on time saved must be positive.

Each timing sample uses one hundred graph repetitions.
Restore persistent state between cases and rounds.
Normalization, Q/K, recurrent, and convolution verification must have one fast and zero generic host dispatches.
Do not substitute graph replay counts for host dispatch counts.
Use [kernel development](kernels.md) for provenance and profiling separation.

## Framework performance

Build the current release binary and set a new external output path for each attempt.
Select one of the fifteen rows in [requirements](requirements.md#req-perf-001-phase-latency).

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

Except for prefix-hit prefill, each phase must have a wall-time median at most three times its theoretical lower-bound median.
Its wall-time spread must be at most 10%.

Record the prefix-hit prefill latency ratio without a three-times gate.
Its spread check stays active. Prefix-hit decode keeps the three-times gate.
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

Use the [acceptance index](acceptance.md) for source identities and measured scope.

## Context and service acceptance

```bash
scripts/with-gpu.sh scripts/with-env.sh python benchmarks/context_boundary.py --binary target/release/oh-my-vllm-zmq-worker --raw-dir "$EVIDENCE_DIR/boundary-logs" --output "$EVIDENCE_DIR/boundary.json"
```

The six rows use 258048 input and 4096 output, ordinary/MTP4, and batches 1/2/4.
Make sure that output is full, source identities match, and workers stop.
Reject OOM or preemption.
Record peak allocated/reserved GPU memory and draft counters.
Raw boundary logs must stay in an external directory.

Start a dedicated MTP4 service in one terminal with the release binary and a unique IPC path.
Use Serving DEBUG logs for mixed-batch and cancellation evidence:

```bash
scripts/with-gpu.sh scripts/with-env.sh env RUST_LOG=info,oh_my_vllm_zmq_worker::serving=debug target/release/oh-my-vllm-zmq-worker --socket /tmp/service-acceptance.ipc --max-model-len 262144 --num-gpu-blocks 4200 --mamba-blocks 128 --num-speculative-tokens 4 serve > "$EVIDENCE_DIR/server.log" 2>&1
```

Run these commands from another terminal after service readiness.
Install the OMP executable on `PATH`, or supply its path with `--omp`.
Use the same explicit URL for all clients. The long-context tool has a different default port.

```bash
export SERVICE_URL="http://127.0.0.1:8000/v1"
scripts/with-env.sh python scripts/serving-acceptance.py --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output-dir "$EVIDENCE_DIR/constraints"
scripts/with-env.sh python scripts/serving-lifecycle.py --api chat --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/lifecycle-chat.json"
scripts/with-env.sh python scripts/serving-lifecycle.py --api responses --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/lifecycle-responses.json"
scripts/with-env.sh python scripts/long-context-acceptance.py --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/long-context.json"
scripts/with-env.sh python scripts/agentic-acceptance.py --api chat --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output-dir "$EVIDENCE_DIR/agentic-chat"
scripts/with-env.sh python scripts/agentic-acceptance.py --api responses --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output-dir "$EVIDENCE_DIR/agentic-responses"
```

The lifecycle tool must find a shared mixed-request batch and worker cancellation/release in the supplied log.
Examine positive MTP proposals and accepted drafts in the server log.
Stop the service and its worker immediately after these tests.

The long-context gate checks strict JSON content, input more than 131072, prefix reuse after the first request, and positive MTP proposals per request.
Complete OMP tasks with the target checkpoint through each API as specified in [service contracts](serving.md#verification).
The tools must read files, and the model must receive their results.
Script success alone is insufficient.

Do the service gates again with a different DSpark worker:

```bash
scripts/with-gpu.sh scripts/with-env.sh env RUST_LOG=info,oh_my_vllm_zmq_worker::serving=debug target/release/oh-my-vllm-zmq-worker --socket /tmp/dspark-service-acceptance.ipc --max-model-len 262144 --num-gpu-blocks 4200 --mamba-blocks 128 --speculative-mode dspark serve > "$EVIDENCE_DIR/dspark-server.log" 2>&1
```

Supply `--speculative-mode dspark` and its `--server-log` to each acceptance script.
Use different evidence paths for each mode and API.
Run lifecycle checks with `--api chat` and `--api responses`.
The constraint and long-context scripts test the two APIs.
Each completed response ID must agree with the active mode and scheduled/accepted counts in the server log.

The OMP script forwards client HTTP through a temporary recording proxy without body changes.
It keeps client JSONL, forwarded requests, response IDs, and tool-result hashes.

The two source reads must complete without error. Their results must enter a subsequent model request before its last answer.
The last answer must cite the source files and use their results.
Examine answer meaning after the automated checks.
Stop the DSpark service and its worker after these tests.

Service scripts reserve each output file or directory before requests. An existing result path fails.
They keep `passed=false` until the applicable contracts pass.
Each received HTTP response is saved before content or log checks.
Failures keep received responses and an explicit error record.
Lifecycle streams keep the consumed SSE frames before disconnect.

## Long-duration agentic service

Daily commands skip this heavy test. Run each mode against a dedicated service with maximum model length `262144`.
Use a new external directory. Keep the source and release binary unchanged throughout collection.
Build the service and CPU request normalizer together:

```bash
scripts/with-env.sh cargo build --release --bins
scripts/with-env.sh python scripts/agentic-soak.py --mode mtp --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output-dir "$EVIDENCE_DIR/mtp-soak"
scripts/with-env.sh python scripts/agentic-soak.py --mode dspark --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/dspark-server.log" --output-dir "$EVIDENCE_DIR/dspark-soak"
```

Set `OH_MY_VLLM_MODEL` or supply `--model`. Install OMP on `PATH` or supply `--omp`.
Use the matching service for each command. Stop that service and worker after collection.
The collector owns four persistent OMP processes and temporary recording proxies.

Each process keeps its session ID and history through successive source-audit tasks.
Two sessions use Chat and two use Responses. A shared four-slot limit also includes summary calls.

Each lane must continue for at least `3600` seconds. There is no request-count minimum.
Initialize the clients after service readiness, then start the clock before their first requests.
First compilation, graph capture, and later new-shape costs enter the measurements.

For this collector, start the service with `--worker-step-timeout-seconds 600` and `--request-timeout-seconds 1800`.
Record these limits from diagnostics. They let cold compilation complete without separate warmup.

Do not insert intermediate drains or fault probes. After the hour, complete active work and drain requests.

Use OMP `max` for tasks and summaries, mapped to service `xhigh`.
The model context window is `262144` and its task output limit is `32768`, with reasoning tokens.
Use the local `qwen3` tokenizer metadata and provider usage to keep OMP aware of the conversation size.

Enable soft automatic compaction with threshold `196608`, reserve `65536`, and recent history `20000` tokens.
OMP selects compaction times, with during tool loops. The controller sends no manual compact command.
The service checks full templated input plus requested output against its total capacity.

The CPU guard repeats the production normalization and template path for independent capacity evidence.
Verify its count against SSE usage, SDK input plus cache usage, and the journal's provider-input snapshot.

OMP's settled `contextUsage` uses a provider-input anchor plus later local changes.
Its compaction decision also considers generated output and the stored-history estimate.
Keep the two records. Do not interpret the displayed percentage as the complete compaction decision.

Check the same session identity, usable summary, and successful model work after each observed compaction.
Natural compaction is not required during the hour. Report absent coverage explicitly.
See the pinned [OMP compaction procedure](https://github.com/can1357/oh-my-pi/blob/v18.4.3/docs/compaction.md).

The four lanes audit scheduling/KV, HTTP/lifetimes, GPU/caches, and tests/documents.
They read bounded source ranges, search measured symbols, and keep progress notes in their external workspace.
Verify source reads and forwarded tool results in each completed audit work item. Keep incremental session journals and complete RPC/wire records.

Verify completed server metrics, SSE terminals, and SDK usage for each model call.
Keep post-terminal `ConnectionResetError` or `BrokenPipeError` as diagnostics only with same-ID protocol, SDK, and server completion.
Check that SDK and SSE stop reasons agree. A `length` completion must match its requested output budget.

Parallel summary calls must have a unique usage and lifecycle match. Ambiguous evidence fails verification.

Check resource count and byte limits throughout collection. Check empty Rust/Python request state after the final drain.
Each logical pool then has `blocks - 1` free blocks, with cached blocks available for eviction.
Block `0` stays allocated. Keep the original HTTP and worker process birth identities.

Examine host RSS, GPU allocated/reserved memory, graph/compiler occupancy, active requests, and input lengths together.
Changing conversation sizes can change bounded caches. Independent review must examine the causes of growth and accept memory stability.
The collector records `operational_passed` and `awaiting_resource_review` separately from formal acceptance.

Keep failed worker diagnostics before GPU process lookup.
If GPU lookup fails, keep its error and a new worker observation.
A changed worker epoch or restart fails the normal workload.
Finish resource sampling before final checks and evidence hashes. A late sampling failure must fail the collection.

Record other GPU processes and device-used HBM. Light contention can continue if request and resource checks pass.
The fifteen phase workloads still must use an exclusive GPU.
Plot all completed task and measured summary calls by purpose. Do not add Chat/Responses color legends.

Use framework processing start on the horizontal axis and output tokens divided by processing wall time on the vertical axis.
Reasoning tokens enter the output count. Report queue time, cancellations, and failures separately.
There is no performance gate. `--diagnostic` permits a shorter smoke test, which cannot supply formal acceptance.



## Worker recovery with allocator OOM

Fault injection has different records from normal soak traffic. Daily commands also skip this target-model collector.
Run it explicitly on an idle B200:

```bash
scripts/with-gpu.sh scripts/with-env.sh python scripts/serving-recovery.py --mode mtp --output-dir "$EVIDENCE_DIR/recovery-mtp"
scripts/with-gpu.sh scripts/with-env.sh python scripts/serving-recovery.py --mode dspark --output-dir "$EVIDENCE_DIR/recovery-dspark"
```

The test creates an external worker launcher that uses `tests/fixtures/worker_faults.py`.
Production imports no fault injector. Each marker lets one explicit fault occur across worker replacement.
Run CUDA allocator OOM tests after sampling, accepted-state commit, proposal generation, and graph capture.

Keep HTTP alive, report stream failure, invalidate old prefixes, and verify successful Chat/Responses service after one worker replacement.
Make sure that a cold request after recovery cannot hit old KV identities, then verify new prefix reuse.
The collector owns and stops its service, worker groups, and temporary IPC endpoints.

## Tutorial and cleanup

Use the checks in [code-journey](../code-journey/README.md#verification).
Examine desktop/mobile screenshots and all thirteen routes after tutorial changes.

Run each GPU program through `scripts/with-gpu.sh`.
Stop all owned processes and descendants after completion or failure.
Make sure that GPU processes stopped, listeners closed, and IPC cleanup completed.
Keep only explicitly user-requested services active. Put their host details in `LOCAL.md`.
