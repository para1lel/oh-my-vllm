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

The FlashInfer artifact tests use the pinned cubin package without CUDA initialization.
They load checked B200 manifests, headers, and a binary kernel with downloads disabled.
They also reject incorrect artifact bytes and package version mismatches.

```bash
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/test_flashinfer_artifacts.py -q
```

These CPU checks do not prove model timing or cold-start acceptance.
Keep new project caches for GPU acceptance. Do not install a prewarmed project or FlashInfer JIT cache.

CPU executable diagnostics tests check executable lifetime, owner intersections, pending callbacks, and explicit incomplete coverage.
They keep diagnostic snapshots without a strong executable reference.
The HTTP regression sends nonempty owner measurements through Python msgpack, the Rust decoder, and HTTP diagnostics.
It checks tuple lengths `0/1/2`, owner intersections, and zero worker restarts.

```bash
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/test_executable_diagnostics.py tests/test_worker_lifecycle.py -q
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/test_serving_http.py::HttpTests::test_executable_diagnostics_cross_msgpack_rust_and_http_boundaries -q
```

Q/K GPU tests include FP64 output, FP32 edge values, and changed-input graph replay at the short dispatch boundary.
Keep their original numerical tolerances and the frozen TileLang comparison.

Each diagnostic source-audit task must have a new successful source read.
Grep output and earlier history do not satisfy the existing per-item source-read gate.

The CPU SDK fixture keeps full request bodies and the number of copies of each body.
With completed history, two automatic summary requests can have different sequences.
Other requests must keep the same sequence. The CPU fixture tests each sequence.

Scripted HTTP fixtures have a `30`-second timeout for initial service readiness.
This wait includes Python imports and the ZMQ connection. Request, recovery, and shutdown assertions keep their existing deadlines.

Selected pytest arguments follow `cpu` or `full`.
Full mode uses two new pytest processes, with the context-boundary module first.
The boundary collector's GPU ownership check runs without CUDA contexts from earlier tests.
The partition plugin keeps selected node IDs and their duplicate counts. Empty groups do not hide failures.

Each process uses the original collection scope and pytest filters before path partition.
This keeps `--ignore` and `--ignore-glob` effective for the boundary module.

Use `full --evidence-dir "$EVIDENCE_DIR/python-tests"` for a new external evidence directory.
Without this option, the launcher prints a new temporary directory and keeps its files after exit.
Each started pytest process has a log and process birth records. Keep JUnit results when pytest writes them.
Do not treat missing group reports as completed test evidence.
`tests/test_python_test_partition.py` compares collection results with the original pytest node multiset without GPU execution.

The existing worker lifecycle test checks model, callable, and graph-pool reference release on CPU.
The six target-model boundary rows check normal process exit before the unchanged worker shutdown deadline.
Removal of the final `torch.cuda.empty_cache()` call uses these behavior checks, without a test that repeats the source statement.

Tests include scheduler transactions, cache ownership, cancellation, RPC framing, serving, inference-path FP64 references, and compiled model units.
GPU coverage includes all six maximum-context cases and changed-metadata graph replay.

The benchmark ownership check queries only the selected GPU UUID through `nvidia-smi -i`.
Its query timeout stays `10` seconds. Query failures stop the collection.
Process acceptance still checks exact owned ancestry and birth identity.
CPU regressions check selected-device isolation, foreign processes, birth reuse, and query failures.

Linux can return `EINVAL` when a process is reaped after kernel PID lookup and before pidfd binding.
The [Linux 6.8 implementation](https://github.com/torvalds/linux/blob/v6.8/kernel/fork.c#L2178) checks for a thread-group leader after lookup.

Collection checks the process birth again after this error.
The observer does not register a missing or changed birth. The same birth still causes failure.
Other `pidfd_open` errors keep their previous handling.
CPU tests check child exit, PID reuse, and these failure conditions.

`tests/test_graph_capture.py` checks effective allocator settings and workspace lifetime fences on CPU.
`tests/test_graph_memory.py` checks allocator release with live shared graphs and outputs.
Its temporary allocation is more than `total_size - allocated_size` for each existing default-pool segment.
The allocator can release all storage in the new segment while other segments keep live tensors.
A new case keeps a live tensor and at least `128` MiB of free storage in the same default-pool segment.
The owner, pool, zero-eviction, `32` MiB release, and exact replay checks stay active.

`tests/test_dspark_context_graph_gpu.py` includes the production 29-by-25600 BF16 context projection.
Its warmup uses the capture stream before pool capture.
It resets unrelated GEMM graphs, applies allocator pressure, and replays changed inputs.
It checks all five cache outputs at `atol=0.03`, `rtol=0.03`, exact untouched-row canaries, and unchanged workspace budgets.

Full-model drift tests use their documented model-level bounds in addition to strict operator-level tolerances.
Use scripted workers for protocol regressions and the target checkpoint for GPU/agentic acceptance.
Record warnings and skipped tests with their reasons.

## Shared service sampling acceptance

Repetition tests compare bounded cycle detection with an independent definition and check speculative prefix distributions.
CPU cases cover joint cycles, forced grammar continuations, prompt isolation, explicit disable, and invalid logits.
GPU cases check filtering order, forced grammar continuations, and speculative prefix distributions.
HTTP regressions check shared defaults, null values, and explicit booleans through Rust, msgpack, and Python.

Tokenizer regressions check canonical reasoning closure, tool opener constraints, schemas, function choice, parallel calls, and speculative rollback.
They keep literal markup, plain whitespace, disabled tools, explicit non-strict arguments, stop strings, and total length.
These CPU cases do not prove that model repetition is resolved or that long-duration agent acceptance passed.

```bash
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/test_serving_worker.py tests/test_thinking_budget.py -q
```

Test HTTP services use production defaults.
The two APIs and each draft mode use the same validation and precedence.
The default presence penalty is `0.5`. The separate reasoning budget default is `-1`.

Persistent acceptance must use these defaults. Record service defaults and request sampling fields after service defaults are added from the original diagnostics and metrics.
Maximum effort and output/context limits stay explicit workload settings.

Offline phase/operator measurements and budget boundary tests keep their specified explicit parameters.

`tests/test_serving_http.py` checks Rust HTTP, ZMQ, Python sampler, and checkpoint merge without GPU inference.
It checks explicit zero, null omission, service changes, and rejection before worker launch.
Rust tests use the same validation function for CLI and request fields.

## Native tool description compatibility

Use the original pinned OMP Bash tool definition through the two APIs.
Its `name` parameter has `maxLength=48`. Keep this description in the model prompt.
The request adapter must accept the description without adding a string-length mask.
OMP must reject a generated name with `49` characters, return that error in subsequent model input, and complete the next valid command.
Explicit `strict=true` must keep unsupported-schema rejection.

The CPU fixture uses deterministic API responses and a sandboxed OMP executable process.
It does not prove target-model output or one-hour stability.
With `OH_MY_VLLM_MODEL` and the release normalizer available, it also uses the production CPU preflight.

```bash
scripts/with-env.sh python -m pytest tests/test_serving_worker.py tests/test_agentic_goal_integration.py -q
```

## Reasoning budget acceptance

Build the release binary and start each service with `serve --thinking-token-budget 32`.
Use modes `none`, `mtp`, and `dspark`, with an idle UUID-pinned B200 and the corresponding startup mode.
The script checks zero, small, default, unlimited, and off budgets through the two APIs.
It checks streams, JSON, strict tools, total length, effective request metrics, and unchanged source identities.
Stop the owned service and worker after each run. Keep original request and response records.

```bash
scripts/with-env.sh python scripts/thinking-budget-acceptance.py --mode mtp --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --output "$EVIDENCE_DIR/thinking-budget.json"
```

## Accuracy

Use independent references with CPU FP64 computation on rounded inference inputs.
Tests must include prefill, decode, boundary writes, strided views, grouped MTP verification, and speculative rollback.
BF16 operator output uses `atol=rtol=0.03`.
Recurrent state must have NRMSE at most 1% and maximum absolute error at most 2% of reference peak.
Examine each written slot and unchanged protected state.

For fused chains, include all rounding, scale, copy, and reduction behavior.
See `tests/probe_worker.py` and the applicable kernel tests.

`tests/test_ir_normalized_linear.py` checks small-row fusion at rows `1/2/4/6/12/24/32`.
Graph replay reads changed inputs and weight scales, keeps compact output shapes, and leaves input bytes unchanged.
Storage-offset cases check each of the three normalization inputs at rows `2/17`.
The previous exact checks and FP8 operator tolerances stay active.

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

Compare the current runtime with the reviewed semantic contract on CPU before GPU collection:

```bash
scripts/with-env.sh env CUDA_VISIBLE_DEVICES= python -m pytest tests/test_performance_metadata.py::test_current_runtime_contract_matches_semantic_review -q
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
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_KERNEL_BACKEND=cuda python benchmarks/kernels.py --output "$EVIDENCE_DIR/operators.json"
```

Set `EVIDENCE_DIR` to a writable evidence directory in an external directory.
The harness uses static cases from `development/kernels/cases.py`.

Use `--operations NAME` to select an operation family.
Give `--case-id ID` for each selected case ID.
A selection with less than 517 case IDs reports `coverage.complete=false` and keeps the full per-case protocol.
All 517 case IDs must pass before matrix acceptance.

The original 317 cases keep their configurations and comparison pins.
The supplements add 200 cases from `development/kernels/long_prefill_cases.py` and `development/kernels/short_prefill_cases.py`.
Partial collections must keep their source scope and cannot supply full-matrix acceptance alone.

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
Use OMP with the target checkpoint through each API as specified in [service contracts](serving.md#verification).
Check original model streams, usage, and server metrics independently of task results.
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
It keeps client JSONL, forwarded requests, original SSE bytes, response IDs, and tool-result hashes.

Keep the original source-read and answer checks as a separate workload audit.
Task errors and incomplete answers do not decide framework acceptance. Zero model requests cannot pass.
Stop the DSpark service and its worker after these tests.

Service scripts reserve each output file or directory before requests. An existing result path fails.
They keep `passed=false` until the applicable contracts pass.
Each received HTTP response is saved before content or log checks.
Failures keep received responses and an explicit error record.
Lifecycle streams keep the consumed SSE frames before disconnect.

Native first-event and idle deadlines are 900 seconds. The collector turn deadline is 7200 seconds.
Oneoff clients use a 30-minute task limit and a 1860-second host limit. Their proxy read deadline is 900 seconds.

The recovery request and stage deadlines are 900 and 3600 seconds.
The user waived more tests for timeout parameters alone. Use meaningful regression checks for collector behavior changes.

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

Use the pinned OMP `18.10.3` executable and Docker with its `rootless` mode enabled.
The collector builds a temporary image with OMP, Python, Bash, and certificates.
Set `--sandbox-host` or `OH_MY_VLLM_SANDBOX_HOST` to the container gateway for host loopback access.

Keep host-specific selection in ignored `LOCAL.md`.
The collector owns four sandbox containers, persistent OMP processes, and temporary recording proxies.

Only each agent's new external directory enters its container. Inference source and the Docker socket do not enter it.
The containers have separate PID namespaces and no GPU devices.

Each process keeps its session ID and history through successive native goals.
Two sessions use Chat and two use Responses. A shared four-slot limit also includes summary calls.

Each inference mode must run for at least `3600` seconds with four persistent request workloads. There is no request-count minimum.
Initialize the clients after service readiness, then start the clock before their first requests.
First compilation, graph capture, and later new-shape costs enter the measurements.

For this collector, start the service with `--worker-step-timeout-seconds 600` and `--request-timeout-seconds 3600`.
Record these limits from diagnostics. They let cold compilation complete without separate warmup.

Do not insert intermediate drains or fault probes. After the hour, pause continuation and send native RPC `abort`.
Drain model requests and record the stop action.

Use OMP `max` for tasks and summaries, mapped to service `xhigh`.
Use production sampling and reasoning-budget defaults for this workload, without different startup overrides.
Record effective service values from diagnostics and sampling fields after service defaults are added from request metrics.
OMP does not supply these fields in the two OpenAI APIs. Service defaults also apply to its summary requests.

The model context window is `262144` and its task output limit is `32768`, with reasoning tokens.
Use the local `qwen3` tokenizer metadata and provider usage to keep OMP aware of the conversation size.

Enable soft automatic compaction with threshold `196608`, reserve `65536`, and recent history `20000` tokens.
OMP selects compaction times, with during tool loops. The controller sends no manual compact command.
The service checks full templated input plus requested output against its total capacity.

The CPU guard repeats the production normalization and template path for independent capacity evidence.
Verify its count against SSE usage. Record SDK and journal comparisons in the separate workload audit.

OMP's settled `contextUsage` uses a provider-input anchor plus later local changes.
Its compaction decision also considers generated output and the stored-history estimate.
Keep the two records. Do not interpret the displayed percentage as the complete compaction decision.

Check the same session identity and report subsequent requests after each observed compaction. Summary content is a workload observation.
Natural compaction is not required during the hour. Report absent coverage explicitly.
See the pinned [OMP compaction procedure](https://github.com/can1357/oh-my-pi/blob/v18.10.3/docs/compaction.md).

Each lane receives an embedded database task in an empty workspace with Python standard-library tools.
The task includes SQL, transactions, indexes, durable storage, crash recovery, independent tests, and measured benchmarks.
The controller uses native RPC `goal create`, with automatic continuation enabled for `rpc`.
No token budget applies. Goal completion, pause, or stopped continuation selects the next prepared goal.
Follow-up goals examine faults, concurrency, SQL results, storage changes, performance, and open requirements.

The same process, session ID, history, and workspace continue. Goal changes keep their original reasons and timestamps.
A goal command ID supplies host correlation. It does not represent a native `prompt_result`.

Record source reads, tool errors, task progress, and goal changes as workload observations. They do not decide framework acceptance.
Keep incremental journals, native session and raw tool-output logs, and complete RPC/wire records.

After the hour, pause continuation and abort active agent work. Report the tail separately from the collection window.
Hash generated source before and after independent sandbox test commands. Record failures without claiming that the database task is completed.

Then remove owned containers, modified files, installed tools, caches, and the temporary image.
Keep original logs, hash manifests, summaries, and resource figures in a separate evidence directory. Failure and cancellation use the same cleanup.
The optional `--workload audit --diagnostic` path keeps the older source-audit fixture. It cannot supply formal acceptance.

Verify original request coverage, completed SSE terminals, usage, and server metrics independently of native task outcomes.
Report cancelled requests with their original protocol and transport observations. Do not infer their cause from a shared error message.

The following SDK and journal checks supply separate native workload diagnostics. Their result does not decide framework acceptance.
Keep post-terminal `ConnectionResetError` or `BrokenPipeError` as diagnostics only with same-ID protocol, SDK, and server completion.
Check that SDK and SSE stop reasons agree. A `length` completion must match its requested output budget.

Parallel summary calls must have a unique usage and lifecycle match. Ambiguous evidence fails verification.

OMP `18.10.3` does not put internal summary usage in its session journal.
The collector uses one owned loopback OTLP endpoint per client for original SDK result traces.
It keeps provider dispatch, model APIs, request bodies, and automatic compaction unchanged.
Native traces supply response IDs, conversation IDs, token usage, finish reasons, and summary purposes.

The API binding comes from hashed isolated configuration and recorded runtime model state. Native traces do not supply that field.

Each wire request must match a unique SDK result trace. Missing, changed, or ambiguous records fail verification.
After active work, the collector waits for completed SDK exports before RPC shutdown and then closes the trace listener.
Original protobuf records have size limits and hashes. Collector errors and closure records enter offline verification.

An exporter can send the same batch again. The collector keeps these original records.
Two SDK completions with the same response ID fail verification.
Only `message_end` supplies an RPC SDK completion. A normal `turn_end` can include the same message.

Run the CPU instrumentation checks with the pinned OMP executable available:

```bash
scripts/with-env.sh python -m pytest tests/test_agentic_sdk_trace.py tests/test_agentic_sdk_integration.py tests/test_agentic_goal_integration.py tests/test_agentic_goals.py tests/test_host_allocator.py
```

The integration fixture supplies API responses from fixed test data. It checks full request JSON equality and all three automatic-compaction purposes.
It does not test target inference or tokenizer counts from the target model.

Check resource count and byte limits throughout collection. Check empty Rust/Python request state after the final drain.
Each logical pool then has `blocks - 1` free blocks, with cached blocks available for eviction.
Block `0` stays allocated. Keep the original HTTP and worker process birth identities.

Examine host RSS, GPU allocated/reserved memory, graph/compiler occupancy, active requests, and input lengths together.
Changing conversation sizes can change bounded caches. Independent review must examine growth, resource limits, and cleanup.
Sampled limits do not prove a memory plateau, full allocation attribution, or stability beyond the measured hour.

Set `OH_MY_VLLM_HOST_ALLOCATOR_DIAGNOSTICS=1` for optional glibc allocator fields at existing diagnostic intervals.
Each record includes the `available` value and call time. Available measurements include glibc version and raw `mallinfo2` fields.
The call uses allocator locks and examines free storage. Its cost can vary with allocator state.

`uordblks + hblkhd` gives values for storage in use or kept by the allocator. `fordblks` gives its free storage.
Thread caches, direct mappings, and other allocators limit these measurements. They do not count all objects in use or replace RSS.
The option does not change allocator settings or start extra resource release. Unsupported hosts report that measurements are not available.

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
