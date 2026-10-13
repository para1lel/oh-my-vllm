# Long-duration service checks

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
scripts/with-env.sh python -m pytest tests/tooling/test_agentic_sdk_trace.py tests/integration/test_agentic_sdk_integration.py tests/integration/test_agentic_goal_integration.py tests/tooling/test_agentic_goals.py tests/unit/test_host_allocator.py
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
