# Long-duration service checks

## Long-duration agentic service

Daily commands skip this heavy test. Run each mode against a dedicated service with maximum model length `262144`.
Use a new external directory. Keep the source and release binary unchanged throughout collection.
Build the service and CPU request normalizer together:

```bash
scripts/with-env.sh cargo build --release --bins
scripts/with-env.sh python scripts/agentic-soak.py --mode mtp --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/server.log" --corpus "$CORPUS_DIR" --output-dir "$EVIDENCE_DIR/mtp-soak"
scripts/with-env.sh python scripts/agentic-soak.py --mode dspark --base-url "$SERVICE_URL" --server-log "$EVIDENCE_DIR/dspark-server.log" --corpus "$CORPUS_DIR" --output-dir "$EVIDENCE_DIR/dspark-soak"
```

Set `OH_MY_VLLM_MODEL` or supply `--model`. Install OMP on `PATH` or supply `--omp`.
Set `CORPUS_DIR` to the external directory that holds the validated book and its original manifest.
Use the matching service for each command. Stop that service and worker after collection.
Use [dedicated process ownership](process-ownership.md) for an isolated service process scope.

Use the pinned OMP `18.10.3` executable and Docker with its `rootless` mode enabled.
The collector builds a temporary image with OMP, Python, Bash, certificates, and the lane forwarder.

Keep host-specific selection in ignored `LOCAL.md`.
The collector owns four sandbox containers, persistent OMP processes, and temporary recording proxies.

Each lane runs with `--network none`, no GPU devices, and no inference source, weight or Docker-socket mount.
Only the disposable lane directory and one owned socket directory enter its container.
The containers have separate PID namespaces.

The collector owns one AF_UNIX bridge per lane for model calls and one for telemetry.
Guest loopback endpoints forward the original request and response bodies to those sockets.
Each bridge upstream must be a local plain HTTP collector origin, so no network fallback exists.
Bodies, connections, listeners and shutdown are bounded, and the sockets are removed after all exits.

The model socket carries generation routes only. Telemetry has its own socket.
A fake-upstream check proves model SSE and protobuf export framing without an external interface.

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

Each lane translates the same complete English book in a new empty workspace.
`--corpus` names the external directory that holds the six volumes and their original manifest.
The manifest pins the book totals and each volume's identifier, size and full SHA256 digest.
Validation checks the pinned manifest digest and each staged input before a lane starts.

```json
{
  "title": "<book>",
  "author": "<author>",
  "source": "Project Gutenberg, <volumes>",
  "derivation": "<normalization note>",
  "book_file": "book.txt",
  "book_sha256": "<64 hex>",
  "book_bytes": 1,
  "book_english_words": 1,
  "downloaded_unix_s": 1.0,
  "volumes": [{"volume": 1, "gutenberg_id": 731, "sha256": "<64 hex>", "bytes": 1}]
}
```

The staged copy is read-only. The lane tool guard allows read, grep, write and goal inspection, and it rejects edit.
Writes to the corpus and the specification are rejected too, because pinned OMP edit calls carry `{i, input}` hashline sections with REM and MV operations.
Translation outputs and the progress notes stay writable.

The controller uses native RPC `goal create`, with automatic continuation enabled for `rpc`.
No token budget applies. Goal completion, pause, or stopped continuation selects the next prepared goal.

The next goal resumes from `notes/progress.md`. No one-hour translation-completion requirement applies.

The same process, session ID, history, and workspace continue. Goal changes keep their original reasons and timestamps.
A goal command ID supplies host correlation. It does not represent a native `prompt_result`.

Record source reads, tool errors, translation progress, and goal changes as workload observations. They do not decide framework acceptance.
Keep incremental journals, native session and raw tool-output logs, and complete RPC/wire records.

After the hour, pause continuation and abort active agent work. Report the tail separately from the collection window.
Use the [deadline-tail evidence rules](deadline-tail.md) for a client abort during server completion.
Hash the staged corpus and the specification before and after collection, and fail on a changed input.
Record translation progress without claiming translation quality or completion.

Then remove owned containers, modified files, installed tools, caches, and the temporary image.
Keep original logs, hash manifests, summaries, and resource figures in a separate evidence directory. Failure and cancellation use the same cleanup.
The `--workload goal` or `--workload audit` diagnostic fixtures cannot supply formal acceptance.

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
scripts/with-env.sh python -m pytest tests/tooling/test_agentic_sdk_trace.py tests/integration/test_agentic_sdk_integration.py tests/integration/test_agentic_goal_integration.py tests/integration/test_agentic_translation_integration.py tests/integration/test_agentic_deadline_cancellation.py tests/tooling/test_agentic_goals.py tests/tooling/test_agentic_lane_transport.py tests/tooling/test_agentic_corpus.py tests/tooling/test_agentic_translation.py tests/tooling/test_agentic_soak_inputs.py tests/unit/test_host_allocator.py
```

The integration fixture supplies API responses from fixed test data. It checks full request JSON equality and all three automatic-compaction purposes.
It does not test target inference or tokenizer counts from the target model.

The sandbox fixtures carry the `omp` marker and use fixed CPU API data only.
Run them with rootless Docker and the pinned OMP executable present.
They prove the sealed container has no network interface, remove their containers, and remove their image.

The goal lane keeps the diagnostic database fixture.
The translation lane stages the accepted corpus and exercises the mounted sockets inside the container.
It reads a staged volume, and the guard rejects a corpus write plus two hashline edit payloads.
One translation output lands, and the original SDK exports stay complete.

The deadline fixture streams an unterminated response, so its request is in flight at the deadline.
It binds the recorded pause and abort, the aborted native result, and the unchanged bytes.
It binds each hop's own failed attempt through the collector path from framework audit to transport audit.
Chat and Responses run separately, and a cancellation carries the same proof.

The transport check drives a fake model server and a fake OTLP receiver through the two socket hops.
It proves SSE delivery order and byte identity, protobuf export framing, body bounds, and loopback-only listeners.
The Docker lanes above cross the same hops from a sealed container, so the mounted-socket path is covered.
Neither check proves target inference or the target model endpoint.

The hop audit demands a closure record for each configured hop.
Journal count, journal error, released handlers, removed sockets and guest endpoint release must agree.
Each digest attempt covers exactly one original or export, so equal bodies stay separate.
A deadline bound accepts one failed attempt of that direction and error between the recorded abort action and the client disconnect.
A digest alone binds nothing, and an unrelayed request fails.
Model-only and zero-request lanes stay valid.

The corpus and workload checks use synthetic books. They prove manifest and digest rejection, staging, objective resumption, and read-only guards.
They do not prove the book contents or translation quality. Independent review must bind the external corpus manifest.

Start the one-hour soak only after all fifteen framework rows pass. Translation completion and quality stay unproven.

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

Use a dark background and a fixed 0–1 gradient for `prefix_hit_tokens/input_tokens`.
Distinguish task and compaction calls only with different markers.

Use framework processing start on the horizontal axis and output tokens divided by processing wall time on the vertical axis.
Reasoning tokens enter the output count. Report queue time, cancellations, and failures separately.
There is no performance gate. `--diagnostic` permits a shorter smoke test, which cannot supply formal acceptance.
