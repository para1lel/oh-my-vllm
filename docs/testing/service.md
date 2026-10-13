# Context and service checks

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
Use OMP with the target checkpoint through each API as specified in [service contracts](../serving/resources.md#verification).
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
