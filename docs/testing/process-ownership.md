# Dedicated service ownership

Keep the service tree isolated from the collector's OMP, Docker, and CPU normalization processes.
The [service supervisor](../../benchmarks/engine_supervisor.py#L201) <!-- source: def supervise( --> starts one engine in a new Linux process.
Its initial child inventory must be empty. It starts no other subprocess roots.

The kernel sends abandoned descendants to this dedicated subreaper.
This scope proves their source even when an intermediate parent exits before an observation.
The supervisor uses pidfds and records PID, process birth time, and cleanup actions.
It does not use shared user identity as an ownership test.

The [critical tests](../../tests/tooling/test_engine_supervisor.py) cover lost session leaders, early root exit, controller failure, and root binding failure.
They also check that unrelated collector children stay active.

## Controller contract

Start the module with external logs and the full service command:

```bash
scripts/with-env.sh python -m benchmarks.engine_supervisor --log "$EVIDENCE_DIR/server.log" --audit "$EVIDENCE_DIR/processes.json" -- "${SERVICE_COMMAND[@]}"
```

Keep the supervisor's standard input open during collection. The engine has different standard input.
Send `stop` and a newline after collection. Wait for supervisor exit and examine its process audit.
Input EOF, unexpected service exit, `SIGINT`, or `SIGTERM` causes cleanup and a failed supervisor result.

Partial control input cannot stop process observations or signal handling.
The supervisor records collected observation intervals. The specified interval alone does not prove process coverage.

Finite collaborator tasks use `--natural-completion` instead of a controller stop.
Only a zero task exit with verified descendant cleanup can pass this mode.
In the default service mode, the controller must send `stop`.

Connect unexpected supervisor exit to the [collector cancellation event](../../benchmarks/agentic_soak.py#L129) <!-- source: def run(args, -->.
Wait for collector cleanup, supervisor exit, and the controller's monitor thread exit.
Do not close ownership handles while that thread still uses them.

## Result scope

Keep different records for collection, process cleanup, and resource review results.
The collector returns a dictionary. Examine `operational_passed`. Do not compare the dictionary with an exit code.

An operational result is not formal acceptance. Do not change the original collector's `passed=false`.
Keep independent resource review and formal acceptance in other records.

Keep primary errors and later cleanup, snapshot, or handle-close errors together.
A successful cleanup retry does not remove the original error.
Check release of owned processes, temporary listeners, containers, files, and images.
Keep user-authorized service exceptions in ignored local records.

These CPU tests do not prove one-hour inference stability.
Final-source MTP4 and DSpark collections are necessary.
