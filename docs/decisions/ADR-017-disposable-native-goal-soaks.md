# ADR-017: Disposable native-goal soak tests

Status: implementation in acceptance. Date: 2026-10-13.

## Problem

The previous four-session workload sent source-audit prompts against project copies.
Use a new project, native goals, and complete removal of agent changes and installed tools.
A goal can complete, pause, or stop automatic continuation before one hour.
These normal goal changes must not end the service test early.

## Decision

Pin OMP `18.10.3`, which supplies native RPC goals and automatic continuation for `rpc`.
Each of four persistent sessions starts an empty database project in its own rootless Docker sandbox.
The task includes code, independent tests, crash checks, and measured benchmarks.
Prepared follow-up goals examine correctness, concurrency, storage, performance, and open requirements.
After an early goal exit, submit the next goal in the same process, history, and workspace.

Only the host can create goals and start continuation again. The agent can read, complete, or drop its current goal.
Check each recorded goal identity against the host command. Clear old todo phases before each new goal.

Keep original transition reasons, timestamps, native journals, and full tool-output logs.
The collection window includes normal goal changes. Report the final paused tail separately.

The sandbox has separate PID and filesystem namespaces, no GPU devices, and no inference-project or Docker-socket mount.
Only the agent's external directory enters its container.
The host Docker client uses its existing configuration. The agent receives only explicit test configuration.
Temporary proxies and SDK trace endpoints stay in the host collector.

After collection, record source hashes and independent test results without claiming that the database task is complete.
Remove agent containers, modified files, installed tools, caches, and the owned image.
Failure and cancellation use the same cleanup. Keep original logs and hash manifests in a separate evidence directory.
Never prune foreign Docker resources.

The hour checks inference service stability. Goal status, task progress, database tests, tool errors, and final answers are workload observations.
They do not decide framework acceptance. Check original model requests, resource limits, and owned sandbox cleanup independently.
At the deadline, pause continuation and send native RPC `abort` before the final model-request drain.

## Verification scope

CPU fixtures check native continuation, early goal completion, the two APIs, automatic compaction, isolation, and cleanup failures.
They supply fixed model responses. They do not prove target inference or one-hour service stability.
Formal MTP4 and DSpark tests must have complete original records and independent resource review.
See [testing](../testing.md#long-duration-agentic-service) for commands and evidence rules.
