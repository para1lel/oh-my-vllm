# ADR-017: Disposable native-goal soak tests

Status: implementation in acceptance. Date: 2026-10-13.

## Problem

The previous four-session workload sent source-audit prompts against project copies.
A later fixture started an empty database project per session.
Use the accepted translation of one complete English book instead.
A goal can complete, pause, or stop automatic continuation before one hour.
These normal goal changes must not end the service test early.

## Decision

Pin OMP `18.10.3`, which supplies native RPC goals and automatic continuation for `rpc`.
Each of four persistent sessions translates the same validated English book in its own rootless Docker sandbox.
The corpus is staged from an external directory with its original manifest, and the volumes are read-only inputs.
Prepared follow-up goals continue the next passage, the next volume, and finally the verification pass.
After an early goal exit, submit the next goal in the same process, history, and workspace.

Only the host can create goals and start continuation again. The agent can read, complete, or drop its current goal.
Check each recorded goal identity against the host command. Clear old todo phases before each new goal.

Keep original transition reasons, timestamps, native journals, and full tool-output logs.
The collection window includes normal goal changes. Report the final paused tail separately.

The sandbox has separate PID and filesystem namespaces, no network interface, no GPU devices, and no inference-project or Docker-socket mount.
Only the agent's external directory and one owned socket directory enter its container.

One task-owned AF_UNIX bridge per lane carries model calls and another carries telemetry.
Guest loopback endpoints forward original request and response bodies to those mounted sockets.
Each upstream is a local plain HTTP collector origin, so a lane cannot fall back to a network path.
Bodies, connections, listeners and shutdown are bounded, and the sockets are removed after all exits.
The lane guard keeps the corpus and the specification read-only, rejects the pinned hashline `edit` tool, and leaves translation outputs and notes writable.

After collection, record corpus and specification hashes and translation progress without claiming translation quality or that the book is translated.
Remove agent containers, modified files, installed tools, caches, and the owned image.
Failure and cancellation use the same cleanup. Keep original logs and hash manifests in a separate evidence directory.
Never prune foreign Docker resources.

The hour checks inference service stability. Goal status, translation progress, tool errors, and final answers are workload observations.
They do not decide framework acceptance. Check original model requests, resource limits, and owned sandbox cleanup independently.
At the deadline, pause continuation and send native RPC `abort` before the final model-request drain.

## Verification scope

CPU fixtures check native continuation, early goal completion, the two APIs, automatic compaction, isolation, cleanup failures, corpus validation, and mounted-socket framing.
The transport fixture supplies a fake model server and a fake OTLP receiver. It proves SSE order and byte identity, protobuf export framing, and loopback-only listeners.
The Docker fixtures cross those hops from inside a sealed container, so the mounted-socket path and the read-only input guard are covered too.

The deadline fixture keeps a request in flight until the native goal deadline.
It binds the recorded pause and abort plus the aborted native result.
It also binds the unchanged request bytes, the downstream disconnect, and each mounted hop's own failed attempt through its direction and collection window.
The fixture runs the accepted collector path from the framework audit to the transport audit, with fixed request probe and metric data.
These fixtures use fixed data. They do not prove target inference, the book contents, or one-hour service stability.

Formal MTP4 and DSpark tests must have complete original records and independent resource review.
Start the one-hour soak only after all fifteen framework rows pass.
See [testing](../testing/soak.md#long-duration-agentic-service) for commands and evidence rules.
