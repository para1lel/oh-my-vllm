# ADR-011: Service resource limits and worker recovery

Status: Active.

## Problem

Long-duration agent tasks reuse prefixes, create different graph shapes, and continue histories while clients can stop or read slowly.
Count limits alone cannot keep copied histories or completed stream buffers below byte limits.
A GPU failure can occur after some persistent state changes but before the worker returns a complete reply.

## Decision

Check and take request capacity before preparation and history copies.
Keep separate limits for request count, request state bytes, response storage, and stream buffers.
Keep the admission reservation until request output and stream drain release their references.
Use one absolute request deadline from admission through queue wait and generation.

Stage new prefix identities during allocation.
Publish them only after a validated result commits the matching state.
Keep referenced blocks safe during LRU eviction and remove failed, uncommitted identities.
Eviction reuses the fixed GPU arena. It does not return the arena's HBM storage to the allocator.

Keep graph and compiler storage at or below their capacity limits while shapes change.
Keep fullgraph compilation, manual CUDA Graphs, stream dependencies, and PDL safety contracts.
Release graph pools, compiled callables, model references, and request state during shutdown.

If worker/device state cannot be used safely after a failure, fail affected requests without automatic replay.
Keep the HTTP listener alive, close the old worker, clear logical caches, and load a new worker.
Return temporary unavailability during recovery. Keep recovery attempts and shutdown waits at their specified limits.
Use new worker state before new admission. Invalidate old logical KV identities before the replacement starts.

## Alternatives

Use the old worker after a partial state change only with proof that all state changes were restored.
The recovery path uses a new worker when that proof is unavailable.
Automatic request replay can duplicate stream output or function calls, so the client controls retries.
Regular worker replacement can hide resource leaks. Normal-load tests must have zero unexpected restarts.

## Verification

Use CPU tests for cache publication, unknown reply IDs, admission limits, deadlines, and stream drains.
Use target-model tests for OOM, partial state changes, shared prefixes, and successful service after recovery.
Run graph eviction and recapture tests with memory pressure and different shapes.

Run separate MTP4 and DSpark soak tests with four oh-my-pi agents.
Each mode runs for at least one hour. Request counts and agent task completion are diagnostic, without a minimum-count gate.

Measure request timing and memory. Apply no soak performance gate.
Measure the fifteen phase workloads again for inference-path changes.
Daily tests skip the heavy soak tests by default.

See [requirements](../requirements/service.md#req-serve-006-resource-limits-and-cache-integrity) and [tests](../testing.md).
