# Model cache policy

## Scope

The [Rust scheduler](../../crates/scheduler/src/lib.rs) uses a model-selected cache policy.
The [policy interface](../../crates/scheduler/src/policy.rs) supplies cache operations through associated Rust types.
Shared scheduling does not interpret cache group names, block tables, or physical slot layouts.
It controls request queues, token budgets, preemption, cancellation, and output validation.

The default policy is the existing Qwen `HybridCoordinator`.
The [Qwen implementation](../../crates/scheduler/src/policy/qwen.rs) keeps its block pools and allocation algorithms.
It supports a shared pool or one pool for each of FA and GDN.
This extraction does not replace those pools with the [state arena](state-arena.md).

The service continues to select Qwen only.
The Qwen worker messages keep their existing field names and values.
DeepSeek cache layouts, rank dispatch, and accepted-state integration are subsequent work.
This interface does not complete Phase 2 or enable DeepSeek execution.

## Model operations

The policy checks admission demand before the scheduler adds a request.
It supplies prefix hashes, prefix lookup, prefill alignment, and the prefill allocation unit.
It also supplies allocation, prefix-eviction checks, step assignments, and cache resource counts.
The scheduler does not calculate FA page demand or GDN slot demand.
Qwen admission and partial-checkpoint rules stay in its policy.

`PrefixHit` contains a token length and model-selected hit metadata.
A default hit has zero length and no new prefix state.
The policy can supply a fallback prefix after allocation fails.
For Qwen, this fallback uses an aligned prefix when a partial-prefix allocation fails.

`ScheduledRequest<State>` contains shared request fields and a model-selected state assignment.
The assignment gives the worker its state locations.
It does not own their storage.
The policy must keep all referenced storage available until release is permitted by the execution lifecycle.
Qwen assignments contain its existing FA, GDN, copy, and checkpoint metadata.
The Qwen client converts these assignments to the existing wire fields.

## Policy obligations

Policy callbacks are trusted implementation code, not worker input.
The token count after prefill alignment must not be more than the supplied count.
Prefix hits must keep a minimum of one accepted-history token for computation.
Allocation failure must not publish a new accepted prefix or keep an unusable request allocation.
A failed host load reservation must release its temporary pins and request allocation.

The scheduler validates the full worker output before it calls `commit_cache` for any request.
The policy uses the validated request progress to select accepted state.
State assignments and copied block values cannot authorize new storage ownership.
An arena-based policy must select storage through live owners, not through wire values.
The caller must prove device completion before it supplies results to the scheduler.
Distributed callers must also prove all-rank completion through the [rank gate](rank-control.md).

`free` releases request ownership only when release is permitted by its caller.
The existing worker lifecycle controls execution cancellation and worker replacement.
The host cache keeps pending transfers after request cancellation until transfer completion.
This extraction does not add a device completion check to `Scheduler::abort`.

## Host transfers

The [host cache](../../crates/scheduler/src/cache.rs) stores three different policy types.
`Snapshot` is cloneable host-snapshot metadata, not a live GPU lease.
`Transfer` holds transfer-specific ownership and has no `Clone` requirement.
`TransferPlan` gives wire metadata without storage ownership.
The scheduler does not interpret these model-selected fields.

The policy prepares request stores, prefix stores, and load reservations.
Qwen request stores keep the existing request ownership.
Qwen prefix stores pin the source tables.
Qwen loads pin the reserved destination tables before installation and next-step allocation.
These pins keep DMA destination storage available while DMA can use it.
They apply when allocation retires a previous GDN checkpoint.

After validated transfer completion, `finish_transfer` releases transfer-specific pins and returns snapshot metadata.
Deferred and capacity responses also pass through this release step.
Such responses must prove that no transfer continues to use the buffers.
Request ownership has a different lifetime from transfer ownership.
The scheduler's existing completion and cancellation branches control request release.
Canceled transfers cannot release their owners only because the request was canceled.

`CacheOperation<TransferPlan>` keeps common transfer identities and a flattened model plan.
Qwen plans keep the `fa_blocks` and `mamba_blocks` wire fields.
A drop operation has no transfer owner and uses the policy's empty plan.
Cloneable plans and snapshots do not replace the pending transfer owner.

## Boundaries

The Qwen service continues to use its existing cache diagnostics keys.
The shared schedule trace records the policy's resource counts.
The Python transfer helpers continue to validate the paired Qwen cache format.
Other model families must have their own wire conversion and transfer validation.

The [phase plan](plan.md) controls the subsequent typed-state and distributed-model work.
The [validation plan](validation.md) controls numerical and performance acceptance.
Generic Rust types do not prove a model's mathematics, capacity, or device safety.
