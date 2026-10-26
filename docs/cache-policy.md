# Model cache policy

The [Rust scheduler](../crates/scheduler/src/lib.rs) uses an explicit model-selected [cache policy](../crates/scheduler/src/policy.rs).
Shared scheduling controls queues, token budgets, preemption, cancellation, and output validation.
It does not interpret model group names, block tables, or physical slot layouts.

## Model operations

The policy supplies admission demand, prefix lookup, valid prefill lengths, allocation units, reservations, assignments, and resource counts.
Valid prefill lengths reflect state correctness constraints.
Scheduling preferences must not exclude usable residual tokens.
The shared scheduler applies the priorities in the [runtime contract](requirements.md).

`PrefixHit` contains a token length and model-selected metadata.
A default hit has zero length and no new prefix state.
Prefix hits leave at least one accepted-history token for computation.
The policy can supply a fallback prefix after allocation fails.

`ScheduledRequest<State>` contains shared request fields and a model-selected assignment.
An assignment names state locations. It does not own storage.
The policy keeps referenced storage available until execution and transfers permit release.
Allocation failure must not publish a prefix or keep an unusable request allocation.

## Commit and ownership

The scheduler validates the full worker output before calling `commit_cache` for any request.
The policy selects accepted state from validated progress.
Copied assignments and block values cannot authorize storage ownership.
An arena-based policy selects storage through live owners in the [state arena](state-arena.md).

The caller proves device completion before supplying results to the scheduler.
Distributed callers also prove [all-rank completion](distributed.md#transactions-and-publication).
Policy callbacks are trusted implementation code, not worker input.
They must validate complete identity sets before transferring ownership.

`free` releases request ownership only when its caller permits release.
`Scheduler::abort` alone does not prove device completion.
Pending host transfers keep their owners after request cancellation until completion.

## Host transfers

The [host cache](../crates/scheduler/src/cache.rs) uses three policy types:

| Type | Meaning |
|---|---|
| `Snapshot` | Cloneable host metadata, not a live GPU lease |
| `Transfer` | Transfer ownership, without a `Clone` obligation |
| `TransferPlan` | Wire metadata, without storage ownership |

The policy prepares request stores, prefix stores, and load reservations.
A failed load reservation releases temporary pins and request allocation.
Transfer pins protect source and destination storage until asynchronous access completes.

After validated completion, `finish_transfer` releases transfer pins and returns snapshot metadata.
Deferred and capacity responses also use this release step.
They must prove that no transfer continues to use the buffers.
Request ownership has a separate lifetime from transfer ownership.

`CacheOperation<TransferPlan>` keeps common transfer identities and a flattened model plan.
A drop operation has no transfer owner and uses the policy's empty plan.
Cloneable plans and snapshots cannot replace a pending transfer owner.

## Implementations

The [Qwen policy](../crates/scheduler/src/policy/qwen.rs) uses `HybridCoordinator` and paired FA/GDN assignments.
Its [adapter cache guide](models/qwen38/cache.md) specifies physical restoration and wire fields.
The [transaction policy](../crates/scheduler/src/policy.rs) connects typed storage with rank execution.
A generic policy interface does not prove model mathematics, capacity, or device safety.
