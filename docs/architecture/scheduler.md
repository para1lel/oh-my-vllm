# Scheduling and cache concepts

## Scheduler and allocation

Continuous batching combines ready token work from active requests.
Decode work keeps forward progress while aligned chunks share the available prefill budget.
A checkpoint boundary makes future reuse possible. Arbitrary partial state cannot substitute for that checkpoint.

The [scheduler](../../crates/scheduler/src/lib.rs) owns queue policy and token budgets.
[Requests](../../crates/scheduler/src/request.rs) distinguish accepted history from unverified drafts.
[Allocation](../../crates/kv-cache/src/coordinator.rs) coordinates groups before execution.

Capacity failure can preempt a later request for recompute. Accepted history stays. Unverified drafts do not.
A worker reply must pass validation before history, computed progress, or reusable identities change.
Read the [commit boundary](../../crates/scheduler/src/lib.rs#L619) <!-- source: pub fn update(&mut self, mut worker: -->
for the implementation.

## Cache groups and prefix reuse

FA pages store attention history. GDN checkpoints store recurrent history.
Their reused prefix must agree. A longer FA prefix alone cannot restore missing recurrent state.
References protect active slots. LRU eviction can reclaim unused cached slots.

[Prefix hashes](../../crates/kv-cache/src/hash.rs) chain token content to its preceding prefix.
[Group state](../../crates/kv-cache/src/group.rs) separates staged identities from completed cache data.
[Block ownership](../../crates/kv-cache/src/pool.rs) controls references and eviction.

Only completed accepted computation can publish cache identities.
Cancellation removes uncommitted identities, while valid committed prefixes can stay reusable.
Logical slot reuse and GPU graph storage have different owners and budgets.

The [model/context requirements](../requirements/core.md) define fixed block size, layers, capacity, and state precision.
