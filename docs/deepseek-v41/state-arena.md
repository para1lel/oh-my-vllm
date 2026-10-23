# Typed state storage ownership

## Scope

The [state arena](../../crates/kv-cache/src/state.rs) supplies typed logical storage reservations for Phase 2.
It does not replace the Qwen coordinator at this time.
It does not allocate GPU buffers, select an accepted model prefix, or publish worker results.
The [distributed interface](distributed.md) supplies rank agreement and scheduler integration.
Model-specific physical state restoration is subsequent work in the [phase plan](plan.md).

## Groups and owners

`StateArena::new` validates the cache schema and parallel plan before it creates pools.
Each owned group has one pool per logical rank.
Slot zero is null. The arena does not allocate it.
Each pool must have at least one writable slot.

Capacities must be from 2 through `u32::MAX - 2` to leave space for queue sentinels.
The sum of all owned pool capacities must not be more than the caller's `max_slots` budget.

This budget counts logical slots, with null slots included.
It is not a GPU byte budget or a measured host-memory peak.
The arena validates the full pool plan before it allocates any pool storage.

Shared-source descriptions are not sufficient proof of physical aliasing.
The arena accepts a shared view only when its kind, block unit, format, and capacity agree with each producer.
The shared view uses its root producer's pool for the same rank.
It cannot receive an independent write allocation.
The general descriptor validator keeps its broader structural producer rules.

## Reservations

An allocation request names an owned group, rank, and positive slot count.
Unknown groups, unknown ranks, shared write targets, duplicate targets, and zero counts cause errors.
The arena verifies all capacities and read leases before it allocates or pins storage.
A capacity error cannot leave a partial reservation in another group or rank.
New writes do not use slots held by accepted readers.

Each transaction has an epoch, step ID, transaction ID, and request ID.
The epoch must be positive.
Step IDs must be positive and cannot decrease in an arena.
Multiple transactions can use the same step ID.
Transaction IDs increase only after a reservation succeeds.
Identity counters do not wrap.

Each pool also supplies increasing allocation generations.
Reusing a physical slot produces a different generation.
`StateBlock` values identify group, rank, slot, and generation.
Copies of these values do not own storage and cannot authorize pinning or release.

## Storage selection

`StorageLease::select` creates a lease for blocks held by its source lease.
`StateTransaction::select_storage` can select from the transaction's read pins and write slots together.
These methods do not release or change the source owners.
The result owns one more reference for each selected block.

The selected list must contain unique values from the authorized source blocks.
Each group, rank, slot, and generation must agree with a source block.
A shared-view group name cannot replace the root producer name in a block identity.
The result keeps the supplied sequence.
An empty selection is valid and holds no slots.

The methods validate all selected identities and reference counts before they change any count.
Reference-count validation and pinning use one arena lock.
An invalid selection keeps its source owners unchanged and does not pin a partial selection.

These operations supply storage ownership.
They do not prove model acceptance or device completion.
The caller must keep an executing transaction alive until all asynchronous read and write operations are completed.
The caller must validate model results before accepted-state publication.

## Lease lifetime

`StorageLease` has private ownership fields and no `Clone` or deserialization implementation.
Its `pin()` method creates one more read lease after reference-count validation.
Reference-count overflow causes an error before any count changes.
The arena rejects read leases from a different arena, even when the epochs are equal.

A transaction pins each read slot once.
Repeated input leases do not create duplicate transaction pins for the same slot.
Those pins keep accepted inputs available after the original request releases its lease.
New write slots have their own lease.

Dropping or aborting a transaction releases its write slots and read pins.
`into_storage()` releases the transaction's read pins and returns ownership of its write slots as a storage lease.
This operation does not declare that model state is accepted.
The caller must validate model results and all necessary rank completions before logical commit or prefix publication.

The caller must also keep the transaction alive until asynchronous readers and transfers finish.
Cancellation alone does not prove that condition.
The arena cannot detect device completion from CPU ownership values.
A storage lease keeps its arena alive after the caller drops the public arena handle.
The last lease release makes its slots available for reuse.
A poisoned arena rejects further allocation and does not recycle possibly inconsistent slots.

## Checks

The small Rust checks exercise cross-group allocation failure, compatible shared views, read pin lifetime, cancellation, stale steps, and slot generations.
They also check mixed read/write selection, lease lifetime, and invalid selections without partial pinning.
Existing Qwen paired-cache tests stay unchanged.
These checks do not prove distributed execution, GPU completion, prefix restoration, or service capacity.

```bash
scripts/with-env.sh cargo test -p oh-my-vllm-kv-cache
```
