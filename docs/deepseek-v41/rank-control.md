# Rank group validation

## Scope

The [group gate](../../crates/zmq-worker/src/group.rs) validates rank readiness and completion metadata for Phase 2.
Its [messages](../../crates/zmq-worker/src/group/protocol.rs) use group protocol version 1.
The existing Qwen worker protocol and client do not use this gate at this time.
This module does not start processes, execute collectives, validate model outputs, or publish accepted prefixes.
Those integration steps in the [phase plan](plan.md) are still necessary.

## Readiness

The caller supplies an epoch, expected model descriptor, ordered GPU UUIDs, capacity limits, and a startup deadline.
The epoch must be positive.
GPU UUIDs must be different and use the canonical lowercase format.
Their list position identifies the logical rank, not the physical GPU index.

Each reply binds its protocol version, epoch, logical rank, device UUID, and model descriptor.
The caller supplies the source rank from the connection, not from the reply.
The gate compares these two rank identities.

All descriptor fields must agree with the expected descriptor, except allocated cache capacities.
Each allocated capacity must be at least the requested minimum.
Each pool must leave space for its null slot and queue sentinels.
Shared views must have the same kind, block unit, format, and capacity as their producers.

The full capacity budget counts owned logical slots across ranks, with null slots included.
It does not count shared views twice.
It is not a GPU byte budget.
No capacity result is available before all ranks supply valid readiness.
The result keeps per-rank capacities and the minimum common capacity for each group.
DEP scheduling must also keep each request's owner rank.

## Group steps

A step supplies an increasing step ID and unique request and transaction IDs.
Transaction IDs must be greater than all transaction IDs from previous steps.
The plan also supplies request owners, row counts, prefill flags, a collective-plan SHA256, and a collective count.
Configured request and row limits apply before dispatch.
An invalid local plan does not start a step and does not fail a ready group.

TP and TEP send the same request work to each rank.
Rank zero owns their request outputs.
DEP sends request work only to the request's owner rank.

Each rank receives a schedule.
This rule also applies to ranks with zero rows.
If any request has prefill, the whole group uses Eager execution.
If no request has prefill, the whole group uses CUDA Graph execution.

Each result must name the active epoch, step, rank, and collective plan.
Its transaction set must be the same as the scheduled local transaction set.
It must report the full collective count and completed device work.
Zero-row ranks must also report completion.

The gate returns one `CompletedStep` only after all ranks report valid completion.
This value has private fields and no cloning or deserialization implementation.
It records rank agreement, not accepted model state or logical storage ownership.

## Failure and resource ownership

Invalid readiness or results fail the group.
This includes stale identities, duplicates, missing transactions, different collective plans, and incomplete device work.
A rank failure also fails the group.
An expired startup or step deadline also fails the group.
A failed group cannot accept later results or start another step.
The gate returns no partial completion value.

The supervisor must do deadline checks while it waits, also when no messages arrive.
The gate does not contain a timer task.
The supervisor must fail the gate when a child exits or a transport operation fails.
The gate trusts completion metadata from the owned workers.
The native worker must wait for device and transfer completion before it sends that metadata.
An echoed digest alone does not prove collective order or correct device execution.

The gate owns no state leases, CUDA resources, or child processes.
The supervisor must keep those resources until all device work finishes or all old workers stop.
Cancellation or a failed gate is not sufficient permission to release them.
Before publication, the scheduler must validate model outputs and bind the completion to its own pending transactions.
The [state arena](state-arena.md) supplies logical leases, not that publication operation.

## Checks

CPU tests supply evidence for readiness, capacity limits, request placement, zero-row ranks, stale results, duplicate results, and deadlines.
They include msgpack completion messages.
They do not prove process cleanup, native collective execution, CUDA Graph safety, numerical accuracy, or model capacity.

```bash
scripts/with-env.sh cargo test -p oh-my-vllm-zmq-worker --test group
```
