# Typed model execution

## Scope

The [group client](../../crates/zmq-worker/src/rank_client/model.rs) connects typed scheduler inputs to four-rank execution.
The model policy supplies `ModelState<L>` with a transaction key, read blocks, write blocks, and model-selected layout `L`.
The shared client does not interpret physical tensor layouts.
The Qwen service keeps its existing client and wire format.
DeepSeek model execution is not enabled.

The [transaction policy](../../crates/scheduler/src/policy.rs) extends the shared cache policy.
Its detach operation transfers active state transactions from the policy to the group client.
Its restore operation returns these owners without publication of accepted state.
Each operation must validate its full identity set before it changes ownership.
These callbacks are trusted model code, not worker messages.

## Inputs and ownership

`execute_scheduled` takes a scheduler, its scheduled batch, and a step plan.
The client checks request order, scheduled token slices, computed positions, row counts, and the prefill decision.
Each input token must be less than the vocabulary size.
The computed end position must not exceed the descriptor's context limit.
The rank gate checks group request limits, row limits, and transaction identities.
It validates the collective plan and completion metadata, not the order of native calls inside a worker.

Each state key must be the same as the active transaction key.
Write bindings must contain all allocated transaction blocks and no other blocks, without duplicates.
Read bindings must come from the transaction's read or write owners, without duplicates.
Bindings must agree with the negotiated common cache schema, rank placement, slot capacity, and generation.
Copied metadata cannot create storage ownership.

The model layout must use only the declared bindings.
The shared client cannot examine references inside an arbitrary layout type.

The client keeps detached owners before its first asynchronous wait.
The scheduler then rejects direct updates and does not schedule more work.
An aborted request can release its request owners, but cannot release the client's transaction owners.

## Wire format

Group protocol version 3 adds `execute_model` and `model_result`.
Metadata-only execution is still available.
The two execution paths reject replies of the other kind.

An `execute_model` command contains the ordinary step dispatch, a SHA256 digest, and a binary msgpack payload.
The payload contains scheduled token chunks, computed positions, typed state assignments, and finished and preempted request IDs.
It does not contain full prompt histories.
Model preparation and history transfer use different lifecycle integration.

Serialization stops before the payload exceeds 64 MiB.
All serialized maps and sequences must have known lengths.
The recursive serializer guard rejects unknown lengths before msgpack can allocate temporary container storage.
Thus, model layouts cannot use `serde(flatten)` or unknown-length iterators.
Strings must be prepared before serialization.
The guard rejects `collect_str` formatting.

The command limit has an additional 1 MiB for dispatch metadata.

Each rank reply has a limit of 1 MiB.
SHA256 is calculated from the serialized payload bytes, not a reconstructed dictionary.
It detects disagreement but does not authenticate a worker.

## Completion and publication

Each rank returns a completion record and the payload digest.
Each rank returns model outputs only for requests that it owns.
TP4 and TEP4 use output owner rank zero.
DEP4 uses each request's scheduled owner rank.
Ranks with no local requests must still acknowledge completion.

The client rejects missing results, duplicate results, and invalid tokens.
It also rejects results for requests that the sending rank does not own.
Result token counts and draft counts must satisfy the descriptor's speculation limits.
The rank gate validates all completion records before publication can continue.

The scheduler then validates the full output against its in-flight requests, with checks of scheduled drafts and expected verified token counts.
Only after these checks does the policy receive its transactions and select accepted state through the existing update path.
Output order is the same as the scheduler's request order, not reply arrival order.

## Failure and cancellation

Any execution error disables the scheduler's execution path.
It also starts group closure and waits for owned-process termination before returning the error.
No valid subset of a failed batch becomes accepted state.

Cancellation of the execution future leaves the scheduler disabled and the client holding its transactions.
Call `close`.
Wait for termination before replacement.
Dropping the client starts the same background cleanup.
Use a new scheduler and epoch for replacement.
The [process ownership guide](rank-supervisor.md) specifies cleanup and shutdown boundaries.

## Subsequent integration

CPU peers check message handling, output validation, state ownership, and failure cleanup.
They do not supply model numerical or GPU execution evidence.
DeepSeek cache policy, physical layout, preparation protocol, operators, and service integration are subsequent work.
The [phase plan](plan.md) and [validation plan](validation.md) control these subsequent steps.
