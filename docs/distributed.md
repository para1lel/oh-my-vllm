# Distributed execution contracts

## Scope

Rank control supplies transaction ownership, GPU leases, and native collectives for model adapters.
The [adapter contract](adapters.md) separates these mechanisms from model execution.
Each adapter must pass its numerical, state, memory, and performance gates before service use.

## GPU leases

Use the [GPU wrapper](../development/serving/gpu.py):

```bash
scripts/with-env.sh scripts/with-gpu.sh COMMAND [ARGS...]
```

Put host settings in ignored `LOCAL.md`.

| Setting | Contract |
|---|---|
| `OH_MY_VLLM_GPU_COUNT` | `1` or `4`, default `1` |
| `OH_MY_VLLM_GPU_UUID` | One UUID, valid only for count `1` |
| `OH_MY_VLLM_GPU_UUIDS` | Ordered UUID list with the selected count |
| `OH_MY_VLLM_GPU_EXCLUDE_INDICES` | Excluded physical indices, default empty |
| `OH_MY_VLLM_RUN_ID` | Caller ID, or a generated UUID when empty |

UUIDs must be distinct, canonical, lowercase, and have the `GPU-` prefix.
Do not combine the two UUID settings.
Lists cannot contain spaces or empty entries.
Exclusions override explicit selection.

Invalid settings, missing devices, and non-B200 selections fail.
Busy explicit selections wait.

Idle devices have at most 64 MiB used memory, zero utilization, and no compute process.
Invalid or unavailable measurements cannot prove idleness.
Automatic selection uses physical index order.
Explicit UUID order sets logical rank order.
The wrapper replaces `CUDA_VISIBLE_DEVICES`.

Locks use increasing UUID order and `/tmp/oh-my-vllm-${uuid}.lock`.
Lock files must be regular files.
An attempt releases partial locks before waiting.
After all locks are acquired, inventory and process checks run again.
Failed attempts release the group.
Retries wait 10 seconds, without a deadline or fairness guarantee.

The wrapper uses `exec` with inherited lock descriptors.
Keep descriptors until all owned GPU children stop.
Use `pass_fds` when ownership moves to children started with `close_fds=True`.
Do not remove live lock files.
Cooperative locks cannot prevent unrelated GPU use.

The wrapper neither changes clocks nor stops other users' processes.
[CUDA identity checks](../development/serving/device.py) verify the visible count and ordered UUIDs.
Single-GPU consumers reject four-device selections.

## Rank startup

The [group client](../crates/zmq-worker/src/rank_client.rs) validates configuration before process startup.
It uses group protocol version 3, a positive epoch, expected descriptor, capacity limits, ordered UUIDs, and positive deadlines.
Each owned process group gets a separate DEALER endpoint.
The caller owns the IPC directory.
Endpoint paths must be absent.
Protect IPC entries from changes by other processes.

Each worker sees only its assigned UUID at local CUDA ordinal zero and validates that identity before readiness.

The client does not own GPU locks.

Replies contain one msgpack frame, at most 1 MiB.
Connection identity supplies the source rank.
One startup deadline covers process creation, sends, bootstrap, and readiness.
Retries do not extend it.
Exit and deadline checks continue when no messages arrive.

For NCCL, only rank zero supplies the opaque 128-byte unique ID.
After valid bootstrap replies, Rust sends that ID and a new random connection nonce to each rank.
Final readiness must return the nonce after native initialization succeeds.
Early, duplicate, stale, or incorrect replies fail startup.
Transport `none` supports CPU peers, not GPU execution.

The [rank gate](../crates/zmq-worker/src/group.rs) validates each descriptor field except negotiated capacities.
Capacities must meet requested minima and leave null slots and queue sentinels.
Shared views must agree with their producers.
Budgets count owned logical slots and null slots, without duplicate shared views.
They are not GPU byte budgets.
Only complete readiness supplies per-rank capacities and common minima.

## Transactions and publication

Steps have increasing IDs and unique request and transaction IDs.
Transaction IDs must exceed those from previous steps.
Request and row limits apply before dispatch.
Invalid local plans leave a ready group usable.

TP/TEP replicate request work and use output owner rank zero.
DEP keeps each request's owner.
Each rank receives a schedule and acknowledges completion.

Zero-row ranks also follow this rule.
Any prefill selects Eager execution for the whole group.
Otherwise, the whole group uses CUDA Graph execution.

The [typed interface](../crates/zmq-worker/src/rank_client/model.rs) checks request order, token slices, positions, row counts, prefill, vocabulary, and context bounds.
State keys and bindings must match live transactions, cache schema, rank placement, capacity, and generation.
Write bindings contain exactly the allocated blocks.
Read bindings come from transaction owners.
Neither permits duplicates.

Copied metadata supplies no ownership.
Model layouts must use declared bindings.
The shared client cannot examine arbitrary layout references.

Trusted [policy callbacks](../crates/scheduler/src/policy.rs) validate full identity sets before ownership transfer.
The client keeps detached transactions before its first asynchronous wait.
The scheduler then blocks further scheduling and direct updates.
Request cancellation cannot release detached transaction owners.

Model payloads contain scheduled chunks and state assignments, not full prompt histories.
The [serializer](../crates/zmq-worker/src/rank_client/encoding.rs) bounds payloads to 64 MiB, plus 1 MiB for command metadata.
It rejects unknown container lengths, `serde(flatten)`, and `collect_str`.
Replies have a 1 MiB limit.

SHA256 binds serialized bytes but does not authenticate workers.
Metadata-only and model replies cannot substitute for each other.

Completion must match the epoch, step, rank, payload, local transactions, and collective plan.
Workers report full operation counts only after device and transfer completion.
An echoed digest alone proves no execution.

The client rejects missing, duplicate, unowned, or invalid outputs and checks speculation limits.
The scheduler validates the full batch before the policy selects accepted state.
Reply arrival order cannot change output order.
No subset of a failed batch becomes accepted state.

## Collective plans and storage

The [Rust plan](../crates/zmq-worker/src/group/collectives.rs) and [Python reader](../python/oh_my_vllm/operators/runtime/distributed.py) share canonical validation and SHA256 encoding.
Version 1 supports world sizes 1 and 4, at most 4096 operations, and empty plans.
World size must match topology.
Unknown fields, operations, and types fail validation.
Counts are unsigned 64-bit integers.
Buffer byte totals must fit signed 64-bit integers.

Validation checks products and route sums before dispatch.

| Operation | Input elements per rank | Output elements per rank |
|---|---:|---:|
| `all_reduce` with sum | count | count |
| `all_gather` | count | count times world size |
| `reduce_scatter` with sum | count times world size | count |
| `route` | Source-row sum | Destination-column sum |

Routes include self-routes and zero counts.
Inputs use destination order, outputs source order.
All ranks receive the same matrix.
One route counts as one operation.
Workers independently validate the plan, digest, and operation count.
Graph keys bind plan identity and topology.

Counts and addresses stay fixed through capture and replay.
The adapter supplies padding and valid-row metadata.

The [native API](../python/oh_my_vllm/operators/runtime/collectives.py) uses one control thread and the caller stream.
Create the communicator before capture.
Poll initialization to `"ready"` before capture.
External deadlines and process termination are necessary because native calls can stall.
Native checks reject incorrect counts, types, devices, strides, bounds, and overlap.
Contiguous flat tensors on `cuda:0` use separate input and output storage.

[Private staging](../python/oh_my_vllm/operators/runtime/cuda/collective_contracts.cuh) protects NCCL LL/LL128 enclosing reads.
A nonempty span uses NCCL directly only when its address and byte extent are 16-byte aligned.
Unaligned nonempty spans get zero-initialized, 16-byte-rounded private storage included in Device accounting.
Zero-count buffers use no private storage.
Prepared owners keep caller and staging tensors.
Padding changes neither wire counts nor the digest.

Input copies run before NCCL and repeat on replay.
Output copies follow host enqueue completion.

## Native completion and closure

After Eager `prepared.run()`, poll `comm.poll()` to `"ready"` before result access or reuse.
That state proves completion through output copies.
Complete Eager warmup before capture.
Call `prepared.prepare_capture()` once for each operation before capture.

Capture the same ordered operations on all ranks.
For each operation, call `prepared.run()`.
Poll `comm.poll_enqueue()` to `"ready"` with a deadline.
Then call `prepared.finish_enqueue()`.

Finish the last operation before the capture callback returns.

Host enqueue readiness does not prove device completion.
Do not use `comm.poll()` during capture.

On capture error, cancellation, or timeout, attempt bounded CPU/IPC error reporting.
Then call `os._exit(nonzero)`.
Do not unwind through Device capture cleanup or call CUDA, abort, or buffer release on this path.

After replay, call `comm.record_completion()` on the caller stream.
Poll to `"ready"` before result access, reuse, or rank completion.
Keep prepared objects and workspace tensors throughout graph lifetime.

For normal native closure, destroy source graphs and executables.
Poll `comm.abort_ready()` to true with a deadline.
Then call `comm.abort()`.
Poll `comm.poll()` to `"aborted"` before releasing prepared objects and tensors.
This also waits for unused capture reservations.

Graph callbacks only set flags.
Caller-thread cleanup releases owners.
Forced exit does not prove graceful Compute Sanitizer cleanup.

Rank failure, invalid replies, transport errors, or expired deadlines fail the group permanently.
Any typed execution error disables the scheduler and closes the group.
The supervisor stops all owned process groups and descendants before releasing transactions or returning an execution error.
Cancellation leaves the scheduler disabled and transactions owned.
Dropping the client starts background cleanup.

Call `close`.
Wait for its result before replacement with a new scheduler and epoch.

Cancelled close waits do not cancel cleanup.
IPC removal checks recorded file identities.
[Normal shutdown](../crates/zmq-worker/src/rank_client/shutdown.rs) accepts only bound `closed` replies and zero exits before its deadline.
Process cleanup follows.
Neither cancellation nor a deadline proves termination.

## Commands and gates

Run CPU contract checks:

```bash
scripts/with-env.sh cargo test -p oh-my-vllm-zmq-worker --test group --test rank_client
scripts/with-env.sh python -m unittest tests.test_collective_plan tests.test_collectives
```

Build the optional native transport:

```bash
scripts/with-env.sh python -m development.native_build --module collectives
```

The [native peer](../development/operators/collective_peer.py) supplies synthetic execution, not a DeepSeek adapter.
CPU checks do not prove GPU numerics, graph safety, memory capacity, or performance.
Model preparation, physical layouts, accepted-state selection, and service integration are subsequent work.
