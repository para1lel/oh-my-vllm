# Process boundary

## Control and state

The [service](../crates/zmq-worker/src/lib.rs) owns CLI, HTTP APIs, worker lifecycle, and transport.
The [scheduler](../crates/scheduler/src/lib.rs) owns admission, request queues, token budgets, preemption, and validated state commit.
The [cache policy](cache-policy.md) supplies model-specific allocation, prefix lookup, state assignments, and restoration.
The [KV crate](../crates/kv-cache/src/lib.rs) supplies logical storage and ownership mechanisms.

Scheduling follows the priorities in the [runtime contract](requirements.md).
The scheduler validates all output rows before it changes accepted history or publishes a prefix.
A transaction failure cannot publish an unvalidated prefix.
Preemption keeps accepted history and discards new drafts.
Physical state release also depends on device and transfer completion.

## Model execution

The [bridge](../python/oh_my_vllm/worker/zmq_bridge.py) owns ZMQ DEALER and msgpack frames.
The [registry](../python/oh_my_vllm/models/registry.py) selects a model adapter before GPU imports.
Each adapter supplies computation, codecs, physical cache, and accepted-state selection through the [common contract](adapters.md).
The explicit `stub` backend supplies service-test behavior only.

Model weights follow the adapter's device and host placement plan.
Temporary buffers can be reused across layers that execute in sequence.
A recorded graph keeps all referenced buffers and resource owners.
Large scheduler steps can use smaller native batches.
The [execution rules](execution.md) specify fragment selection, rank coordination, and Graph resource ownership.

Target and proposal distributions have separate storage until verification completes.
Only accepted tokens change constraints and request history.
A prefill chunk before the final chunk returns no output tokens.
A completed generation step returns accepted proposals and a target token, subject to request stop conditions.
New proposals are not accepted history.

The [single-worker protocol](../crates/zmq-worker/src/protocol.rs) pairs replies with RPC IDs and scheduled request results.
The [rank protocol](distributed.md) adds group epochs, steps, transactions, and all-rank completion checks.
Model-specific clients translate logical assignments into the selected wire layout.

## GPU computation

| Part | Responsibility |
|---|---|
| Python adapter | Call operators in dependency order and own model state |
| Native operators | Compute values with the selected precision and state rules |
| Device resources | Allocate storage and manage streams, Graphs, leases, and release |

The [operator design](operators/design.md) separates semantic interfaces from kernel and library boundaries.
Eager execution and Graph recording use the same native operators.
Fusion can cross internal stages when the complete operation benefits and ownership rules hold.

## Failure and resources

The service reserves request count and bytes before body reads or stored-history copies.
Streams and stored responses have count and byte limits.
Cancellation releases state at safe operation boundaries.
A worker failure fails affected requests while the HTTP listener stays active.

Recovery removes old workers and logical identities before admission in a new epoch.
It does not replay output or tool calls.
Cleanup stops owned process groups and removes their IPC paths.
Request, step, and shutdown deadlines have separate scopes.
See the [service guide](serving.md) for commands and limits.
