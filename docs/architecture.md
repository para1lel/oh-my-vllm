# Process boundary

## Rust

The [KV crate](../crates/kv-cache/src/lib.rs) controls block identities, reference counts, LRU, and prefix reuse.
The [scheduler](../crates/scheduler/src/lib.rs) controls request admission, decode priority, prefill chunks, preemption, and state commit for speculative tokens.

The [service crate](../crates/zmq-worker/src/lib.rs) controls the CLI, HTTP APIs, worker lifecycle, and transport.

A scheduled step reserves FA and GDN state.
The scheduler validates all output rows before it updates token history or publishes a prefix.

The scheduler cannot publish an unvalidated prefix after a transaction error.
A preempted request keeps its accepted history. Preemption removes its new drafts.

## Python

The [bridge](../python/oh_my_vllm/worker/zmq_bridge.py) controls the DEALER socket and msgpack frames.
The [placeholder](../python/oh_my_vllm/worker/placeholder.py) supplies token preparation and simulated execution.

The inference backend will use this process boundary.
Rust keeps logical cache ownership. Python will store the tensors. It will do the GPU computation.

The [protocol types](../crates/zmq-worker/src/protocol.rs) give the fields for Init, Register, Prepare, Execute, Abort, Stats, and Shutdown messages.
Replies use the related RPC ID. Each scheduled request has one result row.

A prefill chunk before the final chunk supplies no output tokens.
A final prefill or decode step supplies accepted drafts plus one target token.

The accepted history does not contain new drafts.

## Failure and resources

The service reserves request count and byte capacity before it reads a body or copies stored history.
The stream buffers and stored responses have count and byte limits.

Cancellation releases the request state at operation boundaries.
A worker failure stops the affected requests. The HTTP listener stays active.

Recovery removes the worker and logical cache identities for one epoch before it admits requests for the next epoch.
The service does not replay output or tool calls.

Worker cleanup stops its process group. It removes the IPC path that it creates.
The request, step, and shutdown deadlines have different scopes.

Use [the service guide](serving.md) for commands and placeholder limits.
