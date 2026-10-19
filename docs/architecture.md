# Process boundary

## Rust

The [KV crate](../crates/kv-cache/src/lib.rs) controls block identities, reference counts, LRU, and prefix reuse.
The [scheduler](../crates/scheduler/src/lib.rs) controls request admission, decode priority, prefill chunks, preemption, and state commit for speculative tokens.

The [service crate](../crates/zmq-worker/src/lib.rs) controls the CLI, HTTP APIs, worker lifecycle, and transport.

The scheduler selects decode requests before prefill requests.
It divides the available token budget between prefill requests, with adjustments for cache boundaries.

A scheduled step reserves FA and GDN state.
The scheduler validates all output rows before it updates token history or publishes a prefix.

The scheduler cannot publish an unvalidated prefix after a transaction error.
A preempted request keeps its accepted history. Preemption removes its new drafts.

## Python

The [bridge](../python/oh_my_vllm/worker/zmq_bridge.py) controls the DEALER socket and msgpack frames.
The [inference worker](../python/oh_my_vllm/worker/inference.py) supplies token preparation and model computation.
The [placeholder](../python/oh_my_vllm/worker/placeholder.py) is used only for `--model stub`.

Rust keeps logical cache ownership. Python stores tensors and private recurrent states.
The worker selects an accepted state slot. It does not copy the full recurrent state for acceptance.

Target and proposal distributions have separate storage until verification completes.
The constraint state changes only with accepted tokens.

Model weights stay in GPU memory. Eager paths reuse forward buffers across language layers.
Each recorded graph keeps the buffers referenced by its operators.
The worker divides a large scheduler step into native batches when necessary.
The [execution rules](execution.md) select Eager or CUDA Graph execution for the full Execute message.

FA and GDN state pools have separate capacities.
Media requests use separate prefix cache identities.

The [protocol types](../crates/zmq-worker/src/protocol.rs) give the fields for Init, Register, Prepare, Execute, Abort, Stats, and Shutdown messages.
Replies use the related RPC ID. Each scheduled request has one result row.

A prefill chunk before the final chunk supplies no output tokens.
A final prefill or decode step supplies accepted drafts plus one target token.

The accepted history does not contain new drafts.

## GPU computation

| Part | Responsibility |
|---|---|
| Python model | Call operators in dependency order and manage tensor storage |
| Native operators | Compute values with the model's precision and private state rules |
| Device resources | Manage streams, CUDA Graph recording, replay, and release |

The [model](../python/oh_my_vllm/worker/model.py) calls the [operator stages](operators/fusion.md) through all language layers.
Eager execution and CUDA Graph recording use the same native operators.
The [device wrapper](../python/oh_my_vllm/worker/device.py) supplies graph resources.
Each workspace keeps its weights, cache, tensors, and resource leases until graph closure.
FP8 conversion and tensor checks are shared across operator families.

## Failure and resources

The service reserves request count and byte capacity before it reads a body or copies stored history.
The stream buffers and stored responses have count and byte limits.

Cancellation releases the request state at operation boundaries.
A worker failure stops the affected requests. The HTTP listener stays active.

Recovery removes the worker and logical cache identities for one epoch before it admits requests for the next epoch.
The service does not replay output or tool calls.

Worker cleanup stops its process group. It removes the IPC path that it creates.
The request, step, and shutdown deadlines have different scopes.

Use [the service guide](serving.md) for commands and worker limits.
