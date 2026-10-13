# Runtime ownership

## Request flow

A request moves through admission, preparation, scheduling, execution, and validated commit.
Each stage has one owner. GPU work must not publish an unvalidated prefix.
Cancellation takes effect at completed operation boundaries. Worker failure invalidates its state before replacement.

Read [service admission](../../crates/zmq-worker/src/serving/), [request preparation](../../python/oh_my_vllm/worker/serving.py),
[worker execution](../../python/oh_my_vllm/worker/model_runner.py), [target execution](../../python/oh_my_vllm/worker/target_execution.py), and [scheduler commit](../../crates/scheduler/src/lib.rs).
[Service contracts](../serving.md) define observable API behavior.

## Rust modules

Rust owns request lifetime, accepted history, scheduling budgets, logical slots, and response delivery.
[Scheduler](../../crates/scheduler/src/lib.rs), [cache coordination](../../crates/kv-cache/src/coordinator.rs),
[child ownership](../../crates/zmq-worker/src/client.rs), and [service resources](../../crates/zmq-worker/src/serving/resources.rs)
are the implementation entry points.

## Python modules

Python owns tokenization, grammar and sampling state, model tensors, graph storage, and GPU execution.
[Bridge](../../python/oh_my_vllm/worker/zmq_bridge.py), [batch plans](../../python/oh_my_vllm/worker/batch_plan.py),
[target model](../../python/oh_my_vllm/models/qwen.py), and [sampler](../../python/oh_my_vllm/worker/sampler.py)
show those boundaries. Python uses Rust-supplied logical IDs.

## Transport contract

The processes exchange msgpack messages through ZMQ DEALER sockets.
Correlated replies must belong to the active request and worker epoch. Late cancelled preparation cannot create live state.

Wire fields and defaults come from the [Rust protocol](../../crates/zmq-worker/src/protocol.rs)
and [Python protocol](../../python/oh_my_vllm/worker/protocol.py).
Their integration tests verify the shared boundary. Do not maintain another field list here.
