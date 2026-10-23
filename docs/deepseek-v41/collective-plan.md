# Collective plans

## Scope

The [Rust plan](../../crates/zmq-worker/src/group/collectives.rs) defines an ordered communication contract.
The [Python reader](../../python/oh_my_vllm/operators/runtime/distributed.py) validates the same contract without GPU imports.
These modules specify communication, not model computation or accepted state.
They do not prove native execution or CUDA Graph safety.

The plan has a version, world size, and ordered operations.
Version 1 permits world sizes 1 and 4, with at most 4096 operations.
An empty operation list is valid.
The world size must agree with the caller's group topology.
Unknown fields, operations, and data types are invalid.

## Operations and buffers

Each operation uses one input buffer and a separate output buffer per rank.
Buffers contain contiguous elements of the specified data type.
Native execution must reject storage overlap and incorrect devices, shapes, types, or strides before enqueue.
The plan defines element counts, not byte offsets or raw device addresses.

| Operation | Count meaning | Input elements per rank | Output elements per rank |
|---|---|---:|---:|
| `all_reduce` | Elements reduced with sum | count | count |
| `all_gather` | Elements supplied by each rank | count | count times world size |
| `reduce_scatter` | Elements received by each rank after sum | count times world size | count |
| `route` | Square source-to-destination count matrix | Sum of the source row | Sum of the destination column |

Route input segments use destination-rank order.
Route output segments use source-rank order.
The matrix includes self-routes and zero-count routes.
Each rank receives the same matrix, also when it has no scheduled requests.
The native implementation must use matching send and receive counts.

Data types are `f32`, `bf16`, `f16`, `i32`, `i64`, and `u8`.
Counts are unsigned 64-bit integers.
Each buffer's total byte size must fit a signed 64-bit integer.
Validation checks multiplication and row and column sums before dispatch.
Python rejects Boolean, negative, and noninteger counts.
The Python reader makes an immutable copy of the validated operations.

## Canonical identity

SHA256 uses this byte sequence for all msgpack map orders:

1. The ASCII domain `oh-my-vllm/collectives/v1`, then one zero byte.
2. Version, world size, and operation count, each as an unsigned 32-bit little-endian integer.
3. Each operation's tag byte and data-type tag byte, in operation order.
4. Its count as an unsigned 64-bit little-endian integer, or all route counts in row-major order.

Operation tags are 0 for `all_reduce`, 1 for `all_gather`, 2 for `reduce_scatter`, and 3 for `route`.
Data-type tags are 0 for `f32`, 1 for `bf16`, 2 for `f16`, 3 for `i32`, 4 for `i64`, and 5 for `u8`.

The [rank gate](rank-control.md) calculates the digest and operation count from the validated plan.
One route operation counts as one operation, not as its individual native sends and receives.
The worker must independently validate the plan and compare the two calculated values before native execution.
It must report completion only after all planned operations and device work finish.

## Graph integration

The adapter must select fixed padded counts before capture.
Data-dependent routing cannot change the plan during replay.
The adapter must supply padding values and valid-row metadata.
The graph key must bind the plan identity and topology.
Its owner must keep communicator and buffer addresses valid through capture, replay, and completion.
The same ordered native operations must execute in Eager mode and CUDA Graph mode.

## Checks

Rust and Python checks use one fixed plan with all four operation types and uneven routes.
The fixture supplies a fixed digest and per-rank buffer counts.
The checks reject malformed topology, overflow, unsupported fields, and invalid types.
They also check that order, type, and route changes change the digest.
These CPU checks do not supply model numerical or GPU performance evidence.
