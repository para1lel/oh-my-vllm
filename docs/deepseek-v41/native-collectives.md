# Native collective owner

## Scope and setup

The [Python API](../../python/oh_my_vllm/operators/runtime/collectives.py#L76) <!-- source: class Communicator: --> supplies the native transport for a [collective plan](collective-plan.md).
Rust controls bootstrap, operation order, rank deadlines, and group termination.
This transport does not enable DeepSeek model execution.
Import of the API alone does not load NCCL.
The optional native module uses the repository TVM FFI ABI3 loader.

Each rank process must see exactly one CUDA device at local ordinal 0.
`local_device()` returns the CUDA device count and UUID.
The worker must compare this UUID with the expected group UUID before bootstrap.
`versions()` returns the NCCL header and loaded library version codes.
`unique_id()` returns 128 opaque bytes.
The supervisor sends the same ID to all ranks.

Construct `Communicator(world, rank, unique_id)` before capture.
Poll `comm.poll()` to `"ready"` with an external deadline.
Initialization uses the installed NCCL configuration with nonblocking mode.
Use one control thread and the caller stream for related work.

Native calls have no internal wait loop.
A library call can still exceed a deadline.
The supervisor must have a process termination path.

## Logical buffers and private storage

`comm.prepare(kind, source, destination)` accepts `all_reduce`, `all_gather`, and `reduce_scatter`.
Reduction uses sum.
`comm.routing(routes)` accepts ordered `Route("send", peer, tensor)` and `Route("recv", peer, tensor)` entries.
Zero-count entries keep their position and matching call.

Supported types are `float32`, `bfloat16`, `float16`, `int32`, `int64`, and `uint8`.
Supply contiguous flat owning tensors on `cuda:0`.
The native checks reject incorrect counts, types, strides, devices, bounds, and storage overlap.
Checked views can have addresses that are not aligned to 16 bytes.
Remote count agreement belongs to the canonical plan.

The [storage check](../../python/oh_my_vllm/operators/runtime/cuda/collective_contracts.cuh#L67) <!-- source: inline int64_t collective_staging_bytes( --> accounts for NCCL LL and LL128 reads of enclosing byte spans.
A nonempty buffer uses the direct path only when its address and byte extent are aligned to 16 bytes.
Other nonempty buffers use private storage with checked capacity rounded up to 16 bytes.
The allocation is aligned to 16 bytes.
The native owner initializes the full private allocation to zero.
Zero-count buffers use no private storage.

The existing Device allocation export supplies private storage.
Device allocation accounting includes this storage.
The [prepared owner](../../python/oh_my_vllm/operators/runtime/cuda/collective_owner.cuh#L42) <!-- source: struct CollectivePlanState { --> keeps the caller tensors and private tensors.
It copies logical input bytes before NCCL submission.
It copies logical output bytes after host enqueue finishes.

Padding does not change wire counts, tensor shapes, route order, or the canonical digest.
Storage addresses and copy sizes are fixed before capture.
Input copies execute again on each replay.

## Eager and capture completion

For Eager execution, call `prepared.run()`.
Poll `comm.poll()` to `"ready"` before result access or buffer reuse.
This poll checks NCCL status before CUDA calls.
After host readiness, it submits output copies and a completion event.
Eager `"ready"` proves device completion through these copies.

Complete Eager warmup before capture.
Call `prepared.prepare_capture()` once for each operation.
Capture the same ordered operations on all ranks.
Use this sequence for each operation:

```python
try:
    prepared.run()
    while True:
        status = comm.poll_enqueue()
        if status == "ready":
            break
        if status != "pending":
            raise RuntimeError(status)
        check_external_deadline()
    prepared.finish_enqueue()
except BaseException as error:
    try:
        report_error_with_deadline(error)
    finally:
        os._exit(1)
```

`poll_enqueue()` only checks NCCL host progress.
It does not call CUDA or prove device completion.
`finish_enqueue()` submits output copies on the original capture stream.
It has no wait loop.
The next operation is blocked until this finalization finishes.

Finalize the last operation before the capture callback returns.
Do not use `comm.poll()` during capture.

After replay, call `comm.record_completion()` on the caller stream.
Poll `comm.poll()` to `"ready"` before result access, reuse, or RankResult.
Keep the full workspace and prepared objects through the graph context.

## Failure and closure

Handle capture submission, polling, and finalization errors inside the capture callback.
On error, timeout, or cancellation, attempt only bounded CPU/IPC Error reporting.
Then terminate the rank with `os._exit(nonzero)`.
Do not return or raise through Device capture cleanup.
Do not call CUDA, abort, or release buffers on this path.

Keep the supervisor deadline active if reporting or native code stalls.
Rust must terminate and reap all group ranks before group lease release.

For normal closure, destroy source graphs and all graph executables.
Poll `comm.abort_ready()` to true with a deadline.
False means a graph lease is still live.
This query only reads callback flags.
Then call `comm.abort()` and poll to `"aborted"`.

After abort, this poll also waits for unused capture reservations to release.
Release prepared objects and workspace tensors after this sequence.

Graph callbacks only set completion flags.
Caller-thread cleanup releases owners.
Unfinished or failed work keeps its owners until explicit cleanup or process exit.
Forced exit is not graceful Compute Sanitizer evidence.
