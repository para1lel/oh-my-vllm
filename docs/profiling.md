# Profiling and diagnostics

Use profiling to examine measured latency. Formal timing uses [performance contracts](requirements/performance.md) with profiling disabled.
Keep original traces in external storage and source-bound summaries in the [acceptance index](acceptance.md).

## Host and device time

[Worker logs](../python/oh_my_vllm/worker/logging_utils.py) identify requests, steps, and host durations.
[Execution records](../python/oh_my_vllm/performance/execution.py) connect model work to the theoretical DAG.
Use CUDA Events or profiler traces for GPU attribution. Host enqueue time is not GPU execution time.
Concurrent kernel durations must not be added as wall latency. Record streams, overlap, and dependencies.

## Stable measurement

Keep compilation, graph capture, and cache eviction out of measured repetitions.
Check source, loaded modules, checkpoint, environment, and GPU identities before and after collection.
Keep failed and interrupted attempts. Report unavailable counters.

[Capture transactions](../python/oh_my_vllm/worker/graph_capture.py#L31) <!-- source: def capture_graph( -->
keep graph storage and workspace lifetimes consistent. [Runtime identity](../python/oh_my_vllm/worker/runtime.py) records the loaded implementation.

## Operator tuning

Measure complete operations with required copies, reductions, scale packing, and output conversion.
Compare candidate and repository baseline in interleaved pairs. Check numerical and mutation contracts first.
Profile separately from timing and check the full framework after selection.
A fused operation can change resource pressure and overlap. Select it from measured overall benefit.

Use the [operator harness](../development/kernels/) and [kernel guide](kernels.md).
Implementation schedules and thresholds belong in [CUDA source](../python/oh_my_vllm/kernels/cuda_backend/), with evidence from the selected candidate.

## Memory and faults

Use [worker diagnostics](../python/oh_my_vllm/worker/model_runner.py) and [graph ownership](architecture/compilation.md) for memory attribution.
RSS, allocator statistics, tensor bytes, and CUDA reserved bytes measure different quantities.
A one-hour observation does not prove a memory plateau or stability after that hour.
Run allocator OOM recovery separately from stable service collection. Release owned processes after profiling or failure.
