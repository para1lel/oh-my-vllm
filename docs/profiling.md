# Profiling and diagnostics

Use profiling to examine measured latency. Formal timing uses [performance contracts](requirements/performance.md) with profiling disabled.
Keep original traces in external storage and source-bound summaries in the [acceptance index](acceptance.md).

## Host and device time

[Worker logs](../python/oh_my_vllm/worker/logging_utils.py) identify requests, steps, and host durations.
[Execution records](../python/oh_my_vllm/performance/execution.py) connect model work to the theoretical DAG.
Use CUDA Events or profiler traces for GPU attribution. Host enqueue time is not GPU execution time.
Concurrent kernel durations must not be added as wall latency. Record streams, overlap, and dependencies.

## Small Torch profiler windows

The benchmark profiler is disabled by default. Use it only for a small diagnostic run:

```bash
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --max-model-len 8192 --num-gpu-blocks 64 bench --input-len 128 --output-len 16 --warmup 1 --repetitions 1 --torch-profile "$EVIDENCE_DIR/timeline.json"
```

Use a new external trace path. Optional `--torch-profile-skip-steps` and `--torch-profile-steps` select the window.
The window counts nonempty worker steps across the run, with warmup, rather than benchmark repetitions.
The CLI limits request length and batch size, not model capacity. The worker checks total query tokens and limits active steps.

[Profiler](../python/oh_my_vllm/worker/torch_profile.py#L41) <!-- source: class BenchmarkProfiler: -->
exports a Chrome trace and logs each CUDA kernel's start, duration, device, stream, and available operator correlation.
These are device timeline intervals. Graph replay can have kernel names without a separate CPU operator identity.
The profiler does not add CPU enqueue durations to GPU runtime.
Keep this collection separate from formal latency measurements and other CUPTI subscribers.
See the [Torch profiler API](https://docs.pytorch.org/docs/2.14/profiler.html).

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
