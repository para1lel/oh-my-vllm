# GPU measurement method

Use three separate collections: framework wall latency, a device timeline, and hardware counters.
The first supplies acceptance. The other two show behavior.
See the [formal protocol](../requirements/performance.md) and [commands](../testing/performance.md).

## Set the workload

Record source, loaded modules, compiler, checkpoint, shapes, process birth identities, and GPU identity.
Finish compilation and graph capture before timing. Put persistent state back to its baseline between comparisons.
Measure the complete operation, with conversion, copies, and reduction.
Use interleaved candidate/baseline pairs before a full framework collection.

Synthetic inputs can change numerical branches. Record branch frequencies from model inputs before selecting a specialization.
The [fixtures](../../development/kernels/fixtures.py) supply reproducible operator inputs, not a model-input distribution.

## Read the timeline

Use the [bounded Torch profiler](../profiling.md#small-torch-profiler-windows) or Nsight Systems for a small diagnostic workload.
Connect CPU launches and graph nodes to device intervals through correlation and stream identities.
CPU annotation intervals can end before their GPU work finishes. Do not filter kernels only by the CPU interval.
The [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/) gives queue time and GPU execution time.

With PDL, dependent kernels can overlap, also on one stream.
An interval can include dependency waits. Adding kernel durations can therefore count the same wall interval more than once.
Interval union gives covered GPU time, not the dependency critical path or exclusive compute time.
Use the completed request interval for phase latency. See [CUDA PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

## Collect counters separately

Begin with a small counter set. Record the profiler version, replay mode, passes, clock policy, and cache policy.
Nsight Compute can serialize launches and replay a kernel to collect counters.
Replay passes are not independent timing samples. Cache restoration and clock control can change the workload.

Use an applicable range replay mode when concurrency is necessary for execution.
Read the [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) before changing replay policy.

The tested command uses `--cache-control none --clock-control none`.
It leaves clocks and cache uncontrolled. It does not show equal cache contents between passes or fixed clocks.
Keep profiler durations diagnostic. Do not run Torch profiler and Nsight Compute together.

| Question | Observation |
| --- | --- |
| Enough independent work? | Total warps, grid size, waves, and workload distribution |
| Resource limit? | Registers, shared memory, resident blocks, and spills |
| Instructions ready? | Eligible warps, issue activity, and dependency stalls |
| Which compute resource? | Tensor, FP32, and special-function pipeline activity |
| Which memory resource? | HBM/L2 bytes, bandwidth, shared-memory traffic, and conflicts |

Keep counter units and denominators. An instruction-normalized stall ratio is not a wall-time percentage.
SM throughput is a pipeline metric, not measured FLOPS divided by the model peak.
Occupancy alone does not show the cause of latency. See [NVIDIA compute triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/).
Report counters that the collection does not supply.

## Reproduced observations

The [source-bound diagnostic summary](../evidence/optimization-observations/verification.json) records the original reports and input scope.
Nsight Compute 2026.1.1 collected counters for one scan from the 32144-token GDN fixture, with 16 replay passes.
Its grid has 192 blocks and 768 warps across 148 SMs. The shared-memory limit is two resident blocks per SM.

The reported 0.65 waves equals `192 / (148 * 2)`. It does not mean 35% of SMs are idle.
Average active warps are 5.251 per SM, with 8.205% occupancy and zero register spills.
Tensor activity is 0.815% for this invocation. Branch frequencies were not collected.

Changing block count without increasing total independent warps cannot show improved latency hiding.
Smaller blocks might improve workload distribution or shared-memory residency. A paired measurement must test that hypothesis.
The 12.456-ms profiler duration is not formal operator timing.

The fused normalization/FP8 gate-up fixture has 32144 tokens, width 5120, and 34816 output columns.
One CUTLASS kernel has 57.649% Tensor Core activity and 31.319% FMA pipeline activity, with 18 replay passes.
Its 5.457-ms profiler duration does not include the other kernels in the operation.
Examine state-scan dependencies and block-scale accumulation as separate research directions.
These observations do not show a complete framework speedup.

### Model-input branches

A separate instrumented model run used one request with token 42 repeated 32768 times and manually supplied cache tables.
It recomputed the current classifier on each GDN layer's inputs, without changing the original prefill result.
After capture and replay, 46 of 48 layers used the precision route for all chunks.
Across the 48 layers, the counts were 1153392 precision, 26256 constant-key, and zero matrix chunk/head pairs.
The captured observer buffers were read after synchronization.

This is one ordinary-model input, not a prompt distribution or formal scheduler collection.
Checkpoint byte hashes and physical layer IDs were not collected.
The run includes capture and observation cost. Its duration is not acceptance timing.
Use these branch counts to prioritize precision-path research, without expanding their input scope.

## Select the change

State one hypothesis and the counter or dependency that gives a reason for it.
Keep numerical, rounding, state, and replay contracts equal. Reject a faster candidate without equivalence evidence.
Compare complete chains, then repeat the applicable framework rows with profiling disabled.
Keep failed attempts and source identities. Stop owned GPU processes after each collection.
