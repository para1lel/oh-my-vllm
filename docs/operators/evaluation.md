# Operator evaluation

Use the [model contracts](model.md) and [fusion plan](fusion.md).
The [workload model](../../development/kernels/workload.py) supplies six request profiles.
Use its 7 serving boundaries for acceptance. Do not apply the gate to the 18 internal stages.

The serving boundaries are plans. The FFN provider is a baseline.
The other serving providers are not available.
Use the [development commands](development.md) for the available tools.

## Shape distribution

| Profile | Weight | Batch | Text input | Output | Mode | Media |
|---|---:|---:|---:|---:|---|---|
| Short text | 0.20 | 1 | 784 | 128 | ordinary | None |
| Text batch | 0.20 | 4 | 32768 | 256 | ordinary | None |
| Long text | 0.20 | 1 | 131072 | 512 | MTP4 | None |
| Image | 0.15 | 1 | 1024 | 256 | ordinary | 448x448 |
| Image batch | 0.15 | 4 | 4096 | 256 | MTP4 | 896x896 |
| Video | 0.10 | 1 | 1024 | 128 | MTP4 | 8 frames, 448x448 |

These weights are a simple request model. They are not measurements of production traffic.
The model uses batches with equal input lengths and balanced prefill chunks.

MTP verification has five rows per request. In the model, each MTP step keeps four tokens.
It accepts 3 proposals and rejects the fourth proposal.
Rejection sampling supplies 1 token.
This is a simple workload model. It does not change the MTP acceptance procedure.

For each profile, decode uses one context at the middle of its output sequence.
Its length is `input_tokens + output_tokens // 2`.

Calculate operator probabilities from request weights, layer counts, and invocation counts.
For each operator, normalize its probabilities.

Merge only cases with the same execution shapes.
Do not use boundary checks in the performance distribution.
Do numerical checks of MTP acceptance branches and other boundary conditions without performance weights.

## Performance gate

Use single-GPU dense B200 peaks: FP8 Tensor 4.5 PFLOPS, BF16 Tensor 2.25 PFLOPS, FP32 75 TFLOPS, HBM 7.7 TB/s.
The [NVIDIA B200 reference](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/_downloads/4b3d3302fa4fc6854b595ccff50b9932/RA11334001-DSPB200-ReferenceArch.pdf) identifies the platform.

Use this work model for each full operator:

```text
t_lower = max(
    necessary_HBM_bytes / HBM_peak,
    FP8_work / FP8_peak + BF16_work / BF16_peak,
    FP32_work / FP32_peak
)
eta(shape) = t_lower(shape) / measured_seconds(shape)
E_eta(operator) = sum(probability(shape | operator) * eta(shape))
PASS(operator) = accuracy_pass AND complete_coverage AND E_eta(operator) >= 0.60
```

Use [the work calculation](../../development/kernels/work.py) for the necessary computation and HBM bytes of each serving boundary.
Calculate HBM bytes from tensors that cross the serving boundary and necessary persistent-state access.
Count the necessary cold weight reads. Do not add HBM bytes for tensors between internal stages.

Include all accesses to temporary storage in the timed operation.
Acceptance selects state slots. It does not copy the full recurrent or convolution state.

FP8 and BF16 share Tensor Core resources. Do not calculate their resource limits as two different resources.
Padding and unnecessary copies cannot increase the lower bound.

Do not add measured launch time or empirical efficiency factors to the lower bound.
The work model has no parameter for a measured or calibrated latency.
Keep each shape's time and efficiency. Do not replace the expected efficiency with a ratio of total times.

Missing implementations, failed numerical checks, missing cases, or mixed source identities prevent acceptance.
If an efficiency is greater than 1.05, do a work-model check.

If a provider is missing or fails, stop the evaluation with an error.
Do not use a reference implementation in its place.

## Measurement

Get one idle B200 through the GPU wrapper.
Build the implementation before measurement.
Prepare the buffers before measurement.

Do the numerical checks before timing. Restore mutable state between all runs.
Use CUDA events before and after the full operation on the caller's stream.
Include all necessary launches in eager and CUDA Graph timing.

Measure eager execution and CUDA Graph replay in different runs.
Use two warmup runs and five measured runs.

Use cold weights for HBM roofline acceptance.
Before each timed invocation, run an L2 eviction with a buffer larger than 2 times the B200 L2 capacity.
Put this eviction before the start event on the caller's stream.
Do not include the eviction in the measured duration.

The weight working set of the full model is larger than L2.
The cold-weight policy is for this condition.
Record the eviction method, buffer size, and L2 capacity.
Do not use isolated runs with warm weights as HBM roofline acceptance evidence.
Warmup runs do not remove the cold-weight rule.

Record the median and spread. Save all measurements at a location external to the repository.
Save the failure records at the same location.
Include the source, binary, compiler, dependency, device, clock, and shape identities.

Use Nsight Systems to find launches with long execution times and GPU gaps.
After the numerical checks show correct results, use Nsight Compute on the selected launches.

Collect duration, Tensor Core throughput, HBM traffic, cache hit rates, occupancy, registers, spills, and warp stalls.
Use the label "unavailable" for counters that are not available.
If counter permissions are not sufficient, use this label.
Do not record missing counters as zero.

Do not use the profiling runs for acceptance timing.
After each run, stop the processes that you started.
