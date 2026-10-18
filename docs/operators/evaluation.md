# Operator evaluation

Use the [model contracts](model.md) and [fusion plan](fusion.md).
The [workload model](../../development/kernels/workload.py) supplies six request profiles.
Use its 7 serving boundaries for acceptance. Do not apply the gate to the 18 internal stages.

Native providers supply the serving boundaries. Full workload acceptance is necessary.
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

## Numerical scope

Use the [fixed numerical cases](../../development/kernels/workload.py#L70)<!-- source: def numerical_cases(): --> for numerical acceptance.
Run each case in eager mode and a CUDA Graph before timing.
Do not run CPU FP64 references for large modeled cases.
A small row count alone does not limit attention reference work. The context length also controls this work.

| Operator | Numerical cases |
|---|---|
| FFN | Rows 1, 4, 5, 20, and 65 |
| GDN | Default rows 1, 4, 5, and 20. One embedding row. 65 rows with two media rows |
| GDN boundaries | Default 65 rows at positions 0 and 770. Empty input guards |
| FA | Rows 1, 4, 5, 20, and 33 with context 256. Five rows with context 786 |
| Sampling | All modeled cases: rows 1, 4, 5, and 20, with the full vocabulary |
| MTP context | Four rows in one request. 16 rows in four requests |
| MTP draft | Rows 1 and 4 with context 256. The `full` and `query_only` variants |
| Vision | Two frame groups of 196 patches. `input`, `default`, and `merge` variants |

The shifted GDN case has checkpoint rows 13 and 64. It crosses the 784-token boundary.
FA checks include the prefill path and a cache page boundary.
Vision checks include changes to one frame group and unchanged outputs in the other group.

The 65-row FFN case uses the prefill projection path.

Reports use version 4 and the `small_shapes_v2` numerical policy.
The `numerical_evidence` record binds checks to the report's sources, binaries, and checkpoint identities.
The audit must include each listed shape and the two execution modes.
Each modeled timing case refers to its matching numerical case or records `status: not_run`.
Unchecked cases must not record a numerical pass.
These checks do not prove numerical accuracy for larger shapes or elements that the provider's stated scope does not include.

## Performance calculation

Operator tests have no minimum efficiency requirement.
Before each delivery, measure all serving operators in eager and CUDA Graph modes.
Update the operator table in [README](../../README.md) and its Chinese translation.
Numerical checks, complete timing coverage, and source identity checks are necessary.

Use the [peak calibration procedure](../calibration.md) for the selected GPU UUID.
Use measured HBM bandwidth and measured FP8, BF16, TF32, FP32, and FP64 throughput.
Do not use nominal specifications or another GPU's calibration.
Use the [intermediate precision contract](model.md#intermediate-precision).
Apply the FP32 and FP64 resources separately.

Use this work model for each full operator:

```text
t_lower = max(
    necessary_HBM_bytes / measured_HBM_peak,
    FP8_work / measured_FP8_peak + BF16_work / measured_BF16_peak + TF32_work / measured_TF32_peak,
    FP32_work / measured_FP32_peak,
    FP64_work / measured_FP64_peak
)
eta(shape) = t_lower(shape) / measured_seconds(shape)
E_eta(operator) = sum(probability(shape | operator) * eta(shape))
PASS(operator) = small_checks_pass AND complete_performance_coverage AND identities_match
```

These measured peaks give the calibration workload and environment.
The latency lower bound is a roofline estimate based on these measurements.
It is not a physical limit. Efficiency can be more than 100% if an operator has a higher throughput than a calibration rate.
Keep such values in the report for analysis.

Use [the work calculation](../../development/kernels/work.py) for the necessary computation and HBM bytes of each serving boundary.
Calculate HBM bytes from tensors that cross the serving boundary and necessary persistent-state access.
Count the necessary cold weight reads. Do not add HBM bytes for tensors between internal stages.

Include all accesses to temporary storage in the timed operation.
Acceptance selects state slots. It does not copy the full recurrent or convolution state.

FP8, BF16, and TF32 share Tensor Core resources. Do not calculate their limits as separate resources.
Padding and unnecessary copies cannot increase the lower bound.

Do not add measured launch time or empirical efficiency factors to the lower bound.
Use calibration only for resource rates. Do not fit the bound to operator latency.
Keep each shape's time and efficiency. Do not replace the expected efficiency with a ratio of total times.

Missing implementations, failed numerical checks, missing cases, or mixed source identities prevent acceptance.
Reports include the maximum normalized tolerance ratio. The ratio must be at most 1. The necessary guard checks must pass.
Examine efficiencies above 100% for calibration or work-model differences.

If a provider is missing or fails, stop the evaluation with an error.
Do not use a reference implementation in its place.

## Measurement

Get one idle B200 through the GPU wrapper.
Build the implementation before measurement.
Prepare the buffers before measurement.

Do the fixed small numerical checks before timing. Restore mutable state between all runs.
Measure each modeled shape. Large modeled cases supply performance evidence only.
Use CUDA events before and after the full operation on the caller's stream.
Include all necessary launches in eager and CUDA Graph timing.

Measure eager execution and CUDA Graph replay in different runs.
Use two warmup runs and five measured runs.

Use cold weights for HBM roofline measurement.
Before each timed invocation, run an L2 eviction with a buffer larger than 2 times the B200 L2 capacity.
Put this eviction before the start event on the caller's stream.
Do not include the eviction in the measured duration.

The weight working set of the full model is larger than L2.
The cold-weight policy is for this condition.
Record the eviction method, buffer size, and L2 capacity.
Do not use isolated runs with warm weights as HBM roofline measurement evidence.
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
