# Operator microbench design

This document gives the numerical and performance acceptance rules for operator development.
The [case manifest](../../development/operators/cases.json) fixes the measured shapes and modes.
The [limits](../../development/operators/limits.json) bind numerical and performance acceptance to those cases.

## Scope and boundaries

Use complete production operations as the primary boundaries:

| Family | Boundaries |
|---|---|
| FA, GDN, FFN | Complete sublayer |
| Output | `hidden`, `distribution`, `sample` |
| MTP | `context`, one `draft`, the four-draft chain, `accept` |
| Vision | Input plus first block, an internal block, last block plus merger |

Each boundary has at most five fixed numerical cases and five fixed performance cases.
A case gives shape, weights, inputs, state, execution mode, and sampling settings.
Choose shapes that occur in inference.
Do not expand these cases into a Cartesian product of flags, seeds, or layer weights.
FA variants used only by MTP are checked through the MTP boundaries.
GDN recurrence and prefill are checked through complete GDN cases.

Do not break production fusion or export extra intermediate tensors for measurement.
Use profiling to find internal costs after a complete-operation failure.
The initial suite does not add a benchmark entry for each CUDA kernel.
Keep full-model and HTTP checks for composition, service behavior, and complete inference cost.

## Numerical reference and fixtures

Use direct PyTorch formulas as the numerical reference.
Use FP64 arithmetic after exact conversion of the fixed inputs and original checkpoint weights.
The reference implements the mathematical operation, masks, positions, state transitions, and sampling rules from the [model contract](model.md).
It must not call production kernels or reproduce their tiling and fusion algorithms.
It must not use old-kernel outputs as expected values.
Use tiled attention only to limit reference memory.

Store the fixture inputs, initial state, metadata, and complete PyTorch expected outputs in the repository.
Use `development/operators/fixtures/<case_id>/manifest.json` and losslessly compressed NumPy shards in that directory.
Keep each shard at most 32 MiB.
Do not copy checkpoint weights into fixtures.
Bind external checkpoint weights by tensor names and the existing checkpoint identity.

The manifest records tensor shapes/dtypes, case identity, input hashes, expected-output hashes, reference source hash, PyTorch version, and precision.
Use public fixed inputs or deterministic synthetic inputs with representative magnitudes.
Do not capture private user requests.
Use the stored arrays as fixture inputs.
A seed alone does not replace the stored reference output.

Usual tests load fixtures and run the candidate only.
They do not import PyTorch or generate reference outputs.
A missing or changed fixture causes an error.
Use PyTorch only for explicit initial generation or an approved fixture update.
Stage changed fixtures with their manifests and reference changes for review.
Do not regenerate or overwrite expected arrays during a test.

Compare all elements of named outputs and assigned state writes.
These include BF16 KV, GDN recurrent/convolution checkpoints, and hidden exports.
Use per-sequence, per-field normalized L2 error and maximum absolute error.
Near-zero references also use an absolute-error condition.
Reject nonfinite values that the contract does not permit.

Check integer indices, masks, slot selection, shapes, and untouched guard storage exactly.
The reference output includes only assigned state writes, without a second copy of the complete read-only cache.

Store numerical limits in versioned `development/operators/limits.json`.
Calibrate limits against the direct PyTorch outputs before kernel changes.
Use the reviewed [field precision policy](precision.md) for each datatype and mathematical operation.
Check simple exact cases when you examine the reference formulas.

Do not copy the full-model limits into all operator fields.
Do not choose limits only by multiplying current-kernel errors.
Missing or unreviewed limits prevent acceptance.
A candidate cannot widen limits or update fixtures to make itself pass.

The FFN and GDN Graph cases use zero inputs and then fixture inputs during numerical checks.
FFN outputs must be zero when its inputs are zero.
Outputs for fixture inputs must meet the same numerical limits after a replay with zero inputs.
This check finds values from previous replays. It uses the same shapes and reference fixtures.

Sampling fixtures include complete distributions and seed/counter expectations calculated without production code.
Use fixed random thresholds away from numerically ambiguous CDF boundaries for exact token checks.
Check greedy ties, masks, rejection residuals, bonus selection, and RNG counters through the fixed cases.

PyTorch generates expected numerical arrays. Usual tests compare arrays with NumPy.
Integer RNG mapping uses a deterministic helper that does not call production code.
Distribution-only execution must not select tokens or advance RNG state.
A worker behavior test checks stop truncation before effective state selection.

## Theoretical latency and performance limits

Use the same nominal B200 resource rates and resource-time formula as [HTTP accounting](../performance.md#nominal-resources).
Use different version identifiers for operator and HTTP accounting.
Use this formula for each fixed performance case:

```text
t_lower = max(
    necessary_HBM_bytes / HBM_rate,
    FP8_work / FP8_rate + BF16_work / BF16_rate + TF32_work / TF32_rate,
    FP32_work / FP32_rate,
    FP64_work / FP64_rate,
)
ratio = measured_GPU_seconds / t_lower
pass: ratio <= case.max_ratio
```

The denominator comes from the case's semantic work and fixed precision assumptions.
It does not come from the candidate's kernel count, temporary buffers, padding, duplicated work, or measured latency.
Count shared data once per logical pass.
Keep omitted costs explicit, as in HTTP accounting.
Do not add a measured launch constant to the theoretical lower bound.
A zero denominator is an accounting error, not a passing case.

Use a declared production working set and cache policy.
Do not count cache-resident inputs and weights as mandatory HBM reads.
For a multi-layer pass, subtract the conservative cache capacity from unique read bytes before giving HBM credit.
Use zero HBM credit when the complete read working set can be resident.
Record the capacity source and assumptions in the accounting identity.
The result is a conservative nominal-resource estimate, not a proof of minimum physical latency.

For FA/GDN/FFN and internal vision blocks, measure a fixed sequence of applicable layer weights with the same shape.
For MTP and output, use their production weights and the declared cache policy.
Use the same sequence for calibration and subsequent tests.
This prevents a single hot layer from representing a streaming model workload.

Calibrate each maximum ratio from the current correct implementation.
Use five new measurement sessions with at least 30 samples per case in each session.
Compute a 95% upper confidence limit for each session's median ratio by fixed-seed bootstrap with 10000 resamples.
The raw proposal is the largest session limit, rounded upward to four decimal places.

For the approved `max_ratio`, multiply that proposal by `1.02` and round upward to four decimal places.
Record the raw proposal and the fixed 2% allowance in the limit provenance.
This allowance does not change after kernel changes.

Examine these limits with the case identity, accounting version, execution mode, device specification, and calibration-source identity.
Freeze the approved limits.
Keep calibration reports external and record their hashes in the limit provenance.
Usual performance tests compare their ratio with this fixed limit.
They do not load old timings, execute an old implementation, or compare baseline and candidate versions.
A candidate cannot recalibrate its own limit.

These limits control the accepted latency budget.
A small slowdown that stays below the frozen limit can pass.
The suite must not claim to detect all performance decreases.

## Timing and decisions

Use one idle B200 through the GPU wrapper.
Use the caller stream and the production Eager/Graph policy.
Vision runs Eager. Decode uses the applicable recorded path.
Measure GPU time with CUDA events on that stream.

Report preparation, recording, host submission, and temporary-storage size independently.
Do not compare host time with a GPU resource lower bound.

Exclude reference work, downloads, allocation, and fixture restoration from GPU execution timing.
Restore state and RNG at stable addresses before each independent sample.
Warmup uses isolated state.
Recording must not consume serving RNG state.
Execute calls again in one timed interval only if their inputs stay unchanged or the fixture gives a valid sequence.
If these conditions do not apply, time one restored call per sample.

Do not subtract reset time from a combined measurement.

Use at least 30 samples after warmup.
For repeat-safe calls, target at least 20 ms in each sample.
Record the GPU UUID, driver, clocks, power/thermal state, source, native binaries, checkpoint, and case identities.

Reject concurrent GPU work or measurement settings that are not compatible.

Compute the median ratio and its 95% confidence interval.
Pass only when the interval's upper end is at most `max_ratio`.
Fail when its lower end is greater than `max_ratio`.
If the interval crosses the limit, collect a total of 120 samples and calculate it again using all samples.
If uncertainty continues, report an unresolved result with a nonzero exit status.

Missing primary cases also prevent acceptance.
No average across cases can hide a failed case.

Keep timing samples and test reports external.
Keep fixed numerical fixtures, reference code, case definitions, accounting rules, and acceptance limits in version control.
