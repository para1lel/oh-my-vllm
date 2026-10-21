# Operator performance accounting

The [microbench contract](microbench.md) defines fixed cases, calibration, timing, and acceptance.
This document defines the `operator-reference-v2` resource policy.
The [calculation](../../development/operators/accounting.py) uses the HTTP resource rates and resource-time formula.
It has a separate version and does not change HTTP accounting.

## Memory credit

Count unique reads for the complete fixed operation pass.
Count shared input arrays once across its layers.
FA, GDN, and output inputs include BF16 values and FP32 residuals.
FFN value and residual inputs use BF16.
Count each layer's different weights and initial state once.
Count matrix scales and normalization vectors at their stored precision.

Use the serial kernel groups from [HTTP accounting](../performance.md#lower-bound-calculation).
Each group first uses its maximum resource time.
Give the complete pass one 128 MiB cache allowance:

```text
raw = sum(group_resource_maximum)
compute_floor = sum(group_compute_maximum)
reference = max(compute_floor, raw - 128MiB / HBM_rate)
```

The cache budget cannot decrease a group's time to less than its compute floor.
Do not subtract a cache allowance from each kernel.
Count shared input bytes only in the first group that reads them.
The runner reads the device L2 capacity through the CUDA driver.
Reject devices with a larger L2 capacity or a different GPU model.

Do not give HBM credit for intermediate values, scratch, padding, copies, or repeated reads.
The policy also omits all output and persistent-state writes.
These omissions can make the lower bound smaller.
They cannot increase credit when the candidate adds work.
Numerical tests still check required state writes.

The four-draft chain shares its layer weights across all stages.
Count these weights and the original cache history once for the complete chain.
New KV values inside the chain have no HBM credit.
The chain also has no embedding-read credit because generated tokens can share embedding rows.
Other fixed embedding inputs use their distinct token rows.

## Arithmetic credit

Count two operations per matrix multiply-add.
All fixed attention cases use BF16 QK and PV operands.
Each prior token has 4096 cache bytes for BF16 K/V.
These assumptions stay fixed when candidate dispatch changes.
The candidate cannot select a larger theoretical workload.

Text projection, normalization, FFN, and GDN recurrence counts use the fixed dimensions and formulas in the calculation.
GDN uses the fixed precision and dispatch rules in [HTTP accounting](../performance.md#arithmetic).
Private checkpoint count does not increase memory credit.

Output distribution and sampling include normalization and the full vocabulary projection.
Sampling penalties, sorting, integer selection, and random generation have no arithmetic credit.
Their costs stay inside the timer.

Acceptance counts attempted FP32 probability ratios.
For probability targets, it also counts residual subtraction, mass summation, and the FP64 threshold product.
Greedy integer targets use only the ratio count.
A very small lower bound can therefore produce a large latency ratio.
Do not add an empirical launch-time floor.

Vision counts patch, attention, MLP, and merger matrix products at BF16 precision.
Its scalar count includes normalization, residual addition, rotary products, and attention scaling and sums.
The input boundary also counts pixel normalization.
Transcendental functions, quantization, conversion, and omitted scalar terms have no credit.

## Input and measurement identity

Performance input generation uses NumPy and the committed numerical fixtures.
It does not execute the PyTorch reference.
Larger cases use the fixed shapes, page tables, slots, and ordered layer bindings.
The generated input and initial-state arrays have a combined SHA256 identity.
Reports also bind the prototype manifest, generator, NumPy version, and checkpoint.

The accounting identity binds operator and imported HTTP calculation sources.
It includes kernel group definitions, rates, and cache policy.
Approved limits also bind the complete resource totals and lower-bound value.
Approved limits bind this identity, the timing-policy identity, the case, generated inputs, and compatible device settings.
Ordinary acceptance does not load calibration timings.
When accounting changes, keep each approved maximum latency during ratio conversion:

```text
new_max_ratio = old_max_ratio * old_reference / new_reference
```

Keep the previous accounting identity, resource values, and ratio in migration provenance.
This conversion must not increase the approved maximum latency.

Restore state and RNG from prepared device copies before each independent sample.
Complete each reset before the start event.
Sampling, individual drafts, and acceptance use one restored invocation when RNG changes.
Repeat-safe operations use enough invocations for at least 20 ms per sample.

For repeat-safe operations, discarded pilots select the repeat count with a 25 ms target.
Keep that count fixed throughout the sample collection.
If any repeat-safe interval is shorter than 20 ms, discard the complete collection.
Then increase the repeat count.

An uncertain 30-sample result extends to 120 samples with the same count.
Start a new complete 120-sample collection after a short repeat-safe extension interval.
Reports record these restarts.

The initial timing policy is `operator-timing-v1`.
Eager timing records events around the production calls on the caller stream.
Graph timing captures the same production function into a separate child graph.
A measurement graph contains serial child copies between two timing-event nodes.
The runner uploads this graph before measurement and keeps its resource owners until destruction.

Capture must leave all input and state values unchanged.
Separate host timing uses the original production Graph replay after restoration.
Measurement-graph construction time is reported separately.
The policy identity binds the timing code and Graph construction code.

Reports separate preparation, recording, host submission, and GPU time.
Temporary-storage size is the device-memory difference across binding, warmup, and Graph creation.
Reset-copy storage is reported separately.
Record clocks, power, temperature, driver, and GPU UUID.
Reject another compute process on the selected GPU and changes to measurement settings.

The calibration command starts five different measurement processes.
Each process measures all fixed primary cases.
Duplicate sessions, missing cases, or different input/accounting identities prevent a limit proposal.
Keep all reports external to the repository.
