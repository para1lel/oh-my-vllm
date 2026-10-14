# Precision-path experiments and limits

No candidate on this page is a selected production change.
The [diagnostic summary](../evidence/precision-lessons/verification.json) binds sources, loaded modules, input snapshots, failed attempts, and raw results.
Use the [algorithm research](optimization.md), [affine diagnostics](affine-diagnostics.md), and [measurement method](../profiling/measurement.md).

## Input scope

An instrumented ordinary-model run stores inputs from GDN invocation indices `0/15/47`.
It uses token `42` repeated `32768` times and manually supplied cache tables.
The snapshots include the full initial state and sequence offsets.
They do not identify physical layer IDs, checkpoint byte hashes, or a prompt distribution.
Observer cost is part of the run. Its duration is not acceptance latency.

Some heads have weak decay. Mean decay alone does not bound earlier state contributions.
Use complete-state comparisons before removal of history or transfer terms.

## Finite history

A candidate runs the direct FP32 recurrence from a finite preceding window.
It uses zero state when the window does not include the sequence start.
The windows contain `128/512/2048` tokens before each segment.
All nine model-input comparisons fail against repository CUDA.
Final-state NRMSE is 11.22%–93.26%.
Longer windows in this collection do not correct the failures.

These comparisons are not independent long FP64 references.
They are sufficient to reject the candidate against the existing runtime contract.
The result does not permit decay truncation or a new tolerance.

## Direct FP32 prefix and matrix suffix

A hybrid uses a direct FP32 token prefix and an FP32 WY matrix suffix for each chunk.
The initial cancellation collection passes 24 cases. Its expanded collection passes 36 cases.
The tested cancellation starts at the first token.
Nine model-input comparisons also pass, but candidate latency is 21.13–24.05 ms against 8.46–16.07 ms for repository CUDA.

A later test sets beta to zero for the first eight tokens of each chunk.
All twelve delayed-cancellation comparisons fail their final-state checks.
Final-state NRMSE is 6.06%–69.85%, although output absolute-tolerance checks pass.
A fixed direct FP32 prefix does not prevent these final-state failures.
Test cancellation positions independently of ordinary random-input verification.

## Native FP64 panels

An eight-token WY candidate uses native FP64 matrix computation for state transfer.
All four delayed-cancellation cases pass an independent FP64 reference.
Three full model-input comparisons pass against repository CUDA.
Full candidate latency is 36.47–36.53 ms against 8.45–16.07 ms for the baseline.
Latency for another candidate that keeps state accumulator fragments is 39.33–39.60 ms.
The two candidates are not selected.

One diagnostic scan has 16.08% occupancy, 12.20% Tensor Core activity, and 0.49% DRAM activity.
Nsight Compute collects 18 replay passes with uncontrolled cache and clocks.
Its 34.89-ms duration is profiler diagnosis, not an acceptance timing sample.
Low DRAM activity does not show that more staging will improve latency.
Examine instruction-dependent stalls through source and device instructions before another schedule choice.

The original first build had a shared buffer that was too small. It was not executed.
The corrected build has a different source and module identity.
The first counter attempt selected no kernel. The two original failed records stay available.

## Ideal-arithmetic bounds and FP32 rounding

For ideal arithmetic, a state-row map is `A = lambda * (I - beta * k k^T)`.
Its spectral norm is `lambda * max(1, abs(1 - beta * ||k||^2))`.
This ideal-arithmetic bound does not prove the same bound for a rounded FP32 recurrence.

A CPU counterexample changes one initial-state value from `1` to its next FP32 value.
The initial difference is `1.1920929e-7`. One rounded state result changes by `1`.
The BF16 output difference is `0.0625`, above the proposed bound `1.0535587e-8`.
Independent exact rational arithmetic verifies each recorded FP32 rounding operation.
This is a CPU/libm/software proof, not a GPU execution or a production failure.

An error guard must include rounding error, all outputs, and final-state relative and absolute limits.
An ideal-arithmetic residual alone is not sufficient.
A fallback must restart from the unchanged initial state and replace all candidate outputs.
These conditions are requirements for a future guard, not a verified implementation.
