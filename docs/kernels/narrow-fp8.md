# Narrow FP8 projection measurements

The native narrow projection changes the matrix orientation.
It keeps the FP32 block scales and BF16 output boundary equal.
Use the [owned CUDA path](../../python/oh_my_vllm/kernels/cuda_backend/groupwise_transposed.cuh) and [Python entry](../../python/oh_my_vllm/kernels/cuda_backend/groupwise.py#L211) <!-- source: def gemm_transposed( -->.
The [source-bound summary](../evidence/narrow-fp8/verification.json) records original files, loaded modules, samples, and verification limits.
Final-source framework acceptance is pending.

## Mathematical basis

For `Y = X W^T`, the equivalent product is `Y^T = W X^T`.
The second orientation puts the narrow dimension on the other matrix axis.
A column-major view of `Y^T` has the same storage as the row-major `Y`.
This identity does not use an output copy.

[CUTLASS's blockwise example](https://github.com/NVIDIA/cutlass/blob/main/examples/81_blackwell_gemm_blockwise/README.md) gives this orientation as a narrow-matrix option.
The FP32 scales still apply to each block product before accumulation.
Power-of-two scale conversion changes that contract.
The candidate keeps the FP32 scales and one final BF16 conversion.

## Paired measurements

The collection uses rows `1/4/20/32`, input width `5120`, and output widths `34816/16384/14336`.
It compares the production entry with the previous padded CUDA path.
Each of the twelve shapes has three rounds of twenty interleaved pairs.
Each sample replays a graph with ninety-six projection calls.
Eight resident weights supply a working set of approximately 0.6–1.4 GB, more than L2 capacity.

Full projection GPU latency decreases by 13.4%–18.4% across these shapes.
Each one-sided 95% confidence lower bound on time saved is positive.
CUDA Events include GPU padding and output work, but not Python dispatch, graph capture, or CPU allocation.
These operator results do not show a framework latency benefit.
The [measurement protocol](../profiling/measurement.md) uses different collections for profiler diagnosis and acceptance timing.

An independent CPU FP64 reference checks sixty-four output columns for the first resident weight of each shape.
Full-output comparisons also pass against the previous CUDA path.
The timing collection uses all eight weights. It does not supply an independent reference for each of those weights.

## Critical verification

[Graph replay tests](../../tests/numerical/test_fp8_transposed_replay.py) change activations, weights, and the two scale tensors after capture.
They check the caller stream, PDL on/off, independent GPU FP64 output samples, and equal outputs from the two entry paths.
Storage before and after the output stays equal.
The sampled columns include the two sides of a scale-block boundary and the last output column.
Fault tests reject aliasing, misalignment, incorrect output shape, and incorrect scale stride before launch.
Two replay cases pass Compute Sanitizer with zero errors, with PDL on and off.

The full diagnostic operator matrix passes all 539 cases without interference.
Its uncommitted-source result keeps `passed=false` and `selected_passed=true`.
The replay tests cover narrow output shapes that the original formal matrix does not contain.
Capacity, service, one-hour, and the fifteen-row framework matrix still await final-source verification.

## Lessons

The ordinary 32768-input/4096-output batch-one framework diagnostic has two warmups and five measurements.
Its prefill and decode ratios are 4.461 and 2.203. The spreads pass, but the latency checks fail.
The result keeps `diagnostic=true` and `passed=false`. It does not replace final-source acceptance.

Measure orientation choices through graph replay of the production entry, with GPU padding and output work.
Keep a weight working set more than L2 capacity when the inference workload reads many different weights.
Compare changed-input replay with an independent reference and check scale-block boundaries.
Bind loaded module hashes to their source and header hashes.
An operator timing improvement does not replace the framework gate.
