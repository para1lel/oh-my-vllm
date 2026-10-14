# CUDA candidate safety

Check storage and shape contracts before a new CUDA candidate runs.
Compilation and `can_implement` success do not prove memory safety for all shapes.
Keep experiments in external storage until full numerical and replay checks pass.

## Narrow scale-copy failure

An external transposed split-K experiment tested 8, 16, and 32-column tiles.
18 combinations at 1 and 4 rows completed sampled numerical, replay, canary, and timing checks.
The experiment failed with an illegal memory access in the 20-row case with an 8-column tile and 2 slices.
The candidate was removed from consideration. No experimental source entered production.

Independent source review found a shared-memory boundary defect in the 8 and 16-column variants.
The pinned [CUTLASS builder](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/include/cutlass/gemm/collective/builders/sm100_blockwise_umma_builder.inl#L178) uses 32 scale-copy threads.
Its [copy predicate](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/include/cutlass/gemm/collective/sm100_mma_warpspecialized_blockwise_scaling.hpp#L963) checks global scale boundaries.
It does not keep those writes in the smaller local tile's scale storage.
More valid global rows can cause incorrect writes to memory after that storage.

This defect agrees with the recorded shape boundary. It does not identify the exact fault instruction or prove that it is the only cause.
The experiment has no Compute Sanitizer fault-location evidence.
Partial numerical and canary results do not prove full memory safety.

The [production narrow entry](../../python/oh_my_vllm/kernels/cuda_backend/groupwise_transposed.cuh) uses a 32-column tile. At most 32 rows are permitted.
This observation does not add a production acceptance result.

## Candidate procedure

1. Check each operand's dtype, device, shape, stride, alignment, capacity, and alias contract.
2. Validate scalar choices before division, multiplication, or pointer arithmetic.
3. Check global, local-tile, stage, batch, and shared-memory boundaries independently.
4. Obtain independent source review before a new shape or layout runs.
5. Run Compute Sanitizer on each boundary shape before timing. Stop on the first failure.
6. Check changed inputs, non-default streams, replay, numerical outputs, and final state.
7. Collect paired timings only after these checks pass.

Use a new process for each initial shape. Do not share a CUDA context with a failed candidate.
Do not move an unexamined failed candidate to another GPU.
Stop exact owned processes and record device health after failure.
Keep failed sources, build identities, logs, and reviews in external evidence storage.

These checks decrease known risks. They do not prove that a candidate cannot fail.
Keep the [numerical contracts](../requirements/kernels.md) and [measurement method](../profiling/measurement.md) active.
