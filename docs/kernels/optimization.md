# Algorithm research and measured lessons

The framework latency gate is pending. No external benchmark shows the best algorithm for this project's contracts.
Use the [measurement method](../profiling/measurement.md), [numerical contracts](../requirements/kernels.md), and [diagnostic evidence](../evidence/optimization-observations/verification.json).
Research snapshots and temporary probes stay in external storage. Do not install external Python operator providers.

## Gated DeltaNet

The recurrence has a decayed state and a rank-one correction from the recalled value.
A token scan has sequential state dependence. More threads do not remove that dependence.
The [Gated DeltaNet paper](https://arxiv.org/html/2412.06464v3) gives an extended WY representation for chunk-parallel matrix computation.
[FLA's WY source](https://github.com/fla-org/flash-linear-attention/blob/1fe52b5d05c53601885610fab3d39157647d836d/fla/ops/gated_delta_rule/wy_fast.py) is a concrete algorithm reference.
Repository [chunk CUDA](../../python/oh_my_vllm/kernels/cuda_backend/operators/gdn_chunk.cuh) and [precision scan](../../python/oh_my_vllm/kernels/cuda_backend/operators/gdn_precision.cuh) supply the current implementation.

[FlashQLA](https://github.com/QwenLM/FlashQLA/tree/da06429d54b0f577de0a638f451ac8f0b395e0ac) supplies Blackwell schedules and context-parallel state transfer.
Its published timings compare different algorithms on specified inputs. They are not this project's measurements or numerical admission.
Examine matrix staging and dependency overlap. Check intermediate precision, cancellation, state transfer, and BF16 boundaries before a CUDA port.

Decay alone does not show that removal of the initial state meets relative-error tolerances.
With beta and values zero, the exact final state is `exp(sum_decay) * initial_state`.
A zero-seeded truncation returns zero. Relative error is 100% if the reference norm is more than the comparison floor.
This holds even when the absolute state is small.

FlashQLA's [decay-dependent context path](https://github.com/QwenLM/FlashQLA/blob/da06429d54b0f577de0a638f451ac8f0b395e0ac/flash_qla/ops/gated_delta_rule/chunk/blackwell/cp_fwd.py) can remove initial-state correction after decay truncation.
Zero decay keeps the full transfer correction. Treat truncation as an unverified numerical change, not a scheduling-only optimization.

## Block-scaled FP8

This project keeps checkpoint and activation FP32 block scales.
[CUTLASS's Blackwell blockwise example](https://github.com/NVIDIA/cutlass/blob/main/examples/81_blackwell_gemm_blockwise/README.md) supplies software scaling of FP32 accumulators.
It also gives transposed operand selection for narrow matrices.
These are useful native CUDA directions for the [owned GEMM implementation](../../python/oh_my_vllm/kernels/cuda_backend/groupwise_fp8.cu).
Measure scale work, conversion, and the final epilogue with the matrix multiplication.

[DeepGEMM's SM100 interface](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/README.md) uses packed UE8M0 scales.
Its power-of-two scale conversion is not equal to arbitrary FP32 scaling.
Do not copy that numerical path or change checkpoint weights. Persistent scheduling and data staging are separate research directions.

## Attention

[FlashAttention-4](https://arxiv.org/html/2603.05451v1) examines Blackwell's different scaling of Tensor Core, shared-memory, and exponential throughput.
Its pipeline design connects these resources instead of assuming matrix throughput alone controls latency.
Use that resource model to examine [Torch SDPA prefill](../../python/oh_my_vllm/kernels/ragged_attention.py) and [repository CUDA decode](../../python/oh_my_vllm/kernels/cuda_backend/operators/decode_attention.cuh).
Approximate exponentials or rescale elimination must have separate numerical proof. The paper's results do not show compliance with project rounding boundaries.

## Local experiments

The [diagnostic summary](../evidence/optimization-observations/verification.json) binds inputs, sources, raw hashes, samples, and limitations.
These experiments did not select a new production implementation.
Later [affine-scan diagnostics](affine-diagnostics.md) record matrix-layout benefits, complete emulation costs, and final-state cancellation failures.
Further [precision experiments](precision-lessons.md) reject finite history, fixed direct FP32 prefixes, and slow native FP64 panels.
The [narrow FP8 measurements](narrow-fp8.md) show a complete-operation benefit with unchanged block scales and output rounding.

### Native FP64 emulation

An isolated cuBLAS handle tested seven matrix shapes in three modes, with finite random FP64 inputs and a CPU FP64 reference.
All 21 comparisons passed their probe tolerance. This does not show compliance for complete GDN behavior or extreme inputs.
Eager emulation took approximately 108–1050 microseconds, compared with approximately 3–26 microseconds for native mode.
The sampled performant-mode timeline used the native FP64 kernel.
The other shapes lack that trace proof.

The [cuBLAS emulation guide](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html#fixed-point) gives mantissa selection and workspace constraints.
An unchanged output sentinel does not identify the executed route.
For the tested small matrices, the complete emulation chain is slower and was not selected.
This result does not reject emulation for larger matrices.

### Three-layer wavefront

A diagnostic compared whole-layer execution, serial token slices, and parallel slices through three GDN model layers.
Thirty interleaved samples per configuration gave 84.250 ms for whole-layer execution and 79.456 ms for parallel slices at 32768 tokens.
This is a 5.69% local timing improvement, not a complete framework measurement or the formal paired protocol.

Serial and parallel slices had the same recorded error summary against whole-layer execution.
The hidden-output NRMSE was 1.790%, and the last convolution state was not bitwise equal.
This does not identify a concurrency race or a formal GDN-state failure.
Complete equivalence evidence is not available. The candidate was not selected.
Make serial slices equivalent before testing their concurrent schedule.

### Current hardware observations

The scan fixture has low Tensor Core activity, low occupancy, and no register spills.
The FP8 fixture has the recorded Tensor Core and FMA activity.
The [counter record](../profiling/measurement.md#reproduced-observations) gives units, shapes, and scope.
Low occupancy does not show that a smaller block improves latency.
Record model-input numerical branch frequencies before tuning a branch-specific path.

## Next experiments

1. Record model-input route frequencies and locate scan dependency stalls in source and device instructions.
2. Compare a dedicated precision schedule with the current complete operation. Keep recurrence, reduction order, and state boundaries equal.
3. Examine FP32 scale accumulation and narrow-matrix scheduling without UE8M0 conversion.
4. Examine WY/state-transfer algorithms with cancellation, nonzero initial states, and zero-decay cases before integration.

Use the formal numerical matrix and applicable replay tests before selection.
Repeat framework measurements after selection. Keep the equations, tolerances, and two-times gate equal.
