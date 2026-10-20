# CUDA operators

Use CUDA C++ and CUTLASS for this implementation of the [internal stages](specification.py).
The [runtime contracts](../../../docs/requirements.md) also allow CuTe DSL.
Use the [GPU execution rules](../../../docs/execution.md).
Use the checkpoint dimensions. Keep the original weight values and scales.

If the full operation becomes faster, fuse the small producer operations into their consumers.
Use operator microbench checks and full HTTP timing for performance measurement.

Use the caller's stream. Use the same operators for Eager and CUDA Graph execution.
Before a launch, do checks of shapes, types, strides, devices, and storage overlap.

Keep persistent writes in private slots until the target accepts the related tokens.
Accept tokens through state slot selection. Do not copy the full state for acceptance.
Keep the 784-token checkpoints and BF16 recurrent storage.

Use [the evaluation procedure](../../../docs/operators/evaluation.md).
Use the PyTorch FP64 full-model logits reference.
Use the [microbench contract](../../../docs/operators/microbench.md) for fixed operator cases, repository fixtures, and acceptance limits.
