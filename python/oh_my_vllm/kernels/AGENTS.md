# CUDA operators

Use CUDA C++ and CUTLASS for this implementation of the [internal stages](specification.py).
The [runtime contracts](../../../docs/requirements.md) also allow CuTe DSL.
Use the [tile design](../../../docs/tile-execution.md) for execution and migration scope.
Use the checkpoint dimensions. Keep the original weight values and scales.

If the full operation becomes faster, fuse the small producer operations into their consumers.
Use full HTTP timing for performance acceptance.

Use the caller's stream. Allocate no memory during a graph replay.
Before a launch, do checks of shapes, types, strides, devices, and storage overlap.

Keep persistent writes in private slots until the target accepts the related tokens.
Accept tokens through state slot selection. Do not copy the full state for acceptance.
Keep the 784-token checkpoints and BF16 recurrent storage.

Use [the evaluation procedure](../../../docs/operators/evaluation.md).
If a provider is missing or fails, stop the evaluation with an error.
Do not use a reference implementation in its place.
