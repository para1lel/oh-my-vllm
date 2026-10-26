# CUDA operators

Use CUDA C++ and CUTLASS, or CuTe DSL as permitted by the [runtime contracts](../../../docs/requirements.md).
Use the [GPU execution rules](../../../docs/execution.md) and [operator design](../../../docs/operators/design.md).
Keep model dimensions and checkpoint formats in the selected adapter's contract.
Keep original weight values and scales.

Fuse producer and consumer work when this improves the complete operation.
Interfaces must permit prepared device layouts, shared quantization, and beneficial PDL.
Measure complete operator boundaries and full inference time.

Use the caller's stream and the same operators for Eager and CUDA Graph execution.
Check shapes, types, strides, devices, and storage overlap before launch.
Keep speculative writes private until model-specific acceptance and stop handling complete.
Follow the adapter's checkpoint, precision, and restoration rules.

Use the selected model's frozen numerical references, fixtures, and limits in the [test scope](../../../docs/testing.md#model-acceptance).
Do not use inference outputs as expected values.
Keep correctness, memory safety, and performance evidence separate.
