# Work state

Use `main` for work.
The [operator packages](../python/oh_my_vllm/operators) keep each family's interface, plan, workspace, and CUDA code together.
The [structure design](operators/design.md) defines their dependencies and ownership.
Keep shared resource mechanisms in `runtime` and shared computation in `primitives`.
The native builder is `development.native_build`. It builds seven native modules.

The worker uses explicit layer weights, named outputs, and operator plan keys.
Use the [execution rules](execution.md) for Eager execution, CUDA Graph replay, and resource release.
Rust and the worker control persistent state selection.
Operators write only to the state locations supplied by their callers.

Small FFN and GDN operations use dependent launches at the boundaries in the execution rules.
FFN prefill uses these dependencies for 65 to 1024 rows.
The [prefix cache](serving.md#prefix-cache) can use a checkpoint from a block that is not full.
Keep source blocks available until continuation results have no validation errors.

Use the [operator validation framework](../development/operators) before kernel changes.
It supplies fixed numerical fixtures, direct PyTorch references, and theoretical-latency ratio limits.
Usual numerical tests load repository fixtures without executing PyTorch.
Keep the [precision policy](operators/precision.md) and [full-model limits](operators/evaluation.md).
Do not automatically change limits after implementation changes.

FA uses BF16 attention operands and cache state. GDN keeps recurrent state in FP32.
FP8 projections use checkpoint E4M3 weights and FP32 block scales.
The [model contract](operators/model.md#intermediate-precision) defines the intermediate precision.

HTTP accounting uses serial kernel groups and the current precision rules.
Use the [performance calculation](performance.md#lower-bound-calculation) when fusion or execution boundaries change.
Operator ratio limits keep the approved maximum latency across accounting changes.

Keep long-output efficiency at more than 40%. Measure each configuration two or more times.
For each short-output configuration with a measured time of 5 seconds or more, the efficiency must be 40% or more.
Continue optimization of large prefill operations.

Run the applicable [test scope](testing.md) before delivery.
Also run all 24 [HTTP configurations](performance.md).
Keep measurement artifacts external to the repository.
Put host settings in ignored `LOCAL.md`.
