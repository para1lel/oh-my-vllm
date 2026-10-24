# Operator structure

This document gives the current operator structure.
The [handoff](../handoff.md) gives the implementation status.
The [runtime contracts](../requirements.md), [adapter contract](../adapters.md), and [GPU execution rules](../execution.md) control behavior.

## Purpose

Keep the interface, execution plan, temporary storage, and CUDA implementation for an operator in one package.
The model calls complete sublayers in dependency sequence.
An operator can change its internal kernels without changes to the model call site.
Operator performance must meet the fixed theoretical-latency ratio limits in the [model acceptance scope](../testing.md#model-acceptance).

Use CUDA C++ and CUTLASS for this implementation.
The [runtime contracts](../requirements.md) also allow CuTe DSL.
CUTLASS, cuBLAS, CUB, and native CUDA libraries are permitted.
Use only project CUDA operators and these permitted libraries.
Do not use other external operator libraries, such as DeepGEMM or FlashInfer.
Python can allocate tensors and launch the CUDA implementations.

The design uses functions and types for each operator.
It does not use an operator base class, a backend registry, or automatic fusion through an operation graph.

## Packages and dependencies

The package is `python/oh_my_vllm/operators`:

```text
operators/
  runtime/
  primitives/
  fa/
    api.py
    plan.py
    workspace.py
    cuda/
      fa.cu
      input/
      attention/
  gdn/
  ffn/
  output/
  mtp/
  vision/
```

This tree shows the separation of responsibilities.
Split files when their size or responsibilities make separation necessary.
Group CUDA files by computation stage or algorithm in each operator package.

The worker calls operator interfaces.
Operator packages use `runtime` and `primitives`.
The MTP package can also use explicit internal component interfaces from FA, FFN, and model output.
The two shared packages do not import an operator package or the worker.

The `runtime` package supplies allocation, tensor views, byte copies, streams, CUDA Graph resources, and native module loading.
The `primitives` package supplies computation and tensor operations shared by two or more operator packages.
An indexed row gather belongs with these primitives.
Model-specific normalization and hidden-state writes belong with model output.

Move code into a shared package only when its users have the same computational or resource contract.
Keep fusion-specific code in its operator package.
The number of operator packages does not control the number of native libraries or kernel launches.

## Interfaces and ownership

Use the [project terms](../glossary.md) for execution plan and operator workspace.
Each operator family has its own plan type, input types, state views, and output fields.
The following call shows the interface form:

```python
result = fa.run(weights, workspace, inputs, batch_view, cache_view)
```

| Object | Responsibility |
|---|---|
| Prepared weights | Store the prepared tensors for one model layer. Two or more operator workspaces can keep references to these tensors. |
| Execution plan | Select the implementation and give storage, preparation, and execution conditions. |
| Operator workspace | Control the lifetime of temporary tensors and mutable native execution contexts for its operations. |
| Batch view | Supply the request and token metadata that the operator uses. |
| State view | Keep references to the persistent cache tensors and the assigned read and write slots. |
| Named outputs | Give the caller fixed views of the operator workspace output tensors. |

Pass layer weights explicitly at each call.
Do not change an operator object's weight field between layers.
Use temporary storage again across model layers that run in sequence when the execution conditions are compatible.
Do not allocate a complete operator workspace for each layer only to bind its weights.

The worker creates narrow batch and state views from its scheduling data.
Operators do not depend on worker request, batch, or cache classes.
Each operator prepares its implementation-specific metadata.
Views use existing tensors. They do not copy cache tensors.
Keep the host preparation and device metadata transfers.

The lifetime of persistent state storage is controlled independently of temporary operator storage.
Rust and the worker keep their existing logical identity and state selection responsibilities.
Operators read and write only the state locations supplied through their views.

## Planning and execution

Plan construction selects an implementation without GPU allocation or execution.
Equivalent inputs give equivalent plans.
The plan supplies the conditions for reuse and the necessary storage descriptions.
The operator prepares buffers and native contexts before Graph recording.
Preparation can use the applicable prepared weights.

The worker puts operator reuse signatures together for its recorded model operations.
A signature includes algorithm choices, capacities, layouts, and scalar values that have an effect on recorded launches.
A supported normalized bound can replace a variable scalar value in this signature.
Keep the applicable values in the signature even when buffer sizes do not change.
Resource identity checks apply independently of plan equivalence.

The worker updates metadata at fixed addresses on the caller's stream.
New metadata must be compatible with the recorded plan's conditions.
A change beyond those conditions causes selection or preparation of a compatible operator workspace before execution.
The worker does not copy operator dispatch thresholds or examine implementation fields to select preparation work.

Each operator gives its preparation and warmup operations.
There is no warmup procedure that executes all operators.
Sampling and MTP acceptance keep their existing recording and RNG behavior.
The [GPU execution rules](../execution.md) control the full Execute message before batch division.

## Output and Graph lifetimes

Named output views are available before recording.
A call returns the same prepared output views without implicit allocation or copying.
Their addresses stay fixed for the recorded operations.
Their contents can change at the next operator workspace reuse or Graph replay.
Keeping a result object does not keep the previous tensor values.

Queue dependent reads before a subsequent write on the ordered caller stream.
Save data explicitly when it must survive reuse.
This ordering does not add a device synchronization between model layers.

A plan can be shared when it has no mutable execution state.
A mutable native context belongs to its operator workspace and Graph lifetime.
Do not share that context between Graph lifetimes controlled independently, or Eager execution that is not compatible.

The model execution owner keeps all resources used by its recorded operations.
These resources include each layer's weights, persistent cache references, buffers, native owners, and resource leases.
Keeping only the final layer's weight binding is not sufficient.
The [Graph resource rules](../execution.md#graph-resources) control failed recording, closure order, and release.

## Serving boundaries

Operator interfaces define complete computation, not separate kernel launches.
Fuse small normalization, activation, quantization, or position work into producers or consumers when the complete operation benefits.
Small-row and prefill shapes can select different implementations.
Related operations can reuse prepared layouts and device metadata.

Keep semantic outputs, private state, and resource lifetimes explicit.
The model adapter applies stop conditions and selects accepted state before publication.
See the [Qwen boundaries](../models/qwen38/operators.md#serving-boundaries) and [DeepSeek contract](../models/deepseek_v41/model.md).

## Native builds

Use one source-path mapping for the development builder and runtime identity checks.
Directory changes must also change include dependency discovery and source fingerprints.
Keep complete dependency hashes, immutable binary identities, and native library identity checks.
The inference loader does not import development tools or invoke a compiler.
The native library count does not define public interfaces.

Change compilation boundaries only when the implementation has a demonstrated reason.
Source packages, semantic interfaces, native libraries, and kernel launches are different boundaries.

## Change acceptance

Update the applicable model contract with implementation changes.
Keep private buffers and kernel selection rules inside operator packages.
Use the shared [test scope](../testing.md) and the selected model's numerical and performance limits.
Run applicable fixed cases when speculation, media, or long contexts change.
Use Compute Sanitizer when device memory or synchronization behavior changes.

Keep measurements external. Update model results after a complete matrix.
New tests follow the [contribution procedure](../../CONTRIBUTING.md).
