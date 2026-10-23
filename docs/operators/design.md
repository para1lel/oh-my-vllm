# Operator structure

This document gives the current operator structure.
The [handoff](../handoff.md) gives the implementation status.
The [runtime contracts](../requirements.md), [model contract](model.md), and [GPU execution rules](../execution.md) control behavior.

## Purpose

Keep the interface, execution plan, temporary storage, and CUDA implementation for an operator in one package.
The model calls complete sublayers in dependency sequence.
An operator can change its internal kernels without changes to the model call site.
Operator performance must meet the fixed theoretical-latency ratio limits in the [microbench contract](microbench.md).

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

## Model output and MTP

FA, GDN, FFN, and vision interfaces supply complete sublayers.
Their internal stages follow the [model contract](model.md).
The model output and MTP interfaces have these responsibilities:

| Interface | Computation |
|---|---|
| `output.hidden` | Final normalization and hidden-state export without vocabulary projection. |
| `output.distribution` | Vocabulary projection, penalties, constraints, and the necessary probability distribution. |
| `output.sample` | Distribution computation and token selection. |
| `mtp.context` | MTP context preparation and its FA component. |
| `mtp.draft` | A complete MTP draft computation with the shared FA, FFN, and output components. |
| `mtp.accept` | Acceptance decisions, rejection sampling, and candidate state-slot selection. |

The output interfaces share the necessary final normalization computation.
Hidden export accepts its full input row count independently of the smaller vocabulary-projection row selection.
The distribution interface can execute without token selection.
These semantic interfaces can use fused physical execution.

The MTP package contains its composite fusion implementation and controls its storage layout.
Its component interfaces give compatible inputs, outputs, storage use, and lifetimes.
It does not infer compatibility from a different operator's private buffer indices or duplicate that operator's complete scratch layout.

MTP acceptance returns candidate results, not published state.
The worker first applies request stop conditions to the returned tokens.
The accepted count after stop handling then controls state selection through the existing state commit procedure.
The GPU candidate slot cannot replace this decision before stop handling.
Keep the RNG state and counter rules for each of the target, proposal, and acceptance domains.

## Native builds

Use one source-path mapping for the development builder and runtime identity checks.
Directory changes must also change include dependency discovery and source fingerprints.
Keep complete dependency hashes, immutable binary identities, and native library identity checks.
The inference loader does not import development tools or invoke a compiler.

Change compilation boundaries only when the implementation has a demonstrated reason.
Source packages, semantic interfaces, native libraries, and kernel launches are different boundaries.

## Change acceptance

Update the [stage inventory](../../python/oh_my_vllm/operators/specification.py) and [serving boundaries](fusion.md) with the related implementation changes.
Keep private buffers and kernel selection rules inside their operator packages.

Use the [test scope](../testing.md) and [numerical procedure](evaluation.md).
Use the [operator microbench design](microbench.md) for local numerical, state, and performance checks.
Extend full-model inputs when a change has an effect on MTP, vision, or long contexts.
Use Compute Sanitizer when device memory or synchronization behavior changes.
New tests follow the [contribution procedure](../../CONTRIBUTING.md).

Run all 24 [HTTP configurations](../performance.md) with the same workload settings for integration checks.
Apply the fixed operator performance limits without comparison to old builds or timings.
Do more measurements when uncertainty prevents a decision against a limit.
Check the other configurations when a new fusion makes selected workloads faster.
Keep timing artifacts external.
Update the [Qwen results](../deepseek-v41/qwen-results.md) after a complete matrix.
