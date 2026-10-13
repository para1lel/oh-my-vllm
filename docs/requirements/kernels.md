# Operator and IR contracts

## REQ-KERNEL-001: Native dependency boundary

Torch standard operations and repository CUDA supply GPU computation.
Native CUDA libraries, such as CUTLASS, CuTe, and CUB, are permitted.
Repository Python bindings can launch owned CUDA. External Python operator libraries are not permitted.

Remove obsolete comparison implementations, dependencies, documents, tests, and evidence from the current tree. Git history keeps past results.
Third-party unused examples do not authorize project imports or installations.

Use the pinned TileFoundry fork for development. Rebase it to compatible current upstream and validate the CUDA harness.
Discuss substantial fork design changes before implementation. Keep temporary tuning in external storage.

## REQ-KERNEL-002: CUDA and PTX

Use owned CUDA C++ kernels with optional inline PTX for B200.
Apply independent accuracy, framework, context, and model service gates.

Split the owned CUDA backend by function. Each `.cu`, `.cuh`, and `.py` file must have at most 800 lines.
Use pinned clang-format 23.1.3 and the pre-commit hook. Keep include dependencies in order.
Connect source and headers to build, load, and formal evidence identities.

Statically derive the largest legal invocation of each implementation path in the fifteen workloads.
Deduplicate identical configurations. Separate phases with different implementation paths.
Do not independently maximize incompatible dimensions or use dynamic tracing for formal case selection.
Keep independent numerical, rounding, and state verification for all legal paths, with supplemental DSpark cases.

Compare fusion candidates with the repository CUDA baseline through at least three rounds of twenty interleaved pairs.
Record medians and confidence bounds. Equivalent chains include necessary copies and reductions.
Use operator timings for diagnosis. Select fusion by measured framework benefit.
The framework two-times lower-bound gate is the performance acceptance criterion.

Keep formal cases, the harness, and summarized evidence in the repository.
Keep raw traces and temporary tuning in external storage. Profile separately from acceptance timing.
Record unavailable counters. Keep the gates active. Use [kernel procedures](../kernels.md).

## REQ-IR-001: Semantic IR

Control semantics and provider selection of project CUDA operations in the Qwen target, MTP, and DSpark paths.
Ordinary Torch operations, embedding, reshapes, and `F.linear` are not in this operator inventory.
Each operation has a Torch reference, shape/dtype contract, mutation schema, and statically selected providers.
Provider selection uses phase, shape, dtype, layout, and device metadata.
Provider failure is an error. The reference is an explicit debug choice.

Compile prefill, target decode, MTP draft, and four-step proposal with `torch.compile(fullgraph=True)`.
Compile DSpark target features, context injection, backbone, Markov step, and greedy proposal with the same fullgraph contract.
Keep semantic nodes until provider lowering. Keep host planning, allocation, scheduling, and ZMQ separate from these units.
Manual CUDA Graphs contain compiled units. Disable compiler-managed graphs.

Keep ordered persistent writes, capture restoration, and memory guards.
Activation donation must have proof for the named temporary. Persistent caches cannot be donated.
Graph rewrites must have equivalence tests and an exception-aware call-site inventory.

Acceptance includes existing accuracy tests, six ordinary/MTP4 context cases, formal operator cases, and fifteen phase-latency rows.
Keep DSpark service and supplemental operator cases active.
A target-model test must exercise all four compiled units and graph replay with changed metadata.
There is no new compile-speedup percentage gate.
