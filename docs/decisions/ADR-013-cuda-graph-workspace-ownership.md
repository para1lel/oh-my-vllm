# ADR-013: CUDA graph workspace ownership

Status: active. Date: 2026-10-12.

## Problem

A full-model DSpark collection failed with CUDA illegal memory access.
A synchronization diagnostic found the first failing operation in context injection.
The context graph used a BF16 projection of 29 rows, with 25600 features per row.

The pinned Torch 2.14.0 caches cuBLAS workspaces by handle and stream.
Graph reset removes cached workspace owners for its capture stream.
Other resident graphs can still use their addresses.

A pre-capture warmup can allocate that workspace in the default pool.
Later cache release and allocator cleanup can remove storage still needed for replay.
[PyTorch issue 193402](https://github.com/pytorch/pytorch/issues/193402) specifies this failure mechanism.
The pinned [graph reset](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cuda/CUDAGraph.cpp) and [workspace cache](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cuda/CublasHandlePool.cpp) contain the affected lifetime rules.

## Decision

Use one project capture boundary for all seven worker capture sites.
Synchronize previous GPU work, then clear the cached cuBLAS workspace owners before entering capture.
The first captured GEMM allocates a new workspace in its owning graph's private pool.
Keep workspace sizes, weights, GEMM configuration, and numerical tolerances unchanged.
Do not change installed Torch or CUDA provider code.

Graph execution must use an enabled native caching allocator without an explicit `max_split_size_mb`.
A finite split limit can release cached private-pool blocks while graph owners are still resident.
The `cudaMallocAsync` backend has different pool rules and lacks verification for this fix.
Disabled caching also bypasses graph-private storage and is rejected.
Validate effective runtime settings at worker startup, before the graph cache's capture fallback.
Validate again at the capture boundary.

Reject unsupported configurations without rewriting them.
Explicit `OH_MY_VLLM_ENFORCE_EAGER=1` keeps its diagnostic behavior with other allocator configurations.

The worker host thread executes captures and replays serially.
Persistent inputs, weights, and KV storage use storage separate from shared graph scratch pools.
Outputs are read or copied before another capture or replay in their family.
Different graph families keep separate pools. Graph eviction keeps its existing ownership and synchronization rules.

## Verification and limits

An external same-stream BF16 GEMM reproducer failed on its first survivor replay after unrelated graph reset.
The capture boundary completed twenty reset and allocator-pressure cycles with unchanged workspace budgets.

An external checkpoint DSpark context test completed 42 cases and 210 layer comparisons.
It kept 32 captured shapes, tested the original 29-row metadata, and used a 33-row compiled path without manual capture.
All comparisons passed `atol=0.03`, `rtol=0.03`, with maximum absolute error zero.
Untouched-row canaries passed exact comparisons. Workspace budgets stayed 32 MiB and 1 MiB.
Native allocator snapshots showed workspace-sized storage in the resident private pool after unrelated reset and cleanup.

These snapshots support the ownership mechanism. They do not directly identify an internal cuBLAS pointer.

A zero-workspace experiment failed the existing numerical tolerance and was rejected.
The successful diagnostic used synthetic target features with the draft checkpoint.
It did not execute the full target model, Rust scheduling, Markov sampling, or HTTP requests.
The repository regression uses the production context method and dimensions with random weights.
CPU tests verify synchronization, validation, error propagation, and unchanged budget APIs.

Measure all affected formal workloads again on committed source.
The fix adds work at capture boundaries. No performance claim follows from the diagnostic.
Original failed and cancelled collections keep their status and source identities.
