# Compilation and graph concepts

## Semantic IR and graphs

Semantic IR names mathematical operations, state effects, and provider contracts before implementation selection.
The [registry](../../python/oh_my_vllm/ir/core.py) keeps references and mutation schemas with those operations.
Compiled fullgraph units contain GPU computation. Host scheduling and transport have different boundaries.

A CUDA Graph replays work with persistent input and output storage.
Changed metadata must not leave stale values in that storage.
Warmup and capture must not change committed cache state.
[Capture transactions](../../python/oh_my_vllm/worker/prefill_graph.py)
restore persistent data before normal execution resumes. See the corresponding [decode transaction](../../python/oh_my_vllm/worker/decode_graph.py).

Compiler entries, generated metadata, executable owners, and graph pools have different lifetimes.
Their budgets belong in [compiler cache](../../python/oh_my_vllm/ir/cache.py),
[metadata ownership](../../python/oh_my_vllm/ir/metadata_cache.py), and
[graph cache](../../python/oh_my_vllm/worker/graph_cache.py).
Weak diagnostics must not extend executable lifetime.

Rewrites must keep observable BF16 casts, state ordering, exception behavior, and alias contracts.
Only proven temporary storage can be donated. Persistent caches cannot.
Independent references and replay tests verify these boundaries.

## CUDA streams and PDL

Streams can overlap independent work. Shared data still uses explicit dependency and lifetime rules.
Programmatic dependent launch can start a dependent block before the previous grid finishes.
The dependent block must wait before it reads unfinished input.

Read [native launch contracts](../../python/oh_my_vllm/kernels/cuda_backend/operators/common.cuh)
and [worker graph execution](../../python/oh_my_vllm/worker/decode_graph.py).
The caller stream owns native launches. Captured work keeps its storage owners through replay.

Graph capture can keep allocator storage that ordinary execution releases.
Use [environment configuration](../development.md) and [memory diagnostics](../profiling.md) for measurement and attribution.
Do not infer a memory plateau from bounded entry counts alone.
