# DeepInfer Operator and Serving Baselines

Audit date: 2026-10-26. Scope: public source review of DeepInfer, FlashMLA, DeepGEMM, DeepEP, and TileKernels. This report separates upstream facts from recommendations for `oh-my-vllm`. It does not add a dependency or claim runtime support.

## Public baseline

### DeepInfer facts

DeepInfer is an offline inference framework for DeepSeek-V4.1-Flash. Its README lists optimized model, sampler, and kernel runtime. It also lists attention DP or balanced CP, MoE EP with EPLB, DSpark, SWA bounded replay, asynchronous scheduling, batch-invariant inference, deterministic sampling, and structured output.

The same README lists a KV cache system, prefill/decode disaggregation, and load balancing across data-parallel replicas as missing service parts. The public release therefore supplies a model runtime baseline, not a complete HTTP service.

`deepinfer/config.py` defines data-parallel, context-parallel, expert-parallel, and Engram-parallel sizes. It derives model parallelism from context parallelism and builds explicit groups in `parallel_state.py`. The configuration also accounts for extra runtime positions for draft and scheduling metadata.

`csrc/engine/scheduler.hpp` and `deepinfer/engine/llm_engine.py` provide iteration scheduling, paged block allocation, prefill and completion engine modes, preemption, sequence outputs, and MTP metadata. The scheduler and engine are C++/Python components in one process group. They are not a Rust service plus a ZMQ worker.

`deepinfer/worker/cache_engine.py` defines paged SWA, compressed-KV, and index-key storage. It uses model-specific shared-layer maps, packed cache formats, and extra state for replay and compression. `cuda_graph_runner.py` captures graph variants for selected batch sizes.

### Operator facts

FlashMLA supplies V4.1 sparse prefill and decode kernels for FP8 or FP4 KV cache. It also supplies fused norm, RoPE, attention, output RoPE, and cast operations. Its current README targets SM100 or SM103, CUDA 13.1 or later, and PyTorch 2.10 or later. The 2026-09-30 release changed the KV format and removed Hopper and earlier-model support.

The README reports V4.1 sparse kernel results up to 1350 TFLOPS for prefill and 1024 TFLOPS for decode on B200 with CUDA 13.3. These are kernel measurements, not service throughput or latency results.
The same README reports 1460 TFLOPS for prefill and 950 TFLOPS for decode in its V4 fused norm-RoPE-attention-RoPE-cast section. Do not use these V4 figures as a DSV4.1 baseline.

DeepGEMM supplies FP8, FP4, and BF16 GEMMs, grouped MoE GEMMs, MQA scoring for the sparse indexer, Hyper-Connection kernels, Mega MoE, and PDL-aware paths. Its README states that input transposition and FP8 casting are caller responsibilities unless a caller fuses them into a prior kernel. Its grouped inference API uses fixed N and K dimensions, aligned M segments, and a masked form for decode graphs.

DeepEP supplies expert-parallel dispatch and combine through `EPBuffer`. A dispatch returns a handle with token layout, expert counts, source metadata, and combine state. The handle is consumed by combine. The library also has experimental bucket collectives, pipeline and context parallel paths, and GPU or CPU Engram storage. It uses PyTorch process groups, NCCL, NVLink, RDMA, and runtime JIT compilation.

The reviewed sources do not define pinned host allocation for Engram weights. They show CPU or GPU Engram storage and shared-memory registration, but they do not bind table residency to `cudaHostRegister` or define a transfer lifetime. Treat pinned Engram residency as an open project decision.

TileKernels supplies TileLang implementations for MoE routing, FP8/FP4 conversion, Engram gates, mHC, and RoPE. Its README targets SM90 or SM100, CUDA 13.1 or later, PyTorch 2.13 or later, and TileLang. It presents kernel and modeling APIs, not a request scheduler or public service.

## Reusable ideas

These ideas can guide model-specific design without importing the upstream libraries:

- Use explicit prefill and decode plans. Give each plan fixed layouts, capacities, cache formats, and graph conditions.
- Keep sparse attention metadata separate from logical cache identity. Candidate blocks, top-k indices, compressed-KV positions, and cache slots are derived execution data.
- Treat MoE dispatch and combine as one semantic operation. Carry a typed handle or equivalent state from route preparation through combine.
- Keep FP8/FP4 scale layouts in the operator contract. State block sizes, packing order, alignment, and output precision before kernel tuning.
- Fuse small producer and consumer work when the complete operation benefits. Candidate examples are norm-plus-cast, norm-plus-RoPE, attention-plus-cast, and mHC stages.
- Make Engram lookup asynchronous. A fetch must return a readiness event or equivalent lifetime token before the model reads the result.
- Make DSpark proposal, confidence, verification, acceptance, and state restoration separate semantic steps. Accepted tokens, not proposed tokens, update public request state.
- Use deterministic metadata and stable ordering when batch composition can change. Batch-invariant behavior is a model capability, not a generic scheduler promise.

## Conflicts with this repository

The repository contracts in [`docs/requirements.md`](../requirements.md) prohibit new external operator libraries such as DeepGEMM and FlashInfer. They permit project CUDA, CUTLASS, native CUDA libraries, and CuTe DSL. Therefore:

- Do not add DeepInfer, FlashMLA, DeepGEMM, DeepEP, TileKernels, TileLang, or their JIT build systems as runtime dependencies.
- Reimplement accepted algorithms behind the existing operator `plan` and `workspace` interfaces. Keep native identity, source hashes, preparation, graph lifetime, and resource ownership in the repository contract.
- Do not copy the DeepInfer C++ scheduler into the service. Rust owns admission, scheduling, logical KV state, deadlines, and validated state commit.
- Do not let a Python process group replace the rank protocol. The Rust client owns rank startup, typed plans, msgpack payloads, completion checks, and failure handling.
- Do not treat DeepInfer's offline engine as proof of HTTP, stored responses, tools, media, cancellation, or worker replacement support. The README explicitly leaves service pieces out.
- Do not adopt DeepInfer's FP8 or FP4 cache layout without binding it to the DSV4 cache schema. The current DSV4 report leaves packed cache versus reference BF16 cache as an open decision.
- Do not claim TP, TEP, or DEP support from DeepInfer's CP, EP, and DP names. The adapter must define rank ownership, request placement, collective plans, cache groups, and accepted-state publication for this repository.

## Recommended boundaries

### Rust serving boundary

Rust should own the public API, admission limits, request queues, prefill and decode priority, logical cache blocks, and prefix identity. It should also own cancellation, deadlines, worker lifecycle, and accepted-state commit.
A step payload should contain typed token work, positions, batch metadata, cache bindings, parallel descriptors, and speculation metadata. It should not contain full request histories or private kernel layouts.

The Rust scheduler should select a complete execution mode before dispatch. It should treat prefill, ordinary decode, and speculative verification as model capabilities supplied by the adapter. It should publish one result only after all ranks and the Python worker report validated completion.

### Python adapter boundary

The adapter should own tokenizer and media codecs, DSV4 model equations, rank-local weights, and physical cache views. It should also own sparse-attention metadata, MoE routing metadata, Engram transfer views, DSpark state, and sampling inputs.
It should translate typed Rust assignments into operator inputs and return named outputs plus state deltas.

The adapter should provide complete semantic calls for MLA compression and sparse attention, routed MoE, Engram lookup, mHC, output projection, and DSpark verification. It may use fused native kernels internally.
It should not send kernel launch counts, private buffer indices, or upstream library objects to Rust.

### Native operator boundary

Each operator should have a plan, prepared weights, workspace, batch view, state view, and named outputs. Plans should be pure selection data. Workspaces should own temporary storage, native contexts, graph resources, and asynchronous transfer lifetimes.

The first DSV4 operator baseline should use these families:

1. FP8/FP4 conversion and projection with explicit scale and packing contracts.
2. MLA compression, index scoring, candidate-block selection, and sparse attention with separate prefill and decode plans.
3. MoE gate, dispatch, grouped expert computation, and combine with a typed route handle.
4. Engram hash, host-row fetch, device staging, gate, and projection with explicit event ownership.
5. mHC normalization, mixing, and residual application.
6. DSpark draft, confidence, verification, correction, and acceptance as separate semantic operations.

These boundaries match the public operator evidence while keeping the service contract local. They also permit PDL and fusion without changing Rust ownership or the ZMQ/msgpack boundary.

## Decision

Use DeepInfer as the public model-runtime and scheduling reference. Use FlashMLA, DeepGEMM, DeepEP, and TileKernels as algorithm, layout, and kernel-shape references. Do not import their serving or build systems.

Freeze DSV4 cache formats, rank strategies, Engram transfer lifetime, and DSpark acceptance state before performance work. Then implement each accepted operator through the repository plan/workspace contract and validate it against the existing FP64 and exact-operator fixtures.

## Sources

- [DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/README.md)
- [DeepInfer configuration](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/config.py)
- [DeepInfer engine](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/engine/llm_engine.py)
- [DeepInfer scheduler](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/scheduler.hpp)
- [DeepInfer cache engine](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/worker/cache_engine.py)
- [DeepInfer V4.1 attention](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/mla_attn.py)
- [FlashMLA README](https://github.com/deepseek-ai/FlashMLA/blob/0f11e178da1f67becb087a324b400854931ec3e3/README.md)
- [FlashMLA cache format](https://github.com/deepseek-ai/FlashMLA/blob/0f11e178da1f67becb087a324b400854931ec3e3/csrc/cuda_kernels/kv_cache_format.h)
- [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md)
- [DeepEP README](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/README.md)
- [DeepEP EP buffer](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/ep.py)
- [DeepInfer shared memory](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/shared_memory.py)
- [DeepEP Engram buffer](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/engram.py)
- [TileKernels README](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/README.md)
- [`oh-my-vllm` runtime contracts](../requirements.md)
- [`oh-my-vllm` operator design](../operators/design.md)
- [`oh-my-vllm` GPU execution rules](../execution.md)
- [DSV4 model and checkpoint contract](dsv41-model-checkpoint-contract.md)
