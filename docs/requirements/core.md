# Core contracts

## REQ-GOAL-001: Inference goal

Supply Rust inference for Qwen3.8-27B-FP8 on one B200 GPU.
Keep all existing offline and HTTP features.
Each performance row must meet `REQ-PERF-001` and `REQ-PERF-002`.

## REQ-ARCH-001: Ownership

Rust controls service, request queues, token budgets, logical KV allocation, prefix reuse, and preemption.
Python controls model loading, GPU forward computation, kernels, and manual CUDA Graphs.
Use one ZMQ DEALER channel with msgpack.
Keep tensors on the Python side.

## REQ-MODEL-001: Target

Use Qwen3.8-27B-FP8 with 48 GDN layers and 16 FA layers.
Use block size 784 on all production paths.
Use FA group 0 and GDN group 1 with `mamba_cache_mode="align"`.
Set the checkpoint location through `OH_MY_VLLM_MODEL` or `--model`.

## REQ-EXEC-001: GPU execution

Use CUDA Graph, multiple CUDA streams, and Programmatic Dependent Launch (PDL) for model computation.

Graph execution must use an enabled native caching allocator without explicit `max_split_size_mb`.
Reject unsupported effective settings at startup without changing them. Explicit eager diagnostics stay available.
Keep captured cuBLAS workspaces in their owning graph pools without changing workspace budgets.

Use semantic operations and tensor/state access to define graph dependencies.
Independent branches can run in parallel.
PDL consumers must wait before they read producer output.

Keep graph capture restoration, cache ownership, and graph storage limits.
Tune execution with full-operation and framework measurements.
Record the configuration, numerical checks, memory, and latency of each selected optimization.
Use separate measurements to identify graph, stream, and PDL effects.
The phase-latency and operator gates stay active.

## REQ-CONTEXT-001: Context and memory

Support input plus output up to 262144 tokens.
Complete the full 258048-input, 4096-output cases in ordinary and MTP4 modes at batches 1, 2, and 4.
These runs must have no OOM or recompute preemption.
Keep the 32768-token step budget and 784-token blocks.
Record performance and peak GPU memory. Add no extra ratio gate.

Include long-context prefix restoration and constrained decoding.
Keep the six ordinary/MTP4 cases as different gates.

## Functional requirements

| ID | Contract |
|---|---|
| REQ-FUNC-001 | Chunk prefill with at most 32768 tokens in one step. Materialize aligned GDN checkpoints. |
| REQ-FUNC-002 | Schedule active decode work before new prefill work in each step. |
| REQ-FUNC-003 | Use chain-hash prefix reuse, LRU eviction, and coordinated FA/GDN groups. |
| REQ-FUNC-004 | Preempt later-admitted running requests first. Keep accepted history and clear unverified drafts for recompute. |
| REQ-FUNC-005 | Supply MTP4 with explicit kept tokens and new draft IDs. Keep BF16 GDN state and block 784. |
| REQ-FUNC-006 | Supply optional DSpark with the local BF16 checkpoint, at most seven drafts, target verification, and block 784. |

DSpark uses zero-based target layer outputs `(5,19,33,47,61)` and five draft GQA layers.
Keep native MTP4 available. Select one draft mode for the worker, not per request.

The initial DSpark cumulative confidence threshold is `0.2`. The proposal count can be zero through seven.
A first confidence product less than the threshold gives one target-token step.
Set `--dspark-confidence-threshold 0.0` for fixed-count proposals up to seven. Output, context, and grammar limits can decrease the proposal count.

For stochastic sampling, use conditional proposal probabilities and target acceptance `min(1,p(x)/q(x))`.
On rejection, sample from normalized `max(p-q,0)`. Sample the bonus token from the target distribution.
Apply user sampling parameters and grammar masks to each conditional distribution.

Keep BF16 GDN state and the existing half-precision rounding constraints in speculative modes.
Only accepted input rows can commit target state or DSpark context.

## REQ-ACC-001: Numerical accuracy

Compare model GQA and GDN inference paths with independent references with CPU FP64 computation.
Include prefill, decode, chunk boundaries, and recurrent state.
Use rounded inference inputs.
BF16 output uses `atol=rtol=0.03`.
Recurrent state must have NRMSE at most 1% and maximum absolute error at most 2% of the reference peak.

Keep path-specific tolerances and existing tests.
Show coherent text, MTP, and prefix hits.
Full-model token equality is not a gate.

## REQ-RUNNER-002: Independent runtime

Use project implementations and third-party libraries.
Project builds, tests, and inference must not install, import, or link vLLM.
They must not depend on its checkout, old environment, or compiled caches.
The former adapters are removed. [ADR-002](../decisions/ADR-002-gpuworker-adapter.md) and [ADR-005](../decisions/ADR-005-v2-model-runner.md) record their replacement.

`REQ-RUNNER-001` is retired and superseded by this requirement.
Select compatible stable dependencies in dependency order and pin versions with satisfactory compatibility tests.
Keep ported-code provenance and licenses.
Document future architectures, multiple GPUs, and other NVIDIA GPUs briefly.

## REQ-OBS-001: Diagnostics

Supply UTC timestamps, monotonic durations, run/request/step correlation, and log levels that the user can set.
Avoid per-step I/O at the default log level.
Debug logs show scheduler, cache, transport, and worker durations.
Identify host time independently from CUDA kernel time.
Make profiling an explicit choice.

Record scheduled, accepted, and returned draft counts for each mode.
Acceptance rate is total accepted drafts divided by total scheduled drafts.
Returned next-step proposals have a different diagnostic counter.

Compare loaded DSpark configuration and weight hashes with the requested checkpoint. They must agree.
