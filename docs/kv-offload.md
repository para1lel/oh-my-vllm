# KV offload

## Scope

The host cache keeps interrupted requests and reusable prefixes.
Rust owns logical identities, references, admission, and eviction.
Python owns physical storage and CUDA transfers.
The GPU cache keeps fixed addresses for CUDA Graph execution.
The GPU capacity still uses the runtime 85% memory budget.

## Ownership

A cache entry can be GPU resident, storing, host resident, or loading.
Each transfer identifier is unique for one worker lifetime.
The scheduler reserves destination capacity before a transfer.
Source references stay valid until the worker reports completion.
Cancellation does not release memory that an unfinished transfer can access.
Worker replacement invalidates all host entries from that worker.

A request becomes eligible only after its load completes.
Other resident requests stay eligible during transfers.
The service polls transfers when no request can execute.
The compute stream does not wait for unrelated loads.

## Storage

The host pool allocates pinned slabs on demand and reuses free storage.
Its byte limit includes allocated slabs, reservations, and transfers.
The automatic limit respects available host memory and process memory limits.
An explicit byte limit can restrict this capacity.
The pool grows only when compute and transfer streams are idle.
It does not allocate or release pinned storage for each copy.

Capacity pressure first removes unused host prefixes.
Request snapshots stay protected until restoration or cancellation.
If host capacity cannot permit progress, the service reports a capacity error instead of resetting request progress.

Dedicated nonblocking streams carry H2D and D2H transfers.
The native runtime submits independent copies in batches.
D2H copies wait for their source producer event.
Events protect source tensors, destination tensors, and host slices.
Event queries report completion without waiting on the CPU.
Transfers can overlap computation, but they can still compete for memory bandwidth.

## State

A request snapshot includes target and MTP attention blocks, GDN state, convolution state, and boundary hidden values.
It includes the selected private endpoint, not only logical checkpoint slots.
Pending stochastic drafts keep their proposal distributions.
CPU history, constraints, decoder state, positions, drafts, and RNG offsets do not change.
Offload does not regenerate proposals or reset computed progress.

A reusable prefix contains validated attention blocks and a matching recurrent checkpoint.
The scheduler stores reusable prefixes when GPU capacity pressure makes eviction necessary, not after each completed request.
Host prefix retention is bounded and does not guarantee a hit for each old prefix.
Partial endpoints keep their exact token count.

Host prefixes do not contain another request's sampling state.
Shared blocks keep their reference and copy-on-write rules.
The host index publishes a prefix only after all required copies complete.
A restored partial checkpoint keeps a separate read slot until the first continuation result passes validation.

## Verification

Ownership checks cover delayed completion, cancellation, capacity pressure, and stale results.
GPU checks compare restored values and fixed-seed continuation.
Timeline checks examine transfer overlap and allocation windows separately.
The AgentX measurement uses MTP4 and four session trees.
Its dataset, seed, profiling duration, and drain limit do not change.
The [test scope](testing.md) controls the other required checks.

## References

The design uses the [vLLM asynchronous transfer pattern](https://github.com/vllm-project/vllm/blob/390836a3936629475696a2416378cf2f2aee8827/vllm/v1/kv_offload/cpu/gpu_worker.py).
The [SGLang cache controller](https://github.com/sgl-project/sglang/blob/d0a75654368a470d8484e3ea45b06e500fee57af/python/sglang/srt/managers/cache_controller.py) also separates transfer completion from cache ownership.
These projects are references, not runtime dependencies.
