# KV offload

## Scope

The host cache keeps interrupted requests and reusable prefixes.
Rust owns logical identities, references, admission, and eviction.
Python owns physical storage and CUDA transfers.
The GPU cache keeps fixed addresses for CUDA Graph execution.
GPU capacity uses the fixed defaults and startup checks in the [service guide](cache.md#cache-memory).

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

The host pool reserves its usable pinned capacity before Ready and reuses free storage.
Its byte limit includes allocated slabs, reservations, and transfers.
The automatic limit respects available host memory and process memory limits.
An explicit byte limit can restrict this capacity.
The reservation waits for idle compute and transfer streams.
Request processing does not allocate or release pinned cache storage for each copy.

The reservation increases startup time and uses host memory before the first transfer.
The byte limit and prefix retention policy do not change.
Worker startup fails if the pinned reservation cannot be allocated.

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
Its dataset, seed, and profiling duration do not change.
The [AgentX guide](agentx.md) gives the current drain limit.
The [test scope](../../testing.md) controls the other required checks.

## Cache memory

FA cache capacity has fixed defaults: `1824` blocks for ordinary decode and `1608` blocks for MTP4.
These defaults target 85% to 88% total GPU memory use during text inference on one B200 with the default scheduler settings.
The worker does not calculate the block count from available memory.

Use `--num-gpu-blocks` or `OH_MY_VLLM_NUM_GPU_BLOCKS` to change this capacity.
The service uses the command option if supplied.
Otherwise, it uses the environment variable if set. Otherwise, it uses the mode default.
Put the option before `serve`, `run`, or `bench`.

At startup, the worker checks the selected capacity against an 88% tensor budget.
This check includes memory already in use, FA and GDN storage, prefill, and stored decode and sampling workspaces.
The memory snapshot includes loaded weights. The check does not count them again.
The worker rejects a capacity that does not fit. It does not reduce the requested or default block count.

Additional media storage is reported separately. It does not reduce the default text cache capacity.
Media, CUDA Graph, and runtime allocations use the available memory.
Large media workloads can exceed available memory. Use a lower block count in this case.
The startup check does not set a strict limit on total process memory.

The [host cache](cache.md) keeps interrupted requests and reusable prefixes in pinned memory.
Request restoration keeps recurrent state, media, pending drafts, and sampling offsets.
Other resident requests can execute during cache transfers.

Use `--host-cache-bytes` to set a host storage limit in bytes.
The default limit is calculated at startup. The worker reserves usable pinned capacity before Ready.
Use a zero limit to disable host offload for comparisons.

The default logical GDN capacity is `128` slots.
Private request states and temporary checkpoints use additional physical slots.
Use `--mamba-blocks` to change the logical GDN capacity.
The worker reports resolved capacities and memory calculations in Ready, the service log, and the execution trace identity.
Rust creates its logical pools and scheduler after Ready.
Replacement workers must supply the same capacities and satisfy the current memory budget.


## Prefix cache

Requests can use cached full blocks and prefixes with a final block that is not full.
For a prompt with a final block that is not full, the scheduler can record a checkpoint before the last token.
The checkpoint contains FA, GDN, convolution, and MTP hidden state for the same position.
The scheduler makes a checkpoint available to other requests only if its checks of the worker output find no errors.

For a media request, the scheduler records only full block checkpoints.

A request with the same prompt can continue from this checkpoint and do computation for its last token again.
A longer prompt can continue from the same recorded position.
The cache cannot continue from a position that has no GDN checkpoint.

Before computation continues, the worker makes a copy of the shared FA tail in a private block.
The copy contains K and V from the target and MTP layers.
GDN reads the checkpoint and writes to a different slot.
The scheduler keeps the source blocks until its checks of the worker result find no errors.

Without sufficient capacity for these private blocks, the cache uses full blocks.


Temporary model buffers use a [CUDA memory pool](../../../python/oh_my_vllm/operators/runtime/cuda/scratch_allocator.cuh#L19)<!-- source: inline cudaMemPool_t scratch_pool(int device) { -->.
The worker prepares storage before it reports readiness.
The size uses the token budget and at most half the free GPU memory.
The pool keeps this storage for later work. The `gpu_reserved_bytes` statistic includes this storage.

## References

The design uses the [vLLM asynchronous transfer pattern](https://github.com/vllm-project/vllm/blob/390836a3936629475696a2416378cf2f2aee8827/vllm/v1/kv_offload/cpu/gpu_worker.py).
The [SGLang cache controller](https://github.com/sgl-project/sglang/blob/d0a75654368a470d8484e3ea45b06e500fee57af/python/sglang/srt/managers/cache_controller.py) also separates transfer completion from cache ownership.
These projects are references, not runtime dependencies.
