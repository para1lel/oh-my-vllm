# GPU cache

Rust owns token identities, reservations, pins, and eviction through the [Qwen policy](../../../crates/scheduler/src/policy/qwen.rs).
Python owns the [physical cache](../../../python/oh_my_vllm/models/qwen38/cache.py).
KV history and recurrent checkpoints stay on the GPU.
There is no KV host storage or host eviction path.
Necessary token, metadata, and draft readback transfers use the [execution rules](../../execution.md).

The new layout has no completed GPU, full-model, HTTP, or AgentX acceptance.
The [recorded results](results.md) apply to their recorded source and older cache layout.

## Storage

The layout identity is `qwen_radix_gdn8192_fp8kv128_v1`.
The [layout calculation](../../../python/oh_my_vllm/models/qwen38/cache_budget.py) separates usable persistent capacity from other storage.

| Quantity | Meaning |
|---|---|
| `C` | Usable persistent GDN checkpoints |
| `P = 64 * C` | Usable persistent KV pages |
| 128 tokens | KV page length |
| 8192 tokens | Periodic GDN checkpoint interval |
| One null slot per pool | Separate from usable capacity |
| `6 * N` | Private checkpoint slots for `N` request slots |
| `ceil(T / 8192) + N` | Scratch checkpoint slots for token budget `T` |

Target FA layers share logical page IDs. MTP adds one FA layer with the same format.
Each layer has fixed-address dense slabs with these shapes:

| Tensor | Type | Shape |
|---|---|---|
| Codes | Raw E4M3FN bytes, `uint8` | `[P+1, 2, 128, 4, 256]` |
| Scales | FP32 | `[P+1, 2, 128, 4]` |

Each scale applies to one 256-element K or V vector for one token and head.
Null page zero is not valid history.
The page-table width is `ceil(max_context / 128)`, or 2048 at 262144 tokens.
There is no complete BF16 copy of KV history.
Recurrent checkpoints use FP32. Convolution and saved hidden values use BF16.

The [FA operator](../../../python/oh_my_vllm/operators/fa_fp8) fuses quantization with the K/V producer and decoding with attention reads.
New rows and reused rows use the same stored codes and scales.
Q and P use BF16. QK and PV use BF16 operands with FP32 softmax and accumulation.
The [format reference](../../../development/operators/qwen_fp8_reference.py) specifies rounding and scale calculation.
Page copies and rollback move codes and scales together without quantization again.

The [GDN operator](../../../python/oh_my_vllm/operators/gdn_radix) uses explicit export slots independent of KV pages.
Its 64-row computation splits at explicit exports and sequence ends.
The [MTP operator](../../../python/oh_my_vllm/operators/mtp_fp8) uses the same FP8 cache for context and drafts.
The older FA, GDN, and MTP operators stay available at the operator layer.

## Independent prefix state

One token radix tree compares exact token values in a cache domain.
Checkpoint attachments and page attachments have separate capacity, versions, eviction, and pins.
Eviction detaches only the selected attachment. It does not delete the other attachment or descendants.

The [prefix lookup](../../../crates/scheduler/src/policy/qwen/lookup.rs) returns three different positions:

| Position | Meaning |
|---|---|
| `D` | Deepest checkpoint hit |
| `K` | End of the continuous KV prefix |
| `R` | Available recovery checkpoint at or before `K` |

`min(D, K)` does not create a recovery checkpoint.
Valid page spans after a missing page stay available for reuse.
Replay follows the complete layer order from `R`.
Reused spans skip K/V projection, quantization, and writes while hidden, GDN, query, attention, and FFN computation continue.
A complete prompt hit uses saved hidden values with the current request's output head and sampler.
It does not compute the last prompt token again.

Allocation pins all selected sources before it reserves destinations.
The checkpoint pool, page pool, private states, exports, and copy-on-write pages form one reservation.
A failed reservation releases its temporary resources.
Copy-on-write protects shared partial pages until validated completion.

## Successful session tails

The [session registry](../../../crates/scheduler/src/session.rs) orders admission before asynchronous request preparation.
One `SessionId` has at most one successful tail.
The tail value contains its `CacheDomain`, so a successful tail in a new domain replaces the previous tail for that session.
A failed new preparation keeps the previous successful tail and does not reverse admission order.
An older request that completes late cannot replace the newer admission's tail.

Responses branches inherit their root session and cache namespace, with separate active pins.
Anonymous requests use the bounded cache without a permanent session directory.
Tail metadata and unpinned cache state have capacity limits.

The [service identity contract](../../serving.md#cache-identity) defines the HTTP fields.

## Exact completion

The parser fixes accepted endpoint `A` before the request enters `Finalizing`.
The [completion policy](../../../crates/scheduler/src/policy/qwen/completion.rs) compares computed endpoint `F` with that endpoint.

| Condition | Completion work |
|---|---|
| `F < A` | Compute only the accepted suffix that has not executed |
| `F = A` | Export or reuse the endpoint |
| `F > A` | Restore the accepted verification endpoint and KV valid ends, or replay from an available earlier checkpoint |

Finalization uses no sampling and adds no output tokens, usage, RNG advance, or grammar changes.
Proposed state cannot replace accepted state.
Ordinary decode also saves the output-head hidden values at the exact endpoint.
MTP context keeps its original rule, token `p` with target hidden `p-1`.

The [output workspace](../../../python/oh_my_vllm/operators/output/workspace.py) separates hidden values for the output head from MTP context hidden values.
The output head uses the original 640-thread sampling normalization.
The older small-row hidden export uses a different reduction and cannot replace output-head hidden values without numerical proof.
MTP storage adds a separate BF16 vector of width 5120, or 10240 bytes per checkpoint bundle.

HTTP success, final SSE markers, and stored-response publication wait for exact completion and full reply validation.
Parser and storage-capacity preparation precede the successful tail replacement.
Cancellation of submitted work waits for computation, auxiliary streams, rollback, exports, copies, and all participating rank acknowledgements before pin release.
The [shared lifecycle](../../cache-policy.md) defines retirement and worker replacement.

## Cache memory

The total cache byte ceiling stays at the older layout's budget level.
Null slots, private slots, scratch checkpoints, hidden values, and FP32 scales count toward that ceiling.
The [workspace plan](../../../python/oh_my_vllm/models/qwen38/memory.py) also counts operator workspaces, metadata, sampling, and MTP backup storage.
Checkpoint-slot counts alone do not bound GDN temporary storage.
CUDA Graph, allocator, and driver peaks are not measured.
No new usable capacity or physical peak is accepted.

`--mamba-blocks` selects usable checkpoints. `--num-gpu-blocks` selects usable pages.
The selected values must satisfy `P = 64 * C` and the byte budget.
Conflicting options cause an error.

Ready reports usable, null, private, and scratch counts, pool bytes, layout identity, and format fingerprint.
Rust constructs its logical pools after Ready validation.
The [wire contract](../../adapters.md#execution-protocol) defines version and replacement checks.
