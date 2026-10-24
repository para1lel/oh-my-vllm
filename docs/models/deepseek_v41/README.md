# DeepSeek V4.1 adapter

## Scope and status

This approved design specifies the DeepSeek model adapter on `nightly`.
It does not declare DeepSeek support in the current executable.
Execute the [phase plan](plan.md) without additional phase approval.
The [validation plan](validation.md) gives the approved measurement matrix.

DeepSeek-V4.1-Flash uses four B200 GPUs.
It is a peer of [Qwen](../qwen38/README.md) in the shared framework.
Support 64 active requests and a total input/output limit of 1048576 tokens per request.
Support ordinary decode and DSpark with five draft tokens.
Keep Engram weights in pinned host memory.

Rust owns the service, admission, scheduling, logical state, and validated state commit.
Python owns model computation and physical tensors.
Keep ZMQ DEALER and msgpack for process control.
Use CUDA C++ or CuTe DSL for model operators.
Native CUDA and NCCL can supply device and communication mechanisms.
Do not add external operator libraries or vLLM dependencies.

## Source facts

The local checkpoint configuration selects `DeepseekV41ForCausalLM` and `deepseek_v41`.
It has 40 target layers, width 5120, and a vocabulary of 129280 tokens.
It has 384 routed experts, six selected experts, and one shared expert per target layer.
It specifies three prediction layers, `dspark_block_size=5`, and `max_position_embeddings=1048576`.
The vision configuration has 32 layers and patch size 14.

The architecture uses causal encoder-decoder computation, compressed sparse attention, shared cache sources, hyper-connections, and token-based Engram lookup.
FP8 matrices use 32-by-32 blocks and UE8M0 scales.
Experts use packed FP4 weights.
Main compressed KV uses FP4 with separate scale rules.
These formats differ from the Qwen operators.

The [fixtures](fixtures.md) bind the complete checkpoint identity.
The [memory plan](memory.md) gives weight placement and allocation gates.

Use the checkpoint's reference model, encoder, and image processor as mathematical sources.
Its generation loop implements ordinary sampling, not full DSpark acceptance.
Use the authoritative Python encoder and the [fixed codec cases](fixtures.md#prompt-codec) for prompt integration.

[DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9) is a reference, not a runtime dependency.
Its default checkpoint has the `-1017` suffix.
Check source and weight compatibility before reuse.

Record a fixed source revision, license, and algorithm differences in the implementation contract.
Do not copy its external operator imports, scheduler, or environment.
Its [parallel configuration](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/config.py) uses context parallelism, not the TP4 definition here.
Its [replay guidance](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md) permits approximate bounded reconstruction.
Its [sampling code](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/fused_sample.py) adjusts the rejection distribution with a numerical penalty.
Do not adopt these approximations as exact state restoration or exact sampling.

## Integration boundaries

Use the shared [adapter contract](../../adapters.md), [rank control](../../distributed.md), and [operator design](../../operators/design.md).
Keep DeepSeek weights, execution, cache, media, and DSpark in `python/oh_my_vllm/models/deepseek_v41/`.
Model computation is pending.

## Parallel strategies

These are approved project definitions. They are not claims about DeepInfer command names.
All strategies use one group of four ranks, not sixteen GPUs.

| Strategy | Attention and dense projections | Routed experts | Requests |
|---|---|---|---|
| TP4 | Tensor shards across four ranks | Each expert has four tensor shards | Same scheduled requests on all ranks |
| TEP4 | Tensor shards across four ranks | Expert ownership across four ranks | Same requests, with token dispatch and combination |
| DEP4 | One attention copy per rank | Expert ownership across four ranks | Request partitions across four ranks |

TEP4 changes tensor layout at the expert boundary.
DEP4 exchanges expert tokens across the same four ranks.
It is not four independent full model replicas.

Specify shared-expert, embedding, vocabulary-head, vision, and DSpark placement in each weight plan.
Count replicated latent KV separately from partitioned attention heads.
DEP4 keeps a request on its attention owner until an explicit state transfer completes.
Empty ranks participate in the agreed collective sequence.

Rust controls one group step and its rank schedule.
Each rank uses a DEALER identity and reports capacities and completion.
Tensor data moves through native collectives, not msgpack.

Rust publishes one request result after all necessary rank results pass validation.
Messages carry protocol version, group epoch, step ID, rank ID, and state transaction ID.
Reject stale, duplicate, incomplete, or inconsistent results.
A failed rank fails the group and its active requests.
Restart only after all old ranks exit. Do not promise transparent recovery of active requests.

## State and execution

Represent cache groups by identity, kind, token interval, position, format, owner ranks, and capacity.
The initial kinds cover full attention, recurrent checkpoints, sliding windows, compressed KV, index keys, and model-specific checkpoints.
Related groups form one state transaction.
Validate allocations and rank completion before state commit or prefix publication.

DeepSeek state also contains compressor partial groups, shared-source positions, candidate indices, Engram history, and DSpark hidden states.
Separate reusable accepted state from speculative writes.
Copy-on-write protects partial prefixes and source blocks.

Prefix keys bind model weights, tokenizer, media, position rules, cache format, and parallel layout.
Host offload stores sufficient accepted state for exact continuation.
Bounded sliding-window replay must include speculative rollback distance and incomplete compression groups.
Derive the replay length from the model contract. Do not copy a reference constant without proof.

An Execute with any prefill uses Eager execution on the whole group.
Decode-only execution uses CUDA Graphs on all participating ranks.
Graph keys include model, topology, batch capacity, draft capacity, and communication plan.
Keep tensor addresses, collective order, and padding rules fixed through capture and replay.
Engram transfers and GPU consumers use explicit completion dependencies.
Keep the existing metadata upload and resource-release rules.

## Memory and capacity

Keep authoritative Engram tensors and scales in pinned host storage.
Share or partition physical host storage across local ranks. Do not create four full host copies by default.
Use bounded device staging for lookup rows and Engram projection computation.

Do not silently replace host storage with a full GPU table.
Separate Engram, host KV, upload buffers, and transfer budgets.
Protect their owners until all readers and transfers complete.
Allocate large host storage before serving. Apply NUMA placement and startup memory checks.

The model report states 890 bytes per token for global KV.
That gives 55.625 GiB for 64 requests at 1048576 tokens, before replication and other state.
This figure is not the service memory budget.

Budget each rank for weights, all cache groups, communication, graph pools, media, and temporary storage.
Bound prefill chunks and candidate workspaces. Do not allocate dense context-squared attention scores.
Use 64-bit byte sizes and checked arithmetic.

Support 64 active requests without serial admission masquerading as concurrency.
Separate scheduler concurrency from smaller physical kernel fragments.
Capacity acceptance must exercise the combined concurrency and context target.

## Function parity and risks

Keep the shared APIs, CLI, tools, structured outputs, sampling, streaming, storage, cancellation, and failure behavior in the validation checklist.
Use model-specific prompt, reasoning, tool, image, and video codecs.
Do not reuse Qwen token IDs or three-axis media positions for DeepSeek.
DeepSeek video uses ordered frame images and timestamps only after encoding and numerical validation.
Do not advertise full parity until that path passes acceptance.

DSpark is a separate algorithm, not MTP4 with a different loop count.
Five drafts can use six target verification rows. Confidence controls verification length, not acceptance probability.
Its contract must specify proposal probabilities, confidence scheduling, verification positions, rejection correction, RNG, and accepted-state restoration.
Target and draft paths apply the same sampling constraints.
Rejected drafts cannot change grammar, penalties, history, or prefix state.

The first gates are checkpoint compatibility, numerical references, four-rank failure safety, and the full memory budget.
Do not change numerical limits to make an implementation pass.
Do not claim throughput before the specified measurements.
