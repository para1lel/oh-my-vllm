# DeepSeek V4.1 adapter

DeepSeek-V4.1-Flash uses four Python ranks and the [shared service](../../serving.md).
The public capability is text target generation with greedy DSpark5 verification.
Image serving and full workload acceptance are not available.

The [adapter](../../../python/oh_my_vllm/models/deepseek_v41/adapter.py) checks the released configuration.
The [loader](../../../python/oh_my_vllm/models/deepseek_v41/loader.py) reads the original 48-shard checkpoint with rank-local placement.
Startup identity includes configuration and index metadata.
Loading does not hash full weight shards before Ready.
Measurement artifacts bind full checkpoint identities.

## Guides

- [Numerical acceptance and independent reference restoration](evaluation.md)
- [Quantization and numerical order](quantization.md)
- [Projection arithmetic and native interfaces](gemm.md)
- [Physical cache and upstream format reference](cache.md)
- [HTTP workloads](performance.md), [reference calculation](accounting.md), and [AgentX workloads](agentx.md)

## Execution

The [worker](../../../python/oh_my_vllm/models/deepseek_v41/inference.py) supports `tp4`, `tep4`, and request-owned `dep4`.
TP4 target decode uses CUDA Graphs when its bindings permit capture.
TEP4 and DEP4 target decode use Eager execution because they read routed counts on the host.
Prefill and draft verification use Eager execution.
The [Graph implementation](../../../python/oh_my_vllm/models/deepseek_v41/decode.py) owns captured buffers and cache leases.
Uncaptured shapes and calls beyond the configured Graph limit use Eager execution.

`OH_MY_VLLM_DEEPSEEK_BATCH_DECODE=1` selects experimental batched target execution.
Its default is off.
The [batched arithmetic contract](gemm.md#batched-target-arithmetic) defines per-request projection and collective rules.
Full checkpoint, state, capacity, and HTTP acceptance are incomplete for DEP4 and experimental batching.

DEP4 keeps each request on its Rust-selected owner rank.
That owner computes dense layers with full weights and reads full Engram rows from host storage.
Expert dispatch sends BF16 rows and returns FP32 local sums.
Ranks with no owned rows also participate in expert exchanges and completion.
The [DEP4 executor](../../../python/oh_my_vllm/models/deepseek_v41/dep4.py) coordinates target and proposal calls.

### MTP proposal Graph

`OH_MY_VLLM_DEEPSEEK_MTP_GRAPH=1` selects the proposal Graph in the [execution settings](../../../python/oh_my_vllm/models/deepseek_v41/trace.py#L40)<!-- source: def execution_modes( -->.
Its default is `0`.
It operates only with TP4 target Graph execution enabled and a positive decode Graph limit.
TEP4 and DEP4 do not use this proposal Graph.

The [proposal path](../../../python/oh_my_vllm/models/deepseek_v41/target_execution.py#L276)<!-- source: def _proposal_graph_step(self, input_ids, main_hidden, start_pos, slot): --> uses one token row of shape `[1,1]` and one hidden-state row of shape `[1,1,15360]`.
Proposal Graph execution is only for greedy proposals in the full-window branch.
The [position limits](../../../python/oh_my_vllm/models/deepseek_v41/proposal_graph.py#L33)<!-- source: def bind(self, input_ids, main_hidden, position): --> are the same for capture and replay.
The [seed path](../../../python/oh_my_vllm/models/deepseek_v41/target_batch.py#L194)<!-- source: def _seed_mtp_batch( --> uses Eager execution.
Target decode Graphs and proposal Graphs are different graphs.
The [cache guide](cache.md#proposal-graph-ownership) gives proposal Graph ownership and release order.

## GPU setup

Use the [development setup](../../development.md#native-modules) and [GPU selection rules](../../distributed.md#gpu-leases).
Build the physical storage module before worker startup:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.native_build --module paged
```

`OH_MY_VLLM_DEEPSEEK_CACHE_CONTEXT` sets physical context capacity, with a default of 65536 tokens.
The descriptor limit is 1048576 tokens.
Physical capacity must cover the requested prompt and output together.
The descriptor limit does not prove that this capacity is available.

`OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS` sets the total rows in a native prefill chunk.
The default is 8192, and the maximum is 32768.
Smaller chunks use less temporary storage.
The [routed projection wave](gemm.md#routed-prefill-waves) has a separate row setting.
