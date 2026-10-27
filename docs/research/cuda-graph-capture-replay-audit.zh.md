# CUDA Graph 捕获和重放审计

审计日期: 2026-10-27. Repository revision: `nightly` at `93a96f7d20d82d4b9197833e517232a0f307e030`.

范围: 只读审计当前 capture, replay, collective 和 shutdown path. 本报告不修改 runtime code. 本报告不把当前会话专用的 operator performance objective 加入 repository gate.

## 决策

当前 Qwen path 满足基本的 single-worker CUDA Graph rule. 它保持 graph buffer 存活, 在 replay 前更新 metadata, 将 RNG state 保存在 device buffer 中, 并在释放 owner 前关闭 graph. 当前 tests 覆盖这些 property.

通用 graph contract 不完整. 必要修改是:

1. 在 capture 前让 private capture stream 等待 caller 和 transfer stream.
2. 限制 graph workspace residency, 并在释放 storage 前关闭被淘汰的 graph.
3. 为未来的 multi-rank model path 保留 NCCL pending-error process exit rule.

Fixed capture bucket, shared graph memory pool, output buffer reuse, graph upload 和更窄的 release synchronization 属于可选 performance work. 它们可以改善 latency 和 memory use. 它们不是本次审计的 correctness gate.

## 固定 source

本审计使用以下 primary source:

- CUDA 13.1 [stream capture API](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/group__CUDART__STREAM.html), [graph API](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/group__CUDART__GRAPH.html) 和 [stream capture guide](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture).
- NCCL commit [`3e46c089f4ae4732722a419eb6096aead7e47a30`](https://github.com/NVIDIA/nccl/tree/3e46c089f4ae4732722a419eb6096aead7e47a30): [CUDA Graph use](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/docs/userguide/source/usage/cudagraph.rst) 和 [communicator progress](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/docs/userguide/source/api/comms.rst).
- vLLM commit [`7c713621f3e67bd8579d320571e06cefdeab30ed`](https://github.com/vllm-project/vllm/tree/7c713621f3e67bd8579d320571e06cefdeab30ed): [graph manager](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/v1/worker/gpu/cudagraph_utils.py), [graph wrapper](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/compilation/cuda_graph.py) 和 [distributed capture stream](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/distributed/parallel_state.py).
- SGLang commit [`3927a73a340708192fb41a4cc2f6c578df734ce0`](https://github.com/sgl-project/sglang/tree/3927a73a340708192fb41a4cc2f6c578df734ce0): [capture bucket](https://github.com/sgl-project/sglang/blob/3927a73a340708192fb41a4cc2f6c578df734ce0/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py), [full graph backend](https://github.com/sgl-project/sglang/blob/3927a73a340708192fb41a4cc2f6c578df734ce0/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py) 和 [static buffer registry](https://github.com/sgl-project/sglang/blob/3927a73a340708192fb41a4cc2f6c578df734ce0/python/sglang/srt/model_executor/cuda_graph_buffer_registry.py).

## Correctness finding

### P1: 通用 capture helper 缺少 stream handoff

[`Device.capture`](../../python/oh_my_vllm/operators/runtime/device.py) 创建新的 nonblocking stream. 它没有让这个 stream 等待 caller stream 或 registered transfer stream.

Model workspace 和 `GraphOperation` 会等待当前 compute stream. Sampling 和 acceptance 可以在 caller-stream copy 后立即 capture, 但没有这个 wait. Helper 本身没有 safe-stream precondition.

CUDA stream capture 记录 work, 而不是执行 work. Cross-stream dependency 仍需要 event 或 stream-wait edge. CUDA 在 [cross-stream capture](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-programming-guide/index.html#cross-stream-dependencies) 中说明此 rule. vLLM 在 capture 前让 capture stream 加入 current stream, 见 [distributed capture context](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/distributed/parallel_state.py#L712-L724).

这是一个通用 correctness defect. 当前 single-stream operator 通常不会显示错误, 因为第一次 replay 在 caller-stream copy 后排队. 未来 operator 或 transfer path 可能在 capture setup 时读取没有完成的 state.

最小修改: 在 caller stream 上 record event, 在 capture stream 上 wait, 并覆盖每个 registered producer stream. 当 caller stream 已经 capture 时拒绝 capture.

### P1: Graph workspace residency 没有上限

[`Model.forward`](../../python/oh_my_vllm/models/qwen38/model.py) 把每个 decode workspace 存入 `self.decode`. Key 包含精确的 rows, context, query, table 和 execution-plan value. Map 没有上限和 eviction. Speculation 在 [`context_decode`](../../python/oh_my_vllm/models/qwen38/speculation.py) 中有相同 pattern.

每个 workspace 持有 device scratch, metadata, operator workspace, output storage 和 graph executable. 具有很多 context 和 shape combination 的 request mix 可以保留足够多的 graph, 最终耗尽 device memory. 这是 resource correctness defect, 不只是 latency issue.

vLLM 将 capture variant 限制为配置好的 descriptor, 并提供 [`release_graphs`](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/v1/worker/gpu/cudagraph_utils.py#L401-L411). SGLang 在 [`get_batch_sizes_to_capture`](https://github.com/sgl-project/sglang/blob/3927a73a340708192fb41a4cc2f6c578df734ce0/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py#L65-L110) 中选择 fixed capture bucket.

最小修改: 使用 bounded capture bucket 或 bounded LRU. 移除 workspace 前先关闭 graph context. Key 必须继续包含 shape, capability, binding 和 execution mode.

### P1 integration blocker: NCCL failure cleanup 必须保持 fatal

NCCL path 遵守所需的 progress order. [`collective_peer.py`](../../development/operators/collective_peer.py) poll host enqueue progress, 只有 NCCL 报告 success 后完成 capture, replay 后 record completion, 并在读取 output 前等待. Native owner 在 [`collectives.cu`](../../python/oh_my_vllm/operators/runtime/cuda/collectives.cu) 中禁止 capture enqueue pending 时进行 CUDA cleanup.

当 pending NCCL capture error 发生时, peer 在 unwind [`Device.capture`](../../python/oh_my_vllm/operators/runtime/device.py) 前调用 `os._exit`. 这符合 NCCL rule: `ncclInProgress` 之后, caller 必须等到 success, 才能进行下一个 NCCL call 或 CUDA kernel. NCCL 还要求所有 participating rank 对 graph capture 和 graph launch 使用一致的 operation. 参见 [NCCL communicator progress](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/docs/userguide/source/api/comms.rst#L221-L236) 和 [NCCL CUDA Graph requirement](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/docs/userguide/source/usage/cudagraph.rst#L13-L18).

当前 probe 是安全的, 因为它强制执行 fatal path. 通用 graph API 对未来 DSV4.1 collective adapter 不安全, 除非 adapter 使用相同 rule. 保持该 rule 在 collective seam. 不要让 generic graph cleanup 在 pending NCCL error 后调用 CUDA.

### P2: Cleanup error 可能被隐藏

当 graph body 抛出 exception 时, [`Device.graph`](../../python/oh_my_vllm/operators/runtime/device.py) suppress synchronization 和 executable destruction 的 error. 它重新抛出原始 exception. 这保留原始 failure, 但可能隐藏 graph leak 或 owner release failure.

Normal closure 的 order 正确: synchronize, destroy executable, 然后 release graph owner. Qwen worker 在 [`Worker.close`](../../python/oh_my_vllm/models/qwen38/inference.py) 中按 offload, model graph, sampler graph 和 device 的顺序关闭. 必要修改是把 cleanup failure 保留在 worker diagnostics 中, 并在 unsafe cleanup failure 后使 worker 不可用.

## 已满足的 contract

| Area | Assessment |
|---|---|
| Preparation 和 warmup | Operator workspace 在 capture 前分配. Qwen 在 path 需要时运行 operator warmup. Sampling 和 acceptance capture kernel, 但不执行 RNG update, 这符合 CUDA capture semantics. |
| 固定地址 | Workspace 和 operator lease 保持 weight, cache view, metadata, scratch 和 output 存活. [`Workspace.bind_media`](../../python/oh_my_vllm/models/qwen38/model.py) 在 binding change 前关闭旧 graph. |
| Metadata | [`Metadata.update`](../../python/oh_my_vllm/worker/metadata.py) 写入稳定的 device arena. Pinned upload buffer 在 [`uploads.py`](../../python/oh_my_vllm/operators/runtime/uploads.py) 中 snapshot CPU input, 并在 caller stream 上 record event. |
| Graph key | Plan key 包含当前 Qwen shape 和 execution-policy field. Sampling key 在 [`sampling.py`](../../python/oh_my_vllm/models/qwen38/sampling.py) 中包含 greedy, compact-history 和 token-selection mode. 缺少的是 lifecycle bound. |
| RNG | Sampling 在 replay 前上传 seed 和 offset. Sampling 和 MTP kernel 更新 device offset. Host 读取更新后的 counter, 并把它提供给下一次 replay. 参见 [`sample.cu`](../../python/oh_my_vllm/operators/output/cuda/sample.cu) 和 [`accept.cuh`](../../python/oh_my_vllm/operators/mtp/cuda/accept.cuh). |
| Output lifetime | Graph owner 持有 output storage. Host download 在 CPU access 前 synchronize caller stream. Existing test 验证 executable destruction 先于 operator owner release. |
| Replay stream | [`Device.graph`](../../python/oh_my_vllm/operators/runtime/device.py) 使用当前 raw CUDA stream replay. Metadata upload 使用相同 stream, 所以 upload 在 graph launch 前执行. |
| Normal shutdown | Qwen path 和 NCCL probe 在释放 buffer 前等待 graph work. NCCL abort 后在 [`collective_peer.py`](../../development/operators/collective_peer.py) 中 poll 到 aborted state. |

Distributed graph implementation 是 development probe. 当前 worker 没有 DSV4.1 multi-rank adapter. 因此在该 adapter 使用既有 rank protocol 前, 本审计不能证明 DSV4.1 rank capture.

## 可选 performance idea

- Capture fixed shape bucket, 把 runtime batch pad 到最小 compatible bucket. 这可以减少 graph count 和 capture churn.
- 使用一个 graph memory pool 并 reuse output buffer. vLLM 和 SGLang 使用 graph pool 和 stable output storage.
- 在第一次 production replay 前 upload executable graph. 否则 CUDA 会在第一次 host launch 时隐式 upload.
- 当 owner contract 允许时, 把 [`release_graph`](../../python/oh_my_vllm/operators/runtime/device.py) 的 device-wide synchronization 换成 graph-stream completion event.
- 只有在 fixed-address 和 metadata contract 通过 test 后, 才使用 graph update 或 executable deduplication.
- 对有 lazy kernel, communication 或 metadata setup 的 model adapter 使用两次 warmup 和 all-rank barrier. 当前 Qwen operator contract 有更窄的 warmup rule.

这些 idea 需要 measurement. 它们不属于本 ticket 的 repository performance gate.

## 最小修改和 test boundary

1. 添加 capture-stream handoff 和 active-capture check. 在 `Device.capture` seam 使用 fake CUDA runtime test. 检查 event record, capture-stream wait, caller-stream restore 和 failure cleanup.
2. 添加 bounded graph cache. 测试超过 bound 的 distinct decode key. 检查 eviction 在 scratch 和 owner release 前关闭 executable. 检查 retained key replay 更新后的 metadata.
3. 保留 NCCL fatal pending-error path. 在 public collective seam 测试相同 rank mode, plan digest, host-only progress, completion fence, graph-owner release 和 abort polling.
4. 把 cleanup failure 保留在 worker diagnostics 中. 测试 body failure 后的 graph destruction failure, 并验证后续 execution disabled.

现有 public boundary 是 [`tests/test_execution.py`](../../tests/test_execution.py), [`tests/test_offload_worker.py`](../../tests/test_offload_worker.py) 和 [`tests/test_collectives.py`](../../tests/test_collectives.py). 本次 document-only change 不增加 test, 也不修改 runtime code.

## 限制

本报告是 contract review. 它没有 GPU benchmark, 也不作 SOTA claim. 它不修改 numerical limit, 不增加 external operator dependency, 也不增加 performance gate. 下一张 implementation ticket 必须在接受 DSV4.1 graph execution 前解决 P1 item.
