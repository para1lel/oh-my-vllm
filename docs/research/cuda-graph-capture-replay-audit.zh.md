# CUDA Graph Capture 和 Replay 审查

审查日期: 2026-10-29.

范围: 审查当前 capture, replay, collective 和 shutdown path.
本审查不将当前会话的 SOTA 目标加入 repository gate.

## 决定

Qwen single-worker path 遵循所需的 CUDA Graph design.
capture helper 将 private stream 与 caller stream 和 registered transfer stream 连接.
它拒绝 active caller capture.
Qwen model 使用 LRU 限制 decode workspace, 并在 release 前关闭被淘汰的 workspace.
测试覆盖 stream handoff, replay, cleanup order 和 failed recording.

DeepSeek 不使用 CUDA Graph execution.
它的四 rank bridge 对 target 和 DSpark work 使用 Eager execution.
这是当前安全选择, 因为 adapter 没有 coordinated rank capture plan, static graph buffer 或 NCCL Graph completion path.

在接受新的 multi-rank Graph adapter 前, 仍有两个 gap:

1. Graph cleanup failure 没有发布到 worker diagnostics.
2. DeepSeek rank group 需要 model-specific Graph eligibility 和 pending-NCCL-error path.

Fixed capture bucket, graph memory pool, graph upload 和更窄的 release synchronization 仍是可选 performance work.

## 固定来源

审查使用以下 primary source:

- CUDA 13.1 [stream capture API](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/group__CUDART__STREAM.html), [Graph API](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/group__CUDART__GRAPH.html) 和 [stream capture guide](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture).
- NCCL commit [`3e46c089f4ae4732722a419eb6096aead7e47a30`](https://github.com/NVIDIA/nccl/tree/3e46c089f4ae4732722a419eb6096aead7e47a30): [CUDA Graph use](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/docs/userguide/source/usage/cudagraph.rst) 和 [communicator progress](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/docs/userguide/source/api/comms.rst).
- vLLM commit [`7c713621f3e67bd8579d320571e06cefdeab30ed`](https://github.com/vllm-project/vllm/tree/7c713621f3e67bd8579d320571e06cefdeab30ed): [Graph manager](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/v1/worker/gpu/cudagraph_utils.py), [Graph wrapper](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/compilation/cuda_graph.py) 和 [distributed capture stream](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/distributed/parallel_state.py).
- SGLang commit [`3927a73a340708192fb41a4cc2f6c578df734ce0`](https://github.com/sgl-project/sglang/tree/3927a73a340708192fb41a4cc2f6c578df734ce0): [capture buckets](https://github.com/sgl-project/sglang/blob/3927a73a340708192fb41a4cc2f6c578df734ce0/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py), [full Graph backend](https://github.com/sgl-project/sglang/blob/3927a73a340708192fb41a4cc2f6c578df734ce0/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py) 和 [static buffer registry](https://github.com/sgl-project/sglang/blob/3927a73a340708192fb41a4cc2f6c578df734ce0/python/sglang/srt/model_executor/cuda_graph_buffer_registry.py).

## 发现

### Stream handoff 已实现

`Device.capture` 在 caller stream 和每个 registered transfer stream 上记录 event.
private capture stream 在 recording 前等待这些 event.
capture 后它恢复 caller stream, 并拒绝 active caller capture.
`tests/test_graph.py` 覆盖这些条件.

### Qwen 的 Graph residency 有界

`qwen38.model.Model` 最多保留八个 decode workspace.
它在移除最旧 workspace 前先关闭它.
workspace key 包含 native plan reuse field.
这与 vLLM 和 SGLang 使用的 fixed-bucket 或 bounded-cache design 一致.

### NCCL failure 仍是 fatal boundary

collective peer 在结束 capture 前轮询 host enqueue progress.
replay 后记录 completion, 并在读取 output 前等待.
pending NCCL capture error 使用 fatal process path.
该 error 之后 generic Graph cleanup 不得发起新的 CUDA 或 NCCL work.

DeepSeek 没有 Graph collective path, 因此它的 rank bridge 尚未执行此 contract.

### Cleanup diagnostics 仍不完整

`Device.graph` 保留原始 body exception, 并在 unsafe cleanup 后将 device 标记为 unusable.
worker 还没有把每个 synchronization 或 destruction failure 发布到 diagnostics.
下一个 Graph adapter 应在 group shutdown 前将 cleanup status 加入 worker failure record.

## DeepSeek acceptance boundary

DeepSeek 使用 CUDA Graph 前, adapter 必须:

1. 在每个 rank 准备 fixed buffer, native workspace 和 communicator state.
2. 证明每个 rank 使用相同的 shape, address, collective order 和 execution mode.
3. 在 capture 前连接 caller 和 transfer stream.
4. 在 replay 期间保持 graph owner 和 cache view 存活.
5. 在结束 capture 前轮询 NCCL enqueue completion.
6. 在读取 output 前在 replay stream 上记录 completion.
7. 在 pending NCCL error 或 unsafe cleanup 后使 worker 不可继续使用.

这些检查完成前, DeepSeek 保持 Eager.
本审查不做 DeepSeek throughput claim.

## 当前 upstream 复查

当前 vLLM source 仍使用 capture-size bucket, graph manager, graph wrapper,
capture memory pool, persistent capture output 和 replay 前的 offloader
synchronization. 参见 [vLLM graph manager](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/cudagraph_utils.py).

当前 SGLang source 将 shape selection 与 graph mechanics 分离.
它的 base runner 计算 capture batch size, phase runner 拥有 static input
buffer 和 output slicing, backend 为每个 shape 管理 capture 和 replay.
参见 [SGLang base runner](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py)
和 [backend registry](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/runner_backend/__init__.py).

这些当前实现确认了上面的决定. DeepSeek 仍缺少 fixed multi-rank input 和
output buffer, shape bucket, rank-symmetric capture 以及 collective completion
和 error contract. 本次 delivery 不需要修改 Graph code.
