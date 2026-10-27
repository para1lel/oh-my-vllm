# CUDA Graph Capture and Replay Audit

Audit date: 2026-10-28.

Scope: review the current capture, replay, collective, and shutdown paths.
This audit does not add the session-only SOTA objective to repository gates.

## Decision

The Qwen single-worker path follows the required CUDA Graph design.
The capture helper joins the private stream to the caller and registered transfer streams.
It rejects an active caller capture.
The Qwen model bounds decode workspaces with an LRU and closes an evicted workspace before release.
The tests cover stream handoff, replay, cleanup order, and failed recording.

DeepSeek does not use CUDA Graph execution.
Its four-rank bridge uses Eager execution for target and DSpark work.
This is the safe current choice because the adapter has no coordinated rank capture plan, static graph buffers, or NCCL Graph completion path.

Two gaps stay before a new multi-rank Graph adapter is accepted:

1. Graph cleanup failures are not published in worker diagnostics.
2. A DeepSeek rank group must have a model-specific Graph eligibility and pending-NCCL-error path.

Fixed capture buckets, graph memory pools, graph upload, and narrower release synchronization stay optional performance work.

## Pinned sources

The audit uses these primary sources:

- CUDA 13.1 [stream capture API](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/group__CUDART__STREAM.html), [Graph API](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/group__CUDART__GRAPH.html), and [stream capture guide](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture).
- NCCL commit [`3e46c089f4ae4732722a419eb6096aead7e47a30`](https://github.com/NVIDIA/nccl/tree/3e46c089f4ae4732722a419eb6096aead7e47a30): [CUDA Graph use](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/docs/userguide/source/usage/cudagraph.rst) and [communicator progress](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/docs/userguide/source/api/comms.rst).
- vLLM commit [`7c713621f3e67bd8579d320571e06cefdeab30ed`](https://github.com/vllm-project/vllm/tree/7c713621f3e67bd8579d320571e06cefdeab30ed): [Graph manager](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/v1/worker/gpu/cudagraph_utils.py), [Graph wrapper](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/compilation/cuda_graph.py), and [distributed capture stream](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/distributed/parallel_state.py).
- SGLang commit [`3927a73a340708192fb41a4cc2f6c578df734ce0`](https://github.com/sgl-project/sglang/tree/3927a73a340708192fb41a4cc2f6c578df734ce0): [capture buckets](https://github.com/sgl-project/sglang/blob/3927a73a340708192fb41a4cc2f6c578df734ce0/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py), [full Graph backend](https://github.com/sgl-project/sglang/blob/3927a73a340708192fb41a4cc2f6c578df734ce0/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py), and [static buffer registry](https://github.com/sgl-project/sglang/blob/3927a73a340708192fb41a4cc2f6c578df734ce0/python/sglang/srt/model_executor/cuda_graph_buffer_registry.py).

## Findings

### Stream handoff is implemented

`Device.capture` records an event on the caller stream and each registered transfer stream.
The private capture stream waits for those events before recording.
It restores the caller stream after capture and rejects an active caller capture.
`tests/test_graph.py` covers these conditions.

### Graph residency is bounded for Qwen

`qwen38.model.Model` keeps at most eight decode workspaces.
It closes the oldest workspace before removing it.
Workspace keys include the native plan reuse fields.
This matches the fixed-bucket or bounded-cache design used by vLLM and SGLang.

### NCCL failure stays a fatal boundary

The collective peer polls host enqueue progress before finishing capture.
It records completion after replay and waits before reading outputs.
Pending NCCL capture errors use the fatal process path.
Generic Graph cleanup must not issue new CUDA or NCCL work after that error.

DeepSeek has no Graph collective path, so this contract is not yet exercised by its rank bridge.

### Cleanup diagnostics stay incomplete

`Device.graph` keeps the original body exception and marks the device unusable after unsafe cleanup.
The worker does not yet publish each synchronization or destruction failure in its diagnostics.
The next Graph adapter should attach cleanup status to the worker failure record before group shutdown.

## DeepSeek acceptance boundary

Before DeepSeek uses CUDA Graphs, the adapter must:

1. Prepare fixed buffers, native workspaces, and communicator state on each rank.
2. Prove the same shape, address, collective order, and execution mode on each rank.
3. Join caller and transfer streams before capture.
4. Keep graph owners and cache views alive through replay.
5. Poll NCCL enqueue completion before ending capture.
6. Record completion on the replay stream before reading outputs.
7. Leave the worker unusable after a pending NCCL error or unsafe cleanup.

Until these checks exist, DeepSeek stays Eager.
This audit makes no DeepSeek throughput claim.
