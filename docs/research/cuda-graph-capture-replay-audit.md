# CUDA Graph Capture and Replay Audit

Audit date: 2026-10-27. Repository revision: `nightly` at `93a96f7d20d82d4b9197833e517232a0f307e030`.

Scope: read-only audit of the current capture, replay, collective, and shutdown paths. This report does not change runtime code. It does not add the session-only operator performance objective to repository gates.

## Decision

The current Qwen path meets the basic single-worker CUDA Graph rules. It keeps graph buffers alive, updates metadata before replay, keeps RNG state in device buffers, and closes graphs before their owners. The current tests cover these properties.

The generic graph contract is incomplete. The required changes are:

1. Join the private capture stream to the caller and transfer streams before capture.
2. Bound graph workspace residency and close an evicted graph before releasing its storage.
3. Keep the NCCL pending-error process exit rule for all future multi-rank model paths.

Fixed capture buckets, a shared graph memory pool, output buffer reuse, graph upload, and narrower release synchronization are optional performance work. They can improve latency and memory use. They are not correctness gates for this audit.

## Pinned sources

The audit uses these primary sources:

- CUDA 13.1 [stream capture API](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/group__CUDART__STREAM.html), [graph API](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/group__CUDART__GRAPH.html), and [stream capture guide](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture).
- NCCL commit [`3e46c089f4ae4732722a419eb6096aead7e47a30`](https://github.com/NVIDIA/nccl/tree/3e46c089f4ae4732722a419eb6096aead7e47a30): [CUDA Graph use](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/docs/userguide/source/usage/cudagraph.rst) and [communicator progress](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/docs/userguide/source/api/comms.rst).
- vLLM commit [`7c713621f3e67bd8579d320571e06cefdeab30ed`](https://github.com/vllm-project/vllm/tree/7c713621f3e67bd8579d320571e06cefdeab30ed): [graph manager](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/v1/worker/gpu/cudagraph_utils.py), [graph wrapper](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/compilation/cuda_graph.py), and [distributed capture stream](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/distributed/parallel_state.py).
- SGLang commit [`3927a73a340708192fb41a4cc2f6c578df734ce0`](https://github.com/sgl-project/sglang/tree/3927a73a340708192fb41a4cc2f6c578df734ce0): [capture buckets](https://github.com/sgl-project/sglang/blob/3927a73a340708192fb41a4cc2f6c578df734ce0/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py), [full graph backend](https://github.com/sgl-project/sglang/blob/3927a73a340708192fb41a4cc2f6c578df734ce0/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py), and [static buffer registry](https://github.com/sgl-project/sglang/blob/3927a73a340708192fb41a4cc2f6c578df734ce0/python/sglang/srt/model_executor/cuda_graph_buffer_registry.py).

## Correctness findings

### P1: The generic capture helper misses stream handoff

[`Device.capture`](../../python/oh_my_vllm/operators/runtime/device.py) creates a new nonblocking stream. It does not make that stream wait for the caller stream or registered transfer streams.

The model workspace and `GraphOperation` wait for the current compute stream. Sampling and acceptance can capture after caller-stream copies without that wait. The helper itself has no safe-stream precondition.

CUDA stream capture records work instead of executing it. Cross-stream dependencies still use event or stream-wait edges. CUDA documents this rule in [cross-stream capture](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-programming-guide/index.html#cross-stream-dependencies). vLLM joins its capture stream to the current stream before capture in [the distributed capture context](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/distributed/parallel_state.py#L712-L724).

This is a generic correctness defect. The current single-stream operators usually avoid a visible error because the first replay is queued after caller-stream copies. A future operator or transfer path can read state during capture setup without that ordering.

Smallest change: record an event on the caller stream, wait for it on the capture stream, and cover all registered producer streams. Reject capture when the caller stream is already capturing.

### P1: Graph workspace residency is unbounded

[`Model.forward`](../../python/oh_my_vllm/models/qwen38/model.py) stores each decode workspace in `self.decode`. The key contains exact rows, context, query, table, and execution-plan values. The map has no bound and no eviction. Speculation has the same pattern in [`context_decode`](../../python/oh_my_vllm/models/qwen38/speculation.py).

Each workspace owns device scratch, metadata, operator workspaces, output storage, and a graph executable. A request mix with many context and shape combinations can hold enough graphs to consume device memory. This is a resource correctness defect, not only a latency issue.

vLLM limits capture variants to configured descriptors and provides [`release_graphs`](https://github.com/vllm-project/vllm/blob/7c713621f3e67bd8579d320571e06cefdeab30ed/vllm/v1/worker/gpu/cudagraph_utils.py#L401-L411). SGLang selects fixed capture buckets in [`get_batch_sizes_to_capture`](https://github.com/sgl-project/sglang/blob/3927a73a340708192fb41a4cc2f6c578df734ce0/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py#L65-L110).

Smallest change: use a bounded set of capture buckets or a bounded LRU. Close the graph context before removing a workspace. Keep the key complete for shape, capability, binding, and execution mode.

### P1 integration blocker: NCCL failure cleanup must use a fatal path

The NCCL path follows the required progress order. [`collective_peer.py`](../../development/operators/collective_peer.py) polls host enqueue progress, finishes capture only after NCCL reports success, records completion after replay, and waits before reading outputs. The native owner rejects CUDA cleanup while a capture enqueue is pending in [`collectives.cu`](../../python/oh_my_vllm/operators/runtime/cuda/collectives.cu).

On a pending NCCL capture error, the peer calls `os._exit` before unwinding [`Device.capture`](../../python/oh_my_vllm/operators/runtime/device.py). This matches NCCL's rule: after `ncclInProgress`, callers must wait for success before issuing another NCCL call or CUDA kernel. NCCL also sets graph capture and graph launch to be uniform across all participating ranks. See [NCCL communicator progress](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/docs/userguide/source/api/comms.rst#L221-L236) and [NCCL CUDA Graph requirements](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/docs/userguide/source/usage/cudagraph.rst#L13-L18).

The current probe is safe because it enforces this fatal path. The generic graph API is unsafe for a future DSV4.1 collective adapter unless the adapter uses the same rule. Keep this rule at the collective seam. Do not make generic graph cleanup call CUDA after a pending NCCL error.

### P2: Cleanup errors can be hidden

On an exception from the graph body, [`Device.graph`](../../python/oh_my_vllm/operators/runtime/device.py) suppresses errors from synchronization and executable destruction. The original exception is re-raised. This keeps the original failure but can hide a graph leak or a failed owner release.

Normal closure is ordered correctly: synchronize, destroy the executable, then release graph owners. The Qwen worker closes offload, model graphs, sampler graphs, and the device in that order in [`Worker.close`](../../python/oh_my_vllm/models/qwen38/inference.py). The required change is to keep cleanup failure in worker diagnostics and keep the worker unusable after an unsafe cleanup failure.

## Contracts that pass

| Area | Assessment |
|---|---|
| Preparation and warmup | Operator workspaces allocate before capture. Qwen runs operator warmup when the path uses it. Sampling and acceptance capture kernels without executing RNG updates, which matches CUDA capture semantics. |
| Fixed addresses | Workspace and operator leases keep weights, cache views, metadata, scratch, and outputs alive. Binding changes close the old graph first in [`Workspace.bind_media`](../../python/oh_my_vllm/models/qwen38/model.py). |
| Metadata | [`Metadata.update`](../../python/oh_my_vllm/worker/metadata.py) copies into stable device arenas. Pinned upload buffers snapshot CPU inputs and record an event on the caller stream in [`uploads.py`](../../python/oh_my_vllm/operators/runtime/uploads.py). |
| Graph keys | Plan keys include the current Qwen shape and execution-policy fields. Sampling keys include greedy, compact-history, and token-selection modes in [`sampling.py`](../../python/oh_my_vllm/models/qwen38/sampling.py). The lifecycle bound is the missing part. |
| RNG | Sampling uploads seed and offset before replay. Sampling and MTP kernels update device offsets. Host code reads the updated counter and supplies it to the next replay. See [`sample.cu`](../../python/oh_my_vllm/operators/output/cuda/sample.cu) and [`accept.cuh`](../../python/oh_my_vllm/operators/mtp/cuda/accept.cuh). |
| Output lifetime | Graph owners hold output storage. Host downloads synchronize the caller stream before CPU access. Existing tests verify executable destruction before operator owner release. |
| Replay stream | Replay uses the current raw CUDA stream in [`Device.graph`](../../python/oh_my_vllm/operators/runtime/device.py). Metadata uploads use that same stream, so uploads precede graph launch. |
| Normal shutdown | The Qwen path and the NCCL probe wait for graph work before releasing buffers. NCCL abort is followed by polling to the aborted state in [`collective_peer.py`](../../development/operators/collective_peer.py). |

The distributed graph implementation is a development probe. The current worker has no DSV4.1 multi-rank adapter. This audit therefore cannot certify DSV4.1 rank capture until that adapter uses the existing rank protocol.

## Optional performance ideas

- Capture fixed shape buckets and pad runtime batches to the smallest compatible bucket. This reduces graph count and capture churn.
- Use one graph memory pool and reuse output buffers. vLLM and SGLang use graph pools and stable output storage.
- Upload the executable graph before its first production replay. CUDA otherwise uploads it during the first host launch.
- Replace device-wide synchronization in [`release_graph`](../../python/oh_my_vllm/operators/runtime/device.py) with a graph-stream completion event when the owner contract permits it.
- Use graph update or executable deduplication only after the fixed-address and metadata contracts are tested.
- Use two warmups and an all-rank barrier for model adapters with lazy kernel, communication, or metadata setup. The current Qwen operator contracts intentionally use narrower warmup rules.

Measure these ideas before use. They do not belong in repository performance gates for this ticket.

## Smallest required changes and test boundaries

1. Add capture-stream handoff and an active-capture check. Test it at the `Device.capture` seam with a fake CUDA runtime. Check event record, capture-stream wait, caller-stream restoration, and cleanup after failure.
2. Add a bounded graph cache. Test more distinct decode keys than the bound. Check that eviction closes the executable before scratch and owner release. Check that kept keys replay updated metadata.
3. Keep the NCCL fatal pending-error path. Test the public collective seam for identical rank mode, plan digest, host-only progress, completion fencing, graph-owner release, and abort polling.
4. Keep cleanup failures in worker diagnostics. Test graph destruction failure after a body failure and verify that later execution is disabled.

Existing public boundaries are [`tests/test_execution.py`](../../tests/test_execution.py), [`tests/test_offload_worker.py`](../../tests/test_offload_worker.py), and [`tests/test_collectives.py`](../../tests/test_collectives.py). This document-only change adds no tests and changes no runtime code.

## Limits

This report is a contract review. It has no GPU benchmark and makes no SOTA claim. It does not change numerical limits, add an external operator dependency, or add a performance gate. The next implementation ticket must resolve the P1 items before DSV4.1 graph execution is accepted.
