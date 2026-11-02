# Torch Native Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Python runtime storage and graph ownership use `torch.Tensor` and Torch CUDA Graph APIs while keeping TVMFFI as the CUDA kernel call boundary.

**Architecture:** Add a focused `torch.library` adapter that converts Torch tensors to TVMFFI views only during native calls. Migrate storage and graph ownership from the bottom up: device allocation, uploads, cache, operators, then development tools. Each task keeps the existing native C++ ABI and has a Torch ownership and stream test.

**Tech Stack:** PyTorch 2.15.0+cu134, `torch.library`, `torch.from_dlpack`, TVMFFI, CUDA C++ kernels, `torch.cuda.CUDAGraph`, `torch.cuda.graph`, NCCL.

**Spec:** `docs/superpowers/specs/2026-10-30-torch-native-runtime-design.md`

## Global Constraints

- Make `torch.Tensor` the only tensor type available by Python model, cache, worker, and runtime code.
- Keep TVMFFI as the CUDA call boundary.
- Use Torch stream and graph APIs for all graph capture, upload, replay, and destruction.
- Do not replace CUDA kernels with ATen kernels.
- Keep model mathematics, quantization, frozen fixtures, scheduler policy, and ZMQ/msgpack contracts unchanged.
- Keep ownership, aliasing, mutation, stream order, graph addresses, and paged storage valid.
- Keep direct TVMFFI use only in the adapter, native identity code, and low-level stream bridge.

## Review Focus

- A Torch view must keep its source allocation alive and keep exact aliasing. test source deletion and mutation in Task 2.
- A DLPack conversion must not free native storage before asynchronous work completes. test event-delayed destruction in Task 3.
- A non-contiguous, wrong-device, or wrong-dtype tensor must fail before a native launch. test each contract in Task 2.
- A captured operation must replay on the current Torch stream with fixed addresses and changed input values. test graph replay and replacement in Task 4.
- A paged tensor must keep its native owner through Torch views and release slots in stream order. test mapping and cleanup in Task 5.

### Task 1: Add the Torch/TVMFFI adapter namespace

**Files:**

- Create: `python/oh_my_vllm/operators/runtime/torch_adapter.py`
- Modify: `python/oh_my_vllm/operators/runtime/__init__.py`
- Test: `tests/test_torch_adapter.py`

**Interfaces:**

- `torch_adapter.native_tensor(value)` returns a TVMFFI tensor view for a contiguous Torch tensor without copying.
- `torch_adapter.torch_tensor(value)` returns a Torch tensor from a native DLPack producer and holds the producer owner.
- `torch_adapter.current_stream(device)` returns the raw stream for the current Torch CUDA stream.
- The `oh_my_vllm` library namespace registers adapter operations and does not make available TVMFFI objects.

- [ ] Write CPU contract tests for dtype, device, contiguity, and unsupported input failures.
- [ ] Run them and check the adapter is missing or rejects the unimplemented operation.
- [ ] Implement DLPack conversion and stream bridging with explicit owner retention.
- [ ] Add Torch library schemas for native allocation, view, copy, zero, and download operations.
- [ ] Run adapter tests on CPU and GPU and assert each public result is `torch.Tensor`.

### Task 2: Migrate Device allocation, views, copies, and downloads

**Files:**

- Modify: `python/oh_my_vllm/operators/runtime/device.py`
- Modify: `python/oh_my_vllm/operators/runtime/storage.py`
- Test: `tests/test_device_tensors.py`

**Interfaces:**

- `Device.empty`, `Device.scratch`, `Device.zeros`, `Device.view`, and `Device.upload` return Torch tensors.
- `Device.download` returns a CPU Torch tensor.
- `Device.module` is not used for tensor allocation, views, copies, or zeroing from Python callers.

- [ ] Add tests for Torch result types, aliasing, offsets, dtype, device, and non-contiguous rejection.
- [ ] Run the tests and observe the current TVMFFI result-type failures.
- [ ] Route methods through `torch.library` adapter operations and keep native allocation owners.
- [ ] Replace NumPy upload arguments with CPU Torch tensors at the device boundary.
- [ ] Run device, constraints, logical-row, and storage tests.

### Task 3: Migrate pinned uploads and transfers

**Files:**

- Modify: `python/oh_my_vllm/operators/runtime/uploads.py`
- Modify: `python/oh_my_vllm/operators/runtime/transfers.py`
- Modify: `python/oh_my_vllm/worker/metadata.py`
- Test: `tests/test_execution.py`
- Test: `tests/test_offload_worker.py`

**Interfaces:**

- Upload buffers hold Torch CPU tensors and Torch CUDA destinations until their CUDA events complete.
- `Uploads.copy` accepts CPU Torch tensors and rejects NumPy runtime buffers.
- Transfer engine submissions hold Torch tensor owners for the complete transfer.

- [ ] Add failing lifetime tests that delete source references before event completion.
- [ ] Run the tests and check the current upload implementation cannot satisfy Torch ownership.
- [ ] Use pinned Torch CPU storage where the existing pool contract permits it.
- [ ] Keep event, quiescence, and stream rules unchanged.
- [ ] Run execution and offload worker tests on CPU and GPU.

### Task 4: Replace Python CUDA Graph management with Torch Graph APIs

**Files:**

- Modify: `python/oh_my_vllm/operators/runtime/device.py`
- Modify: `python/oh_my_vllm/models/qwen38/inference.py`
- Modify: graph-owning Qwen and shared workspace modules found by `rg 'device\.graph|device\.capture'`
- Test: `tests/test_graph.py`
- Test: `tests/test_execution.py`

**Interfaces:**

- `Device.graph(operation)` owns a `torch.cuda.CUDAGraph` and returns a replay/close resource.
- Capture uses `torch.cuda.graph(graph, pool=..., stream=...)`.
- Replay calls `graph.replay()` on the current Torch stream.
- Close drops Torch graph resources only after the stream is safe.

- [ ] Add tests using Torch Graph objects for capture, replay, changed inputs, replacement, and cleanup failure.
- [ ] Run the tests and check the current raw-CUDA event expectations fail.
- [ ] Implement warmup, graph pool selection, capture, replay, and close with Torch APIs.
- [ ] Remove Python `cudaGraphInstantiate*`, `cudaGraphUpload`, `cudaGraphLaunch`, and `cudaGraph*Destroy` calls.
- [ ] Run all graph, logical transform, logical rows, paged storage, and DeepSeek graph tests.

### Task 5: Migrate paged storage and model cache ownership

**Files:**

- Modify: `python/oh_my_vllm/operators/runtime/paged.py`
- Modify: `python/oh_my_vllm/models/qwen38/cache.py`
- Modify: `python/oh_my_vllm/models/qwen38/offload.py`
- Modify: `python/oh_my_vllm/models/deepseek_v41/cache.py` only where the public tensor type is not already Torch
- Test: `tests/test_paged_storage.py`

**Interfaces:**

- `allocate_paged` returns a Torch CUDA tensor with a native owner attached through a private Torch-side owner object.
- Cache state and views are Torch tensors.
- Slot commit and release use the current Torch stream through the adapter.

- [ ] Add tests for Torch type, owner retention, view aliasing, commit, release, and mapped-byte accounting.
- [ ] Run the tests and observe any owner or stream failures.
- [ ] Remove direct TVMFFI tensor access from `paged.py` and keep only the low-level bridge.
- [ ] Run paged, cache, prefix reuse, and preemption tests.

### Task 6: Migrate model weights, workspaces, collectives, and operator callers

**Files:**

- Modify: `python/oh_my_vllm/models/qwen38/weights.py`
- Modify: `python/oh_my_vllm/operators/runtime/collectives.py`
- Modify: operator workspace modules with `python/oh_my_vllm/operators/**/workspace.py`
- Modify: Qwen and DeepSeek model callers that use `device.module` tensor methods
- Test: relevant existing operator, Qwen, and DeepSeek tests

**Interfaces:**

- Weight containers, workspace fields, collective buffers, and operator results are Torch tensors.
- Collective preparation accepts Torch tensors and keeps their owners through completion.
- Native operator wrappers receive TVMFFI views only inside the adapter.

- [ ] Add assertions to the existing operator tests for Torch tensor types and current-stream use.
- [ ] Run the focused tests before changing callers.
- [ ] Convert checkpoint arrays at the weight boundary and replace direct module tensor calls.
- [ ] Adapt collectives and operator workspaces without changing numerical layouts.
- [ ] Run the full CPU/GPU unit suite and all frozen DeepSeek numerical checks.

### Task 7: Migrate development tools and remove obsolete Python FFI tensor access

**Files:**

- Modify: `development/model/capture.py`
- Modify: `development/operators/run.py`
- Modify: `development/operators/validation.py`
- Modify: `development/operators/storage.py`
- Modify: tests that assert raw TVMFFI tensor values

**Interfaces:**

- Development tools convert to NumPy only at file or report boundaries.
- No repository Python module not in the adapter uses `tvm_ffi.from_dlpack` for runtime tensors.
- No Python runtime or test uses raw CUDA Graph C API for graph lifecycle.

- [ ] Search and classify staying direct TVMFFI tensor and raw Graph calls.
- [ ] Replace runtime tensor calls with Torch adapter calls.
- [ ] Keep low-level stream and collective ABI calls where required by native contracts.
- [ ] Run source policy checks and full tests.

## Delivery

- Run native build and unit tests in the unified Conda environment.
- Run Qwen and DeepSeek numerical acceptance.
- Run HTTP acceptance and AgentX.
- Get an independent code review before each delivery.
- Commit only files for the active task and push `nightly` after each delivery.
- Stop all workers and temporary services started by the task.
