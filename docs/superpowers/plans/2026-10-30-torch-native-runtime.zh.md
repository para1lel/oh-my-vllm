# Torch Native Runtime Implementation Plan

> **执行说明** - Use superpowers: executing-plans to implement this plan task-by-task.

**目标:**  Python runtime storage 和 graph ownership 使用 `torch.Tensor` 与 Torch CUDA Graph APIs. TVMFFI 继续作为 CUDA kernel call boundary.

**架构:**  增加 focused `torch.library` adapter. 只有 native call 期间将 Torch tensors 转为 TVMFFI views. 按 device allocation, uploads, cache, operators 和 development tools 的顺序迁移. 每个 task 都有 Torch ownership 和 stream test.

**技术栈:**  PyTorch 2.15.0+cu134, `torch.library`, `torch.from_dlpack`, TVMFFI, CUDA C++ kernels, `torch.cuda.CUDAGraph`, `torch.cuda.graph`, NCCL.

**规格:**  `docs/superpowers/specs/2026-10-30-torch-native-runtime-design.md`

## 全局约束

- Python model, cache, worker 和 runtime 只暴露 `torch.Tensor`.
- CUDA call boundary 使用 TVMFFI.
- 所有 graph capture, upload, replay 和 destruction 使用 Torch stream 和 graph APIs.
- 不使用 ATen kernels 替换 CUDA kernels.
- 不改变 model mathematics, quantization, frozen fixtures, scheduler policy 和 ZMQ/msgpack contracts.
- 保持 ownership, aliasing, mutation, stream order, graph addresses 和 paged storage 有效.
- Direct TVMFFI 只在 adapter, native identity code 和 low-level stream bridge 中出现.

## Review Focus

- Torch view 必须保持 source allocation 和 aliasing. Task 2 测试 source deletion 和 mutation.
- DLPack conversion 不得在 async work 完成前释放 native storage. Task 3 测试 delayed destruction.
- Non-contiguous, wrong-device 和 wrong-dtype tensors 必须在 native launch 前失败. Task 2 测试这些 contracts.
- Captured operation 必须在当前 Torch stream replay, 使用 fixed addresses 和 changed inputs. Task 4 测试 replay 和 replacement.
- Paged tensor 必须保持 native owner 和 stream order. Task 5 测试 mapping 和 cleanup.

### Task 1: Add the Torch/TVMFFI adapter namespace

**文件:**
- Create: `python/oh_my_vllm/operators/runtime/torch_adapter.py`
- Modify: `python/oh_my_vllm/operators/runtime/__init__.py`
- Test: `tests/test_torch_adapter.py`

**接口:**
- `torch_adapter.native_tensor(value)` 返回 contiguous Torch tensor 的 TVMFFI view, 不做 copy.
- `torch_adapter.torch_tensor(value)` 从 native DLPack producer 返回 Torch tensor, 并保持 producer owner.
- `torch_adapter.current_stream(device)` 返回当前 Torch CUDA stream 的 raw stream.
- `oh_my_vllm` library namespace 注册 adapter operations, 不暴露 TVMFFI objects.

- [ ] 写 CPU contract tests, 覆盖 dtype, device, contiguity 和 unsupported inputs.
- [ ] 运行 tests, 检查 adapter missing 或 unimplemented operation failure.
- [ ] 实现 DLPack conversion 和 stream bridge, 明确 owner retention.
- [ ] 增加 native allocation, view, copy, zero 和 download 的 Torch library schemas.
- [ ] 在 CPU 和 GPU 运行 adapter tests. 每个 public result 都必须是 `torch.Tensor`.

### Task 2: Migrate Device allocation, views, copies, and downloads

**文件:**
- Modify: `python/oh_my_vllm/operators/runtime/device.py`
- Modify: `python/oh_my_vllm/operators/runtime/storage.py`
- Test: `tests/test_device_tensors.py`

**接口:**
- `Device.empty`, `Device.scratch`, `Device.zeros`, `Device.view` 和 `Device.upload` 返回 Torch tensors.
- `Device.download` 返回 CPU Torch tensor.
- Python callers 不使用 `Device.module` 做 tensor allocation, view, copy 或 zeroing.

- [ ] 增加 Torch result types, aliasing, offsets, dtype, device 和 non-contiguous rejection tests.
- [ ] 运行 tests, 检查 current TVMFFI result-type failures.
- [ ] 使用 `torch.library` adapter methods, 保留 native allocation owners.
- [ ] 在 device boundary 将 NumPy upload arguments 替换为 CPU Torch tensors.
- [ ] 运行 device, constraints, logical-row 和 storage tests.

### Task 3: Migrate pinned uploads and transfers

**文件:**
- Modify: `python/oh_my_vllm/operators/runtime/uploads.py`
- Modify: `python/oh_my_vllm/operators/runtime/transfers.py`
- Modify: `python/oh_my_vllm/worker/metadata.py`
- Test: `tests/test_execution.py`
- Test: `tests/test_offload_worker.py`

**接口:**
- Upload buffers 在 CUDA events 完成前保持 Torch CPU tensors 和 Torch CUDA destinations.
- `Uploads.copy` 接受 CPU Torch tensors, 拒绝 NumPy runtime buffers.
- Transfer engine 在 transfer 完成前保持 Torch tensor owners.

- [ ] 增加 source reference deletion before event completion 的 failing lifetime tests.
- [ ] 运行 tests, 检查 current upload implementation 的 owner failure.
- [ ] 在 pool contract 允许时使用 pinned Torch CPU storage.
- [ ] 保持 event, quiescence 和 stream rules.
- [ ] 在 CPU 和 GPU 运行 execution 和 offload worker tests.

### Task 4: Replace Python CUDA Graph management with Torch Graph APIs

**文件:**
- Modify: `python/oh_my_vllm/operators/runtime/device.py`
- Modify: `python/oh_my_vllm/models/qwen38/inference.py`
- Modify: graph-owning Qwen and shared workspace modules found by `rg 'device\.graph|device\.capture'`
- Test: `tests/test_graph.py`
- Test: `tests/test_execution.py`

**接口:**
- `Device.graph(operation)` owns a `torch.cuda.CUDAGraph` and returns replay/close resource.
- Capture 使用 `torch.cuda.graph(graph, pool=..., stream=...)`.
- Replay 调用 `graph.replay()` on current Torch stream.
- Close 在 stream safe 后释放 Torch graph resources.

- [ ] 增加 Torch Graph capture, replay, changed inputs, replacement 和 cleanup failure tests.
- [ ] 运行 tests, 检查 current raw-CUDA event expectations.
- [ ] 使用 Torch APIs 实现 warmup, graph pool, capture, replay 和 close.
- [ ] 删除 Python `cudaGraphInstantiate*`, `cudaGraphUpload`, `cudaGraphLaunch` 和 `cudaGraph*Destroy` calls.
- [ ] 运行 graph, logical transform, logical rows, paged storage 和 DeepSeek graph tests.

### Task 5: Migrate paged storage and model cache ownership

**文件:**
- Modify: `python/oh_my_vllm/operators/runtime/paged.py`
- Modify: `python/oh_my_vllm/models/qwen38/cache.py`
- Modify: `python/oh_my_vllm/models/qwen38/offload.py`
- Modify: `python/oh_my_vllm/models/deepseek_v41/cache.py` only where public tensor type is not Torch
- Test: `tests/test_paged_storage.py`

**接口:**
- `allocate_paged` 返回 Torch CUDA tensor, 并通过 private Torch-side owner object 连接 native owner.
- Cache state 和 views 是 Torch tensors.
- Slot commit 和 release 使用 current Torch stream.

- [ ] 增加 Torch type, owner retention, view aliasing, commit, release 和 mapped-byte accounting tests.
- [ ] 运行 tests, 检查 owner 或 stream failures.
- [ ] 从 `paged.py` 删除 direct TVMFFI tensor exposure, 只保留 low-level bridge.
- [ ] 运行 paged, cache, prefix reuse 和 preemption tests.

### Task 6: Migrate model weights, workspaces, collectives, and operator callers

**文件:**
- Modify: `python/oh_my_vllm/models/qwen38/weights.py`
- Modify: `python/oh_my_vllm/operators/runtime/collectives.py`
- Modify: operator workspace modules under `python/oh_my_vllm/operators/**/workspace.py`
- Modify: Qwen and DeepSeek model callers that use `device.module` tensor methods
- Test: relevant existing operator, Qwen, and DeepSeek tests

**接口:**
- Weight containers, workspace fields, collective buffers 和 operator results 是 Torch tensors.
- Collective preparation 接受 Torch tensors, 并在 completion 前保持 owners.
- Native operator wrappers 只在 adapter 内接收 TVMFFI views.

- [ ] 在 existing operator tests 增加 Torch tensor types 和 current-stream assertions.
- [ ] 在修改 callers 前运行 focused tests.
- [ ] 在 weight boundary 转换 checkpoint arrays, 替换 direct module tensor calls.
- [ ] 适配 collectives 和 operator workspaces, 不改变 numerical layouts.
- [ ] 运行 full CPU/GPU unit suite 和 frozen DeepSeek numerical checks.

### Task 7: Migrate development tools and remove obsolete Python FFI tensor access

**文件:**
- Modify: `development/model/capture.py`
- Modify: `development/operators/run.py`
- Modify: `development/operators/validation.py`
- Modify: `development/operators/storage.py`
- Modify: tests that assert raw TVMFFI tensor values

**接口:**
- Development tools 只在 file 或 report boundary 转换为 NumPy.
- Adapter 之外的 repository Python modules 不使用 `tvm_ffi.from_dlpack` 作为 runtime tensors.
- Python runtime 和 tests 不使用 raw CUDA Graph C API 管理 graph lifecycle.

- [ ] 搜索并分类 remaining direct TVMFFI tensor 和 raw Graph calls.
- [ ] 使用 Torch adapter calls 替换 runtime tensor calls.
- [ ] 保留 native contracts 所需的 low-level stream 和 collective ABI calls.
- [ ] 运行 source policy checks 和 full tests.

## Delivery

- 在 unified Conda environment 运行 native build, CPU tests, GPU tests, Qwen regression, DeepSeek numerical acceptance, HTTP acceptance 和 AgentX.
- 每次 delivery 前获取 independent code review.
- 只 commit active task 的 files. Review 后 push `nightly`.
- 停止 task 启动的 workers 和 temporary services.
