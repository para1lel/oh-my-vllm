# Torch Native Runtime 设计

日期: 2026-10-30

## 目标

本 repository 使用一个 Conda 环境. 使用支持 CUDA 13.4 runtime 的最新 PyTorch wheel. Python 模型, cache, worker 和 runtime 只向外暴露 `torch.Tensor`.

CUDA 调用边界继续使用 TVMFFI. 增加 Python `torch.library` namespace, 将 TVMFFI CUDA 函数适配为 Torch tensor.

## 目标范围

- 只保留一个安装文件: `requirements/development.txt`.
- 本 repository 的代码使用 `oh-my-vllm` Conda 环境.
- 目标为 Python 3.15.2 和带 CUDA 13.4 wheel 的 PyTorch 2.15 或更新版本.
- device allocation, view, upload, download, weight, cache, collective 和 model workspace 返回 Torch tensor.
- 所有 Graph capture, upload, replay 和销毁使用 Torch stream 和 Graph API.
- tensor 边界迁移期间保持 CUDA kernel 和现有数值契约不变.
- 保持所有权, alias, mutation, stream 顺序, Graph 地址和 paged storage 有效.

## 非目标

- 不使用 ATen kernel 替换 CUDA kernel.
- 不改变模型数学, quantization, frozen fixture, scheduler policy 或 ZMQ/msgpack 契约.
- 不把 NumPy 作为 runtime tensor 类型.
- repository test, service, native build 和 AgentX 不使用第二个环境.
- 保留明确 reference generation 和 vLLM/SGLang 运行作为文档规定的特殊情况.

## 约束

当前主机有 CUDA 13.1 和 13.2 toolkit, 没有 CUDA 13.4 toolkit. CUDA 13.4 PyTorch wheel 和 native compiler toolkit 是两个输入. Native build 必须记录两者 identity, 并在选择的 toolkit 不符合契约时失败.

当前 runtime 的 `Device.empty`, `scratch`, `zeros`, `view` 和 `upload` 返回 TVMFFI tensor. Upload buffer 在手工分配的 host memory 上创建 TVMFFI view. Graph manager 直接调用 CUDA Graph C API. 这些路径首先迁移.

Native C++ module 继续接收 TVMFFI `TensorView`.
Python adapter 只在 `torch.library` implementation 内通过 DLPack 将 Torch tensor 转成 TVMFFI view.
Adapter 在每个异步操作期间持有 Torch tensor.

第一个 environment task 从官方 CUDA 13.4 wheel index 固定 `torch==2.15.0+cu134`.
Native build task 使用 CUDA 13.4 toolkit.
它不混用 CUDA 13.2 headers 和 CUDA 13.4 runtime.
如果 toolkit 不可用, native rebuild 和 runtime acceptance 以明确的 environment error 停止.

## 方案

### 完整 native 重写

将所有 CUDA module 改为 Torch C++ extension, 并删除 TVMFFI tensor 参数. 该方案只有一个 binding system, 但会同时改变所有 kernel, build identity, error path 和 Graph test. 这不适合当前的数值和性能验收风险.

### 使用 TVMFFI CUDA 调用的 Torch 边界

保留 kernel implementation 和 TVMFFI ABI. 为 allocation, view, copy, upload, zero, collective 和 model kernel 增加 `oh_my_vllm` Torch operator. Python storage 和 Graph ownership 迁移到 Torch. 该方案保持 kernel 契约稳定, 同时明确 Python tensor 边界.

这是选定方案, 因为它满足 Torch-native 要求, 并将数值变化限制在 adapter layer.

### 只转换 model

将 model tensor 转为 Torch tensor, 但保留 device, upload, cache 和 Graph object 的 TVMFFI value. 该方案减少初始 diff, 但会在 inference engine 中保留两个 tensor ownership system. 它不满足请求的 runtime 契约.

## 设计

增加小型 runtime adapter module.
它管理 `torch.library.Library` definition 和 implementation.
每个 implementation 检查 device, dtype, shape, stride, contiguity 和 alias.
它只为 native call 在 TVMFFI 中进入当前 Torch CUDA stream.
Native result 使用 `torch.from_dlpack` 变成 Torch tensor.

Core schema 是 `native_view`, `native_copy`, `native_zero`, `native_upload` 和 `native_download`.
`native_view` 返回 source 的 alias.
`native_copy` 和 `native_zero` 将 destination 标记为 mutable.
Adapter 注册 CUDA 和 Meta behavior.
Tracing 需要 shape-only result 时, adapter 注册 Fake behavior.

`Device.empty`, `scratch`, `zeros`, `view`, `upload` 和 `download` 返回 Torch tensor.
View 保留 source tensor 作为 owner.
Scratch 或 paged tensor 通过 DLPack deleter 保留 native allocation owner.
Adapter 检查 result 的 device, dtype 和 storage 契约是否符合请求.

当 DLPack 创建 TVMFFI view 时, Torch input 是 owner.
`torch.from_dlpack` 消费 capsule 前, native producer 是 owner.
Async upload 和 transfer objects 在 CUDA events 完成前持有 source 和 destination tensors.
Paged tensor 在 private Torch-side object 中持有 native allocation owner.
Graph resource 在 graph close 完成前持有 captured tensors 和 pool.

Installed Torch 支持时, packed FP4 storage 使用 project Torch FP4 dtype.
Raw packed bytes 使用 `torch.uint8`, 并在 adapter boundary 带有 explicit logical dtype.
FP8 E4M3 和 E8M0 使用 Torch float8 dtypes.
Paged non-contiguous strides 保持在 native storage owner 中, 并通过 Torch view 提供.

将 runtime code 中的 NumPy upload input 替换为 CPU Torch tensor. Checkpoint reader 可以在文件解码时使用 NumPy, 但在 weight boundary 转换, 并让 model weight 保持 Torch tensor. 需要 array 的 development tool 在输出边界调用 `.cpu().numpy()`.

使用 `torch.cuda.current_stream()` 作为 caller stream. 用 `torch.cuda.CUDAGraph`, `torch.cuda.graph` 和 `torch.cuda.graph_pool_handle` 替换 raw CUDA Graph manager. Capture 前在相同 Torch stream 上 warmup. Capture 记录 Torch library operation. Replay 使用相同 stream 和固定 tensor address. Binding 改变前关闭旧 Torch Graph. Python runtime code 删除 raw `cudaGraph*` call.

在 transfer 契约允许时, 将 pinned upload storage 改为 Torch pinned CPU tensor.
保持现有 event 和 quiescence 规则.
Upload object 在 event 完成前持有 source Torch tensor 和 destination Torch tensor.

按以下顺序迁移 runtime module:

1. Device allocation, view, copy 和 Graph ownership.
2. Upload, transfer, metadata 和 sampling buffer.
3. Weight, model workspace, cache 和 paged storage.
4. Collective 和所有 model operator wrapper.
5. Development tool 和 test fixture.

每个阶段完成后删除 tensor operation 对 `device.module` 的 Python direct call. 只在 adapter, native build identity 和 low-level stream bridge 中保留 direct TVMFFI.

## 验证

Environment check 记录 Python, Torch, CUDA runtime, CUDA toolkit, TVMFFI, compiler 和 native module identity. 它确认所有 repository command 使用 `OH_MY_VLLM_CONDA_PREFIX`.

Runtime test 检查 public allocation, view, upload, weight, cache tensor 和 operator result 都是 `torch.Tensor`. Test 覆盖 DLPack ownership, alias, non-contiguous rejection, dtype/device contract, asynchronous upload lifetime, paged mapping 和 multi-rank collective.

Graph test 使用 Torch Graph capture 和 replay. 它检查 changed input, fixed address, stream handoff, Graph replacement, capture failure cleanup 和 close. Python test 不得调用 raw CUDA Graph C API.

每个迁移阶段完成后运行 CPU 和 GPU tests.
Compiler environment ready 后运行 native build checks.
Operator migration 后运行 Qwen regression 和 DeepSeek numerical acceptance.
Service migration 后运行 HTTP acceptance 和 AgentX.
保留现有 frozen reference 和 performance accounting.
