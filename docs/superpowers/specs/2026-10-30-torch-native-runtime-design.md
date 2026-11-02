# Torch Native Runtime Design

Date: 2026-10-30

## Intent

Use one Conda environment for this repository. Use the latest supported PyTorch wheel with CUDA 13.4 runtime support. Make `torch.Tensor` the only tensor type available by Python model, cache, worker, and runtime code.

Keep TVMFFI as the CUDA call boundary. Add a Python `torch.library` namespace that adapts TVMFFI CUDA functions to Torch tensors.

## Goals

- Keep one install file: `requirements/development.txt`.
- Run repository code with the `oh-my-vllm` Conda environment.
- Target Python 3.15.2 and PyTorch 2.15 or newer with the CUDA 13.4 wheel.
- Return Torch tensors from device allocation, views, uploads, downloads, weights, cache, collectives, and model workspaces.
- Use Torch stream and graph APIs for all graph capture, upload, replay, and destruction.
- Keep CUDA kernels and existing numerical contracts unchanged while the tensor boundary moves.
- Keep ownership, aliasing, mutation, stream order, graph addresses, and paged storage valid.

## Non-goals

- Do not replace the CUDA kernels with ATen kernels.
- Do not change model mathematics, quantization, frozen fixtures, scheduler policy, or ZMQ/msgpack contracts.
- Do not make NumPy the runtime tensor type.
- Do not use a second environment for repository tests, service execution, native builds, or AgentX.
- Keep explicit reference generation and vLLM/SGLang runs as documented special cases.

## Constraints

The current host has CUDA 13.1 and 13.2 toolkits, but no CUDA 13.4 toolkit. The CUDA 13.4 PyTorch wheel and the native compiler toolkit are separate inputs. Native builds must record the two identities and fail when the selected toolkit does not satisfy the build contract.

The current runtime returns TVMFFI tensors from `Device.empty`, `scratch`, `zeros`, `view`, and `upload`. Upload buffers create TVMFFI views over manually allocated host memory. The graph manager calls CUDA Graph C APIs directly. These paths call for the first migration.

The native C++ modules continue to accept TVMFFI `TensorView` values.
Python adapters convert Torch tensors to TVMFFI views with DLPack only inside `torch.library` implementations.
The adapter holds the Torch tensor through each asynchronous operation.

The first environment task pins `torch==2.15.0+cu134` from the official CUDA 13.4 wheel index.
The native build task uses a CUDA 13.4 toolkit.
It does not mix CUDA 13.2 headers with a CUDA 13.4 runtime.
If the toolkit is not available, native rebuild and runtime acceptance stop with an explicit environment error.

## Approaches

### Full native rewrite

Change each CUDA module to a Torch C++ extension and remove TVMFFI tensor arguments. This gives one binding system, but changes all kernels, build identities, error paths, and graph tests at once. It does not fit the current numerical and performance acceptance risk.

### Torch boundary with TVMFFI CUDA calls

Keep kernel implementations and their TVMFFI ABI. Add `oh_my_vllm` Torch operators for allocation, views, copies, uploads, zeroing, collectives, and model kernels. Move Python storage and graph ownership to Torch. This keeps the kernel contract stable while making the Python tensor boundary explicit.

This is the selected approach because it satisfies the Torch-native requirement and limits numerical change to the adapter layer.

### Model-only conversion

Convert model tensors to Torch tensors while leaving device, upload, cache, and graph objects as TVMFFI values. This reduces the first diff but leaves two tensor ownership systems in the inference engine. It does not meet the requested runtime contract.

## Design

Create a small runtime adapter module.
It owns the `torch.library.Library` definitions and implementations.
Each implementation validates device, dtype, shape, stride, contiguity, and aliasing.
It enters the current Torch CUDA stream through TVMFFI only for the native call.
Native results become Torch tensors with `torch.from_dlpack`.

The core schemas are `native_view`, `native_copy`, `native_zero`, `native_upload`, and `native_download`.
`native_view` returns an alias of its source.
`native_copy` and `native_zero` mark their destination as mutable.
The adapter registers CUDA and Meta behavior.
It registers Fake behavior when a shape-only result is required by tracing.

`Device.empty`, `scratch`, `zeros`, `view`, `upload`, and `download` return Torch tensors.
A view keeps its source tensor as an owner.
A scratch or paged tensor keeps the native allocation owner through the DLPack deleter.
The adapter rejects a result whose device, dtype, or storage contract differs from the requested value.

Torch inputs stay the owner when DLPack creates a TVMFFI view.
The native producer stays the owner until `torch.from_dlpack` consumes its capsule.
Asynchronous upload and transfer objects hold source and destination tensors until their CUDA events complete.
Paged tensors hold the native allocation owner in a private Torch-side object.
Graph resources hold captured tensors and their pool until graph close completes.

Packed FP4 storage uses the project Torch FP4 dtype when the installed Torch version supports it.
Raw packed bytes use `torch.uint8` with an explicit logical dtype at the adapter boundary.
FP8 E4M3 and E8M0 use their Torch float8 dtypes.
Paged non-contiguous strides stay in the native storage owner and are available through Torch views.

Replace NumPy upload inputs in runtime code with CPU Torch tensors. Checkpoint readers may use NumPy for file decoding, but convert at the weight boundary and keep model weights as Torch tensors. Development tools that call for arrays call `.cpu().numpy()` at their output boundary.

Use `torch.cuda.current_stream()` for the caller stream.
Replace the raw CUDA Graph manager with `torch.cuda.CUDAGraph`, `torch.cuda.graph`, and `torch.cuda.graph_pool_handle`.
Warmup runs on the same Torch stream before capture.
Capture records the Torch library operations.

Replay uses the same stream and fixed tensor addresses.
Graph replacement closes the old Torch graph before changing bindings.
Raw `cudaGraph*` calls are removed from Python runtime code.

Move pinned upload storage to Torch CPU tensors with pinned memory where the transfer contract permits.
Keep the existing event and quiescence rules.
The upload object holds the source Torch tensor and the destination Torch tensor until the event completes.

Migrate runtime modules in this order:

1. Device allocation, views, copies, and graph ownership.
2. Uploads, transfers, metadata, and sampling buffers.
3. Weights, model workspaces, cache, and paged storage.
4. Collectives and all model operator wrappers.
5. Development tools and test fixtures.

Remove direct Python calls to `device.module` for tensor operations after each stage. Keep direct TVMFFI use only in the adapter, native build identity, and low-level stream bridge.

## Verification

The environment check records Python, Torch, CUDA runtime, CUDA toolkit, TVMFFI, compiler, and native module identities. It checks that all repository commands use `OH_MY_VLLM_CONDA_PREFIX`.

Runtime tests assert that public allocations, views, uploads, weights, cache tensors, and operator results are `torch.Tensor`. Tests cover DLPack ownership, aliasing, non-contiguous rejection, dtype/device contracts, asynchronous upload lifetime, paged mapping, and multi-rank collectives.

Graph tests use Torch graph capture and replay. They check changed inputs, fixed addresses, stream handoff, graph replacement, capture failure cleanup, and closure. No Python test may call the raw CUDA Graph C API.

Run CPU and GPU tests after each migration stage.
Run native build checks after the compiler environment is ready.
Run Qwen regression and DeepSeek numerical acceptance after operator migration.
Run HTTP acceptance and AgentX after service migration.
Keep the existing frozen references and performance accounting.
