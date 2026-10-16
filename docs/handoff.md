# Work state

Rust supplies the HTTP APIs, CLI, scheduler, logical KV cache, and worker lifecycle controls.
The service modes are ordinary decode and MTP4.
HTTP input can contain text, images, and video.

The Python worker loads the checkpoint and executes the language, MTP, and vision model paths.
The `stub` model selects the placeholder for service tests.
Rust sends media references through the inference worker's process and protocol boundary.

The [operator contracts](operators/model.md) have 18 internal stages and 7 serving boundaries.
The workload has six request profiles and one decode context per profile.

Repository CUDA C++ providers use pinned CUTLASS headers for the seven boundaries through the [kernel development tools](operators/development.md).
The build tools are different from the inference loader.
Each provider supplies numerical checks and restores mutable state before measurement.
Reports bind checkpoint shards, source files, binaries, and build dependencies.
Vision MLP activation uses projection output stores.

Sampling normalization uses 640 threads per block.
MTP input normalization uses 640 threads per block when the input has 32 rows or fewer.
Residual normalization uses [16-byte input loads](../python/oh_my_vllm/kernels/cuda_backend/normalization.cuh#L19)<!-- source: uint4 input = *reinterpret_cast<const uint4 *>(x + row * 5120 + column); -->.

FFN and FA use a shared [projection implementation](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L140)<!-- source: inline void decode_projection_launch(TensorView activation, TensorView weight, --> for inputs with eight rows or fewer.
Native operators use ordinary CUDA stream launches. PDL is not used.
When the input has 1, 4, or 5 rows, FFN uses one [CUDA launch](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L173)<!-- source: inline void paired_up_launch(TensorView input, TensorView input_scale, TensorView weight, --> for its upper projections, activation, and quantization.

GDN uses one [CUDA launch for gate projections and convolution](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L152)<!-- source: if (rows > 1 && rows <= 32) --> when the input has 2 to 32 rows.
For one verification row, GDN uses one [CUDA launch](../python/oh_my_vllm/kernels/cuda_backend/gdn_input.cuh#L142)<!-- source: if constexpr (FuseConvolution) { --> for input projection and convolution.
For prefill inputs with more than 32 rows, GDN uses [separate blocks](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L178)<!-- source: scan(std::false_type{}, std::integral_constant<int, 16>{}, std::integral_constant<int, 256>{}); --> for groups of 16 state value rows.
A separate kernel applies output normalization and quantization.
GDN recurrent computation uses [512 threads per block](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L187)<!-- source: run(std::integral_constant<int, 512>{}); --> for verification inputs with more than one row.

FA has a separate prefill implementation.
FA uses [16-row query tiles](../python/oh_my_vllm/kernels/cuda_backend/fa.cu#L156)<!-- source: if (max_queries * 6 <= 16) --> for one or two queries per request when the input has 32 rows or fewer.
FA combines two or more partial attention results with 256 threads per block.
It uses 32 threads per block when there is only one partial result.

The [numerical contract](operators/model.md#intermediate-precision) follows the pinned vLLM source types and rounding points.
The project has no vLLM runtime dependency.
Named tolerances apply to specific outputs. Cache checks and state selection keep their separate requirements.

More work is necessary for the 60% performance requirement.

The worker supplies constrained decoding, MTP4, and CUDA Graph execution for batches that contain only decode requests.
Sampling uses [fixed CPU buffers](../python/oh_my_vllm/worker/sampling.py#L9)<!-- source: class SamplingInputs: --> for masks, parameters, random counters, and token history.
It copies token history to the GPU only when a penalty uses the history.
MTP verification does not download unused sampled tokens.

Operator and end-to-end optimization is in progress.
Next work: decrease execution time. Complete the two performance audits with the same source files.
Use producer and consumer fusion. Do not use PDL in the inference worker.

Measure the full HTTP request time with the [end-to-end requirements](performance.md).
The performance limit uses the recorded trace and the sum of its serving operator roofline times.
Keep numerical checks and full-boundary measurements with each provider.

The scheduler selects decode first and divides the available token budget between prefill requests.
The default token budget is 32768. The default maximum number of running requests is 4.
Use the [service options](serving.md#start) to change these settings.

Use `main` for work.
Keep host paths and artifact locations in ignored local configuration.
Keep the [runtime contracts](requirements.md) and [process boundary](architecture.md).

Keep all English documents in ASD-STE100. Do not put test results in this document.
