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
For these inputs, MTP uses a [BF16 projection](../python/oh_my_vllm/kernels/cuda_backend/mtp_projection.cuh#L114)<!-- source: inline void mtp_projection_launch(TensorView input, TensorView weight, TensorView partials, --> with four K partitions.
It adds the FP32 partial results before the BF16 output store.
Residual normalization uses [16-byte input loads](../python/oh_my_vllm/kernels/cuda_backend/normalization.cuh#L19)<!-- source: uint4 input = *reinterpret_cast<const uint4 *>(x + row * 5120 + column); -->.

FFN down, FA output, and GDN output use a shared [projection implementation](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L147)<!-- source: inline void decode_projection_launch(TensorView activation, TensorView weight, --> for inputs with eight rows or fewer.
Native operators use ordinary CUDA stream launches. PDL is not used.
When the input has 1, 4, or 5 rows, FFN uses one [CUDA launch](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L173)<!-- source: inline void paired_up_launch(TensorView input, TensorView input_scale, TensorView weight, --> for its upper projections, activation, and quantization.

GDN uses one [CUDA launch for gate projections and convolution](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L208)<!-- source: if (rows > 1 && rows <= 32) --> when the input has 2 to 32 rows.
For one verification row, GDN uses one [CUDA launch](../python/oh_my_vllm/kernels/cuda_backend/gdn_input.cuh#L142)<!-- source: if constexpr (FuseConvolution) { --> for input projection and convolution.

For prefill inputs with more than 32 rows, GDN [precomputes the gates](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L81)<!-- source: bool precomputed_gates = !verify && rows > 32; --> after the BF16 projection.
Separate blocks process groups of 16 state value rows.
Each block [prefetches 32 tokens](../python/oh_my_vllm/kernels/cuda_backend/gdn_kernels.cuh#L143)<!-- source: constexpr int GDNPrefetchRows = 32; --> into two shared-memory slots.
The FP32 carry stays in registers across these slots and BF16 checkpoint stores.
A separate kernel applies output normalization and quantization.

The value groups also apply to [one five-row verification request](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L79)<!-- source: bool split_values = (!verify && rows > 32) || (verify && batch == 1 && rows == 5); -->.
GDN recurrent computation uses [512 threads per block](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L253)<!-- source: run(std::integral_constant<int, 512>{}); --> for other verification inputs with more than one row.

FA has a separate prefill implementation.
FA uses [16-row query tiles](../python/oh_my_vllm/kernels/cuda_backend/fa.cu#L156)<!-- source: if (max_queries * 6 <= 16) --> for one or two queries per request when the input has 32 rows or fewer.
FA combines two or more partial attention results with 256 threads per block.
It uses 32 threads per block when there is only one partial result.

The [numerical contract](operators/model.md#intermediate-precision) follows the pinned vLLM source types and rounding points.
The project has no vLLM runtime dependency.
Named tolerances apply to specific outputs. Cache checks and state selection keep their separate requirements.

More work is necessary for the 60% performance requirement.

The worker supplies constrained decoding, MTP4, and CUDA Graph execution for batches that contain only decode requests.
The model, MTP context, and MTP draft use [fixed metadata buffers](../python/oh_my_vllm/worker/metadata.py#L5)<!-- source: class Metadata: -->.
Arrays with the same data type share a buffer. Each update copies each buffer once.
Each view starts at a 16-byte boundary.

Sampling uses [fixed CPU buffers](../python/oh_my_vllm/worker/sampling.py#L9)<!-- source: class SamplingInputs: --> for masks, parameters, random counters, and token history.
It copies token history to the GPU only when a penalty uses the history.
MTP verification does not download unused sampled tokens.

For batches with zero temperature in all rows, sampling uses a [two-stage maximum reduction](../python/oh_my_vllm/kernels/cuda_backend/sample_greedy.cuh#L6)<!-- source: __global__ void greedy_scores_kernel( -->.
It keeps one-hot probabilities for MTP acceptance. Random and mixed batches use the sort path.
The sampler keeps a [separate CUDA Graph for each path](../python/oh_my_vllm/worker/sampling.py#L113)<!-- source: self.replays[self.greedy] = self.resources.enter_context( -->.
It [calculates sampling random counters on the CPU](../python/oh_my_vllm/worker/sampling.py#L122)<!-- source: counters = self.inputs.random[:, 1] + (self.inputs.parameters[:, 0] > 0) -->.

The acceptance audit uses [native sources from the kernel directory](../development/kernels/acceptance.py#L117)<!-- source: source = SOURCES / f"{name}.cu" -->.
It must include [greedy numerical evidence](../development/kernels/acceptance.py#L236)<!-- source: def verify_greedy(check): -->.

Operator and end-to-end optimization is in progress.
`copy_bytes` now synchronizes only when the destination is on the CPU.
`gather_rows_kernel` uses 16-byte vectorized loads when the row and base pointers are 16-byte aligned.

FFN uses TMA with `EVICT_FIRST` and `CU_TENSOR_MAP_L2_PROMOTION_NONE` for weight loads in [`paired_up_kernel`](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L59)<!-- source: uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST), --> and [`decode_projection_kernel`](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L71)<!-- source: uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST), -->.

The [row copy](../python/oh_my_vllm/kernels/cuda_backend/device.cu#L182)<!-- source: int column_blocks = std::clamp<int64_t>(row_bytes / 4096 + (row_bytes % 4096 != 0), 1, --> can use more than one block for each row.
The number of rows limits the number of blocks for each row. Each row uses at least one block.

MTP [acceptance](../python/oh_my_vllm/worker/speculation.py#L148)<!-- source: self.gather = device.empty((batch * 5,), "int32") --> keeps a fixed row-index buffer.
MTP context [reads the model and boundary hidden buffers](../python/oh_my_vllm/worker/speculation.py#L53)<!-- source: self.boundary_hidden = cache.hidden --> directly.
The context holds these buffers. Its graph cache key includes the source buffer identity.
Draft computation uses [views of the sampling output](../python/oh_my_vllm/worker/speculation.py#L469)<!-- source: previous[index] = d.view( --> for the next stage.
Each stage copies all necessary rows before sampling uses the source buffer again.

Greedy MTP requests can use [one CUDA Graph for four draft stages](../python/oh_my_vllm/worker/speculation.py#L399)<!-- source: if can_chain(batch, active, graph, self.cache.blocks): -->.
This path is for requests without penalties, grammar, thinking, stop strings, or a repetition guard.
The minimum output length must prevent EOS selection in all four stages.
Four draft positions must have private KV pages.

The stages use the same sampling input buffers. Each stage has different output buffers.
The worker [downloads the four draft tokens once](../python/oh_my_vllm/worker/draft_chain.py#L122)<!-- source: tokens = d.download(self.tokens, np.int64) --> and copies probabilities to request buffers.
Other requests have a token download after each draft stage.

Next work: decrease execution time. Complete the two performance audits with the same source files.
Use producer and consumer fusion. Do not use PDL in the inference worker.
Use the [optimization procedure](operators/development.md#optimization-procedure) for model checks, reusable lessons, and temporary artifact cleanup.

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
