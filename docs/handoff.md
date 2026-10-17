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

Vision uses one [kernel](../python/oh_my_vllm/kernels/cuda_backend/vision_residual_norm.cuh#L3)<!-- source: __global__ void vision_residual_norm_kernel( --> for the first residual update and the second normalization.
It keeps the original BF16 rounding points, centered variance, and reduction tree.
The rounded residual stays in registers for normalization and in global storage for the later residual update.

Vision [MLP down](../python/oh_my_vllm/kernels/cuda_backend/vision_classic.cuh#L147)<!-- source: inline void vision_classic_down( --> uses classic cuBLAS when the input has more than 32 rows.
BF16 inputs produce FP32 projected storage. The original BF16 bias and residual rounding stay unchanged.
The owned handle and workspace are prepared before first capture and closed after the related graphs are destroyed.
The builder and loader bind vendor headers, images, and link flags, then examine the loaded images.

Vision [Q/K/V packing](../python/oh_my_vllm/kernels/cuda_backend/vision.cu#L49)<!-- source: __shared__ float cosines[72], sines[72]; --> computes position coefficients once for each token row.
A block shares BF16-rounded sine and cosine values across attention heads.
Bias rounding, paired dimensions, padded zeros, and frame order stay unchanged.

Vision [attention](../python/oh_my_vllm/kernels/cuda_backend/vision_attention.cuh#L18)<!-- source: static_assert(Dimension == 80); --> computes 80 dimensions for heads with 72 dimensions.
Global storage keeps a 128-element row stride. The unpack operation reads dimensions 0 through 71.
Grouped TMA copies use the shared-memory layout from the pinned CUTLASS collective and the original softmax scale.

Sampling normalization uses 640 threads per block.
MTP input normalization uses 640 threads per block when the input has 32 rows or fewer.
For these inputs, MTP uses a [BF16 projection](../python/oh_my_vllm/kernels/cuda_backend/mtp_projection.cuh#L114)<!-- source: inline void mtp_projection_launch(TensorView input, TensorView weight, TensorView partials, --> with four K partitions.
It adds the FP32 partial results before the BF16 output store.

MTP [endpoint copies](../python/oh_my_vllm/kernels/cuda_backend/mtp.cu#L151)<!-- source: for (int vector = threadIdx.x; vector < 640; vector += blockDim.x) { --> use 16-byte loads and stores.
The original tensor alignment contract applies.
Residual normalization uses [16-byte input loads](../python/oh_my_vllm/kernels/cuda_backend/normalization.cuh#L19)<!-- source: uint4 input = *reinterpret_cast<const uint4 *>(x + row * 5120 + column); -->.

FFN down, FA output, and GDN output use a shared [projection implementation](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L191)<!-- source: inline void decode_projection_launch(TensorView activation, TensorView weight, --> for inputs with eight rows or fewer.
Single-row inputs use a [fixed row count](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L196)<!-- source: decode_projection_rows_launch<1>(activation, weight, activation_scale, weight_scale, partials, -->, one-column TMEM loads, and fixed arrays.
Other inputs use runtime row counts.
Native operators use ordinary CUDA stream launches. PDL is not used.
When the input has 1, 4, or 5 rows, FFN uses one [CUDA launch](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L173)<!-- source: inline void paired_up_launch(TensorView input, TensorView input_scale, TensorView weight, --> for its upper projections, activation, and quantization.

FFN prefill uses [paired FP32 accumulation](../python/oh_my_vllm/kernels/cuda_backend/ffn_pair_accum.cuh#L41)<!-- source: template <class Base> struct FFNPairedAccum : Base { --> for inputs with more than 64 rows.
It keeps the original K-group order, factors, and BF16 stores.
The upper projection uses [TMA multicast](../python/oh_my_vllm/kernels/cuda_backend/ffn_prefill.cuh#L43)<!-- source: using Cluster = std::conditional_t<MulticastA, Shape<_1, _2, _1>, Shape<_1, _1, _1>>; --> for two blocks with different output columns.
These blocks share the same input tile. The down projection uses separate TMA loads.

GDN uses one [CUDA launch for gate projections and convolution](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L208)<!-- source: if (rows > 1 && rows <= 32) --> when the input has 2 to 32 rows.
For one verification row, GDN uses one [CUDA launch](../python/oh_my_vllm/kernels/cuda_backend/gdn_input.cuh#L142)<!-- source: if constexpr (FuseConvolution) { --> for input projection and convolution.

For one verification row in one request, GDN uses [64-bit state loads and stores](../python/oh_my_vllm/kernels/cuda_backend/gdn_kernels.cuh#L234)<!-- source: uint2 packed = *reinterpret_cast<const uint2 *>(address); -->.
Paired BF16 conversions keep the RN conversion for each element. The live carry and recurrent arithmetic stay in FP32.

For prefill inputs with more than 32 rows, GDN [precomputes the gates](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L81)<!-- source: bool precomputed_gates = !verify && rows > 32; --> after the BF16 projection.
Separate blocks process groups of 16 state value rows.
Each block [prefetches 32 tokens](../python/oh_my_vllm/kernels/cuda_backend/gdn_kernels.cuh#L145)<!-- source: constexpr int GDNPrefetchRows = 32; --> into two shared-memory slots.
The FP32 carry stays in registers across these slots and BF16 checkpoint stores.
A separate kernel applies output normalization and quantization.

Each [16-lane group](../python/oh_my_vllm/kernels/cuda_backend/gdn_kernels.cuh#L189)<!-- source: __device__ inline float gdn_half_warp_sum(float value) { --> processes one value row in this prefill path.
Two four-key sums reproduce the original reduction order.

The value groups also apply to [one five-row verification request](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L79)<!-- source: bool split_values = (!verify && rows > 32) || (verify && batch == 1 && rows == 5); -->.
GDN recurrent computation uses [512 threads per block](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L259)<!-- source: run(std::integral_constant<int, 512>{}); --> for other verification inputs with more than one row.

FA input projection uses [64 output columns for each block](../python/oh_my_vllm/kernels/cuda_backend/fa_small_input.cuh#L139)<!-- source: inline void fa_small_input_launch(TensorView input, TensorView input_scale, TensorView weight, -->.
The full and query-only paths use this kernel for 1, 4, or 5 input rows.
FP32 scale updates keep K-group order and the original BF16 output conversion.

The [KV-only path](../python/oh_my_vllm/kernels/cuda_backend/fa.cu#L103)<!-- source: bool split_kv = variant == 1 && (rows == 4 || rows == 16) && --> uses eight K partitions for 4 or 16 rows when the attention scratch buffer has sufficient space.
The head kernel adds the FP32 partials before BF16 conversion, K normalization, RoPE, and private KV writes.
This changes FP32 association. The named tolerances apply.
Other KV-only inputs use the groupwise implementation.

FA prefill uses [TMA copies](../python/oh_my_vllm/kernels/cuda_backend/fa_prefill.cuh#L136)<!-- source: if (first % 16 == 0 && base + 64 <= last) { --> for complete aligned KV tiles.
Each descriptor box is in one 784-token page.
Vector copies handle unaligned starts and final partial tiles.

FA [decode loads](../python/oh_my_vllm/kernels/cuda_backend/fa_attention.cuh#L75)<!-- source: Position base_offset = base % 784; --> calculate cache page pointers before the vector loop.
FA uses a [transposed warp matrix product](../python/oh_my_vllm/kernels/cuda_backend/fa_q1_warp_transpose.cuh#L17)<!-- source: __global__ void fa_q1_warp_partial_kernel( --> for one query per request with at most 32 input rows and 64 KV partitions.
This path keeps BF16 probabilities and the original scalar softmax reduction tree.

Other small inputs use [16-row or 32-row query tiles](../python/oh_my_vllm/kernels/cuda_backend/fa.cu#L206)<!-- source: if (max_queries * 6 <= 16) -->.
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

The worker [examines token history](../python/oh_my_vllm/worker/repetition.py#L9)<!-- source: def cycle_continuations(history): --> with CPU prefix filters and complete token span comparisons.
The history window, cycle lengths, and minimum repeat count stay unchanged.

For batches with zero temperature in all rows, sampling uses a [two-stage maximum reduction](../python/oh_my_vllm/kernels/cuda_backend/sample_greedy.cuh#L6)<!-- source: __global__ void greedy_scores_kernel( -->.
It keeps one-hot probabilities for MTP acceptance. Random and mixed batches use the sort path.
The sampler keeps a [separate CUDA Graph for each path](../python/oh_my_vllm/worker/sampling.py#L113)<!-- source: self.replays[self.greedy] = self.resources.enter_context( -->.
It [calculates sampling random counters on the CPU](../python/oh_my_vllm/worker/sampling.py#L122)<!-- source: counters = self.inputs.random[:, 1] + (self.inputs.parameters[:, 0] > 0) -->.

The acceptance audit uses [native sources from the kernel directory](../development/kernels/acceptance.py#L117)<!-- source: source = SOURCES / f"{name}.cu" -->.
It must include [greedy numerical evidence](../development/kernels/acceptance.py#L236)<!-- source: def verify_greedy(check): -->.

The selected kernel and worker changes are integrated. Further optimization is deferred.
The root [README](../README.md#performance-measurements) gives the operator summary, HTTP measurements, and warmup scope.
`copy_bytes` now synchronizes only when the destination is on the CPU.
`gather_rows_kernel` uses 16-byte vectorized loads when the row and base pointers are 16-byte aligned.

FFN uses TMA with `EVICT_FIRST` and `CU_TENSOR_MAP_L2_PROMOTION_NONE` for weight loads in [`paired_up_kernel`](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L59)<!-- source: uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST), --> and [`decode_projection_kernel`](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L76)<!-- source: uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST), -->.

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

If optimization starts again, decrease execution time. Complete the two performance audits with the same source files.

Compare TMA weight layouts and native BF16 matrix libraries before further kernel changes.
Compare score-buffer schedules and direct TMA descriptors before you add packed weight storage.
Compare weight refill points with completion of the accumulator readers.
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
