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

Vision [MLP down](../python/oh_my_vllm/kernels/cuda_backend/vision_classic.cuh#L255)<!-- source: inline void vision_classic_down( --> uses classic cuBLAS when the input has more than 32 rows.
BF16 inputs produce FP32 projected storage. The original BF16 bias and residual rounding stay unchanged.
The owned handle and cuBLAS buffer are prepared before capture.
The builder and loader bind vendor headers, images, and link flags, then examine the loaded images.

Each context prepares a [GELU table](../python/oh_my_vllm/kernels/cuda_backend/vision_lookup_init.cuh#L3)<!-- source: __global__ void vision_lookup_initialize_kernel(cutlass::bfloat16_t *table) { --> before capture.
The table has one BF16 output for each BF16 input bit pattern.
Its entries use the original GPU functor and BF16 output conversion.

A [native reference](../python/oh_my_vllm/kernels/cuda_backend/vision_classic.cuh#L95)<!-- source: class VisionClassicGraphLeaseObj : public tvm::ffi::Object { --> holds this context until graph cleanup finishes.
The context waits for GPU work before it releases its table and cuBLAS buffer.

Vision [Q/K/V packing](../python/oh_my_vllm/kernels/cuda_backend/vision.cu#L49)<!-- source: __shared__ float cosines[72], sines[72]; --> computes position coefficients once for each token row.
A block shares BF16-rounded sine and cosine values across attention heads.
Bias rounding, paired dimensions, padded zeros, and frame order stay unchanged.

Vision [attention](../python/oh_my_vllm/kernels/cuda_backend/vision_attention.cuh#L18)<!-- source: static_assert(Dimension == 80); --> computes 80 dimensions for heads with 72 dimensions.
Q/K/V storage keeps a 128-element row stride.
The [packed epilogue](../python/oh_my_vllm/kernels/cuda_backend/vision_attention.cuh#L30)<!-- source: using Epilogue = cute::conditional_t<Packed, VisionPackedEpilogue<BaseEpilogue>, --> writes 72 dimensions per head directly to the output tensor.
Grouped TMA copies use the shared-memory layout from the pinned CUTLASS collective and the original softmax scale.

Sampling normalization uses 640 threads per block.
MTP input normalization uses 640 threads per block when the input has 32 rows or fewer.
For these inputs, MTP uses a [BF16 projection](../python/oh_my_vllm/kernels/cuda_backend/mtp_projection.cuh#L117)<!-- source: inline void mtp_projection_launch(TensorView input, TensorView weight, TensorView partials, --> with four K partitions.
It adds the FP32 partial results before the BF16 output store.

MTP [endpoint copies](../python/oh_my_vllm/kernels/cuda_backend/mtp.cu#L151)<!-- source: for (int vector = threadIdx.x; vector < 640; vector += blockDim.x) { --> use 16-byte loads and stores.
The original tensor alignment contract applies.
Residual normalization uses [16-byte input loads](../python/oh_my_vllm/kernels/cuda_backend/normalization.cuh#L19)<!-- source: uint4 input = *reinterpret_cast<const uint4 *>(x + row * 5120 + column); -->.

FFN down, FA output, and GDN output use a shared [projection implementation](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L194)<!-- source: inline void decode_projection_launch(TensorView activation, TensorView weight, --> for inputs with eight rows or fewer.
Single-row inputs use a [fixed row count](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L199)<!-- source: decode_projection_rows_launch<1>(activation, weight, activation_scale, weight_scale, partials, -->, one-column TMEM loads, and fixed arrays.
Inputs with 4 or 5 rows use [fixed kernels](../python/oh_my_vllm/kernels/cuda_backend/projection_decode_16dp.cuh#L195)<!-- source: inline void launch(TensorView activation, TensorView weight, TensorView activation_scale, -->.
Their TMEM loads use 16 data paths per warp.
Paired FP32 instructions keep the original K-group order.
Other inputs use runtime row counts.

Native operators use ordinary CUDA stream launches. PDL is not used.
When the input has 1, 4, or 5 rows, FFN uses one [CUDA launch](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L176)<!-- source: inline void paired_up_launch(TensorView input, TensorView input_scale, TensorView weight, --> for its upper projections, activation, and quantization.

FA and GDN inputs with more than 32 rows use a [native projection pipeline](../python/oh_my_vllm/kernels/cuda_backend/projection_prefill.cuh#L31)<!-- source: __global__ __launch_bounds__(640, 1) void projection_kernel( -->.
FFN uses the same pipeline for inputs with more than 64 rows.
It keeps the original K-group order, FP32 factors, and BF16 rounding points.
Each block loads all scales before the repeated matrix products. Separate slots protect input storage and TMEM readers.

The FFN [upper projection](../python/oh_my_vllm/kernels/cuda_backend/ffn_prefill_fused.cuh#L37)<!-- source: __global__ __launch_bounds__(640, 1) void upper_kernel<false>( --> applies SiLU and FP8 quantization to BF16 results in shared memory.
The intermediate gate and up values stay in shared memory.

GDN uses one [CUDA launch for gate projections and convolution](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L213)<!-- source: if (rows > 1 && rows <= 32) --> when the input has 2 to 32 rows.
For one verification row, GDN uses one [CUDA launch](../python/oh_my_vllm/kernels/cuda_backend/gdn_input_pipeline.cuh#L160)<!-- source: if constexpr (FuseConvolution) { --> for input projection and convolution.

For verification inputs, GDN uses [64-bit state loads and stores](../python/oh_my_vllm/kernels/cuda_backend/gdn_kernels.cuh#L289)<!-- source: uint2 packed = *reinterpret_cast<const uint2 *>(address); -->.
Paired BF16 conversions keep the RN conversion for each element. The live carry and recurrent arithmetic stay in FP32.

For prefill inputs with more than 32 rows, GDN [precomputes the gates](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L86)<!-- source: bool precomputed_gates = !verify && rows > 32; --> after the BF16 projection.
The [value tile selection](../python/oh_my_vllm/kernels/cuda_backend/gdn_value_pair.cuh#L323)<!-- source: inline bool gdn_prefill_two_pairs(int rows, int max_queries, int device) { --> uses 16 or 32 state value rows for each block.
It uses token counts, the longest sequence, and the SM count.

Each block [prefetches 16 tokens](../python/oh_my_vllm/kernels/cuda_backend/gdn_kernels.cuh#L145)<!-- source: constexpr int GDNPrefetchRows = 16; --> into two shared-memory slots.
The slots keep scaled Q values in FP32 and K values in BF16.
The FP32 carry stays in registers across these slots and BF16 checkpoint stores.
A separate kernel applies output normalization and quantization.

Each [group of 16 lanes](../python/oh_my_vllm/kernels/cuda_backend/gdn_value_pair.cuh#L52)<!-- source: __device__ __forceinline__ uint64_t gdn_pair_half_sum(uint64_t value) { --> processes independent value rows with paired FP32 instructions.
Two four-key sums keep the original reduction order.
The [checkpoint stores](../python/oh_my_vllm/kernels/cuda_backend/gdn_value_pair.cuh#L150)<!-- source: pool[((int64_t(target) * 48 + head) * 128 + value_base + unit) * 128 + --> follow the state updates in one conditional block.

GDN also divides state values between blocks for [verification inputs](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L83)<!-- source: bool split_values = (!verify && rows > 32) || --> with 1 or 4 requests of 5 rows each.
GDN recurrent computation uses [512 threads per block](../python/oh_my_vllm/kernels/cuda_backend/gdn.cu#L280)<!-- source: run(std::integral_constant<int, 512>{}); --> for other verification inputs with more than one row.

FA input projection uses [64 output columns for each block](../python/oh_my_vllm/kernels/cuda_backend/fa_small_input.cuh#L225)<!-- source: fa_small_input_launch(TensorView input, TensorView input_scale, TensorView weight, -->.
The full and query-only paths use this kernel for 1, 4, or 5 input rows.
FP32 scale updates keep K-group order and the original BF16 output conversion.

The [KV-only path](../python/oh_my_vllm/kernels/cuda_backend/fa.cu#L106)<!-- source: bool split_kv = variant == 1 && (rows == 4 || rows == 16) && --> uses eight K partitions for 4 or 16 rows when the attention scratch buffer has sufficient space.
The head kernel adds the FP32 partials before BF16 conversion, K normalization, RoPE, and private KV writes.
This changes FP32 association. The named tolerances apply.
Other KV-only inputs use the groupwise implementation.

FA prefill uses [separate roles](../python/oh_my_vllm/kernels/cuda_backend/fa_prefill_pipeline.cuh#L39)<!-- source: __global__ void fa_prefill_kernel( --> for copies, matrix products, softmax, and output correction.
[Grouped TMA descriptors](../python/oh_my_vllm/kernels/cuda_backend/fa_prefill.cuh#L58)<!-- source: uint32_t key_box[5] = {64, 64, 4, 1, 1}; --> supply complete aligned KV tiles.
Each descriptor box is in one 784-token page.
Vector copies handle unaligned starts and final partial tiles.

The [softmax output conversion](../python/oh_my_vllm/kernels/cuda_backend/fa_prefill_pipeline.cuh#L328)<!-- source: asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(word) : "f"(p0), "f"(p1)); --> converts two FP32 probabilities to BF16 with one RN instruction.
Each vector store writes eight BF16 probabilities. The FP32 sum keeps the original order.

FA [decode loads](../python/oh_my_vllm/kernels/cuda_backend/fa_attention.cuh#L75)<!-- source: Position base_offset = base % 784; --> calculate cache page pointers before the vector loop.
FA uses a [transposed warp matrix product](../python/oh_my_vllm/kernels/cuda_backend/fa_q1_warp_transpose.cuh#L17)<!-- source: __global__ void fa_q1_warp_partial_kernel( --> for one query per request with at most 32 input rows and 64 KV partitions.
This path keeps BF16 probabilities and the original scalar softmax reduction tree.

Other small inputs use [16-row or 32-row query tiles](../python/oh_my_vllm/kernels/cuda_backend/fa.cu#L231)<!-- source: if (max_queries * 6 <= 16) -->.
FA combines two or more partial attention results with 256 threads per block.
It uses 32 threads per block when there is only one partial result.

The [numerical contract](operators/model.md#intermediate-precision) follows the pinned vLLM source types and rounding points.
The project has no vLLM runtime dependency.
Named tolerances apply to specific outputs. Cache checks and state selection keep their separate requirements.

For performance measurements, use the [evaluation procedure](performance.md).
Operator tests have no minimum efficiency requirement.
Each delivery must include complete operator measurements and an update to the two README tables.
The roofline calculations must use [measured peaks](calibration.md) for the selected GPU UUID.
The HTTP comparison keeps its 50% efficiency requirement.

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

The acceptance audit uses [native sources from the kernel directory](../development/kernels/acceptance.py#L120)<!-- source: source = SOURCES / f"{name}.cu" -->.
It must include [greedy numerical evidence](../development/kernels/acceptance.py#L239)<!-- source: def verify_greedy(check): -->.

`copy_bytes` now synchronizes only when the destination is on the CPU.
`gather_rows_kernel` uses 16-byte vectorized loads when the row and base pointers are 16-byte aligned.

FFN uses TMA with `EVICT_FIRST` and `CU_TENSOR_MAP_L2_PROMOTION_NONE` for weight loads in [`paired_up_kernel`](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L59)<!-- source: uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST), --> and [`decode_projection_kernel`](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L76)<!-- source: uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST), -->.

The [row copy](../python/oh_my_vllm/kernels/cuda_backend/device.cu#L182)<!-- source: int column_blocks = std::clamp<int64_t>(row_bytes / 4096 + (row_bytes % 4096 != 0), 1, --> can use more than one block for each row.
The number of rows limits the number of blocks for each row. Each row uses at least one block.

MTP [acceptance](../python/oh_my_vllm/worker/speculation.py#L158)<!-- source: self.gather = device.empty((batch * 5,), "int32") --> keeps a fixed row-index buffer.
MTP context [reads the model and boundary hidden buffers](../python/oh_my_vllm/worker/speculation.py#L53)<!-- source: self.boundary_hidden = cache.hidden --> directly.
The context holds these buffers. Its graph cache key includes the source buffer identity.
Draft computation uses [views of the sampling output](../python/oh_my_vllm/worker/speculation.py#L479)<!-- source: previous[index] = d.view( --> for the next stage.
Each stage copies all necessary rows before sampling uses the source buffer again.

Greedy MTP requests can use [one CUDA Graph for four draft stages](../python/oh_my_vllm/worker/speculation.py#L409)<!-- source: if can_chain(batch, active, graph, self.cache.blocks): -->.
This path is for requests without penalties, grammar, thinking, stop strings, or a repetition guard.
The minimum output length must prevent EOS selection in all four stages.
Four draft positions must have private KV pages.

The stages use the same sampling input buffers. Each stage has different output buffers.
The worker [downloads the four draft tokens once](../python/oh_my_vllm/worker/draft_chain.py#L122)<!-- source: tokens = d.download(self.tokens, np.int64) --> and copies probabilities to request buffers.
Other requests have a token download after each draft stage.

Use the same source files for the two performance audits.

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
