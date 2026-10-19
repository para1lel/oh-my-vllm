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
Residual normalization uses [16-byte input loads](../python/oh_my_vllm/kernels/cuda_backend/normalization.cuh#L30)<!-- source: uint4 input = *reinterpret_cast<const uint4 *>(x + row * 5120 + column); -->.

FFN down, FA output, and GDN output use a shared [projection implementation](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L194)<!-- source: inline void decode_projection_launch(TensorView activation, TensorView weight, --> for inputs with eight rows or fewer.
Single-row inputs use a [fixed row count](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L199)<!-- source: decode_projection_rows_launch<1>(activation, weight, activation_scale, weight_scale, partials, -->, one-column TMEM loads, and fixed arrays.
Inputs with 4 or 5 rows use [fixed kernels](../python/oh_my_vllm/kernels/cuda_backend/projection_decode_16dp.cuh#L195)<!-- source: inline void launch(TensorView activation, TensorView weight, TensorView activation_scale, -->.
Their TMEM loads use 16 data paths per warp.
Paired FP32 instructions keep the original K-group order.
Other inputs use runtime row counts.

Native operators use ordinary CUDA stream launches. PDL is not used.
When the input has 1, 4, or 5 rows, FFN uses one [CUDA launch](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L176)<!-- source: inline void paired_up_launch(TensorView input, TensorView input_scale, TensorView weight, --> for its upper projections, activation, and quantization.

FA and GDN inputs with 33 to 128 rows use a [native projection pipeline](../python/oh_my_vllm/kernels/cuda_backend/projection_prefill.cuh#L31)<!-- source: __global__ __launch_bounds__(640, 1) void projection_kernel( -->.
Larger fallback output projections also use this pipeline.
FFN uses the same pipeline for inputs with 65 to 128 rows.

It keeps the original K-group order, FP32 factors, and BF16 rounding points.
Each block loads all scales before the repeated matrix products. Separate slots protect input storage and TMEM readers.

For 65 to 128 rows, the FFN [upper projection](../python/oh_my_vllm/kernels/cuda_backend/ffn_prefill_fused.cuh#L37)<!-- source: __global__ __launch_bounds__(640, 1) void upper_kernel<false>( --> applies SiLU and FP8 quantization to BF16 results in shared memory.
The intermediate gate and up values stay in shared memory.

Larger FFN inputs use an [owned MXFP8 context](../python/oh_my_vllm/kernels/cuda_backend/mx_fp8_plan.cuh#L203)<!-- source: class MXContextObj : public tvm::ffi::Object { -->.
The context keeps prepared weights and scratch storage through CUDA Graph cleanup.
Each FFN owner prepares an FP32 SiLU table for all 65536 BF16 gate bit patterns.
The table keeps the original multiplication order and BF16 product store.
The [precision contract](operators/model.md#intermediate-precision) gives its operand types and named tolerances.

Full FA inputs with more than 128 rows use an [owned MXFP8 context](../python/oh_my_vllm/kernels/cuda_backend/fa_mx_plan.cuh#L44)<!-- source: class MXContextObj : public tvm::ffi::Object { -->.
It supplies input projection.
It supplies output projection when attention has one split.
Other attention splits keep the previous output projection.

GDN inputs with more than 128 rows use a [separate owned MXFP8 context](../python/oh_my_vllm/kernels/cuda_backend/gdn_mx_plan.cuh#L44)<!-- source: class MXContextObj : public tvm::ffi::Object { -->.
It keeps each layer's prepared weights and all call tensors through graph cleanup.
It also supplies output projection for prefill and verification with `max_queries > 1`.
Single-query verification keeps the previous output projection.

For one verification row, GDN uses one [CUDA launch](../python/oh_my_vllm/kernels/cuda_backend/gdn_input_pipeline.cuh#L160)<!-- source: if constexpr (FuseConvolution) { --> for input projection and convolution.
Verification uses a [separate recurrent implementation](../python/oh_my_vllm/kernels/cuda_backend/gdn_recurrent_core.cuh#L6)<!-- source: namespace gdn_recurrent { --> when `max_queries > 1`.
It reads the interleaved Q/K/V and gate projections directly.
It does checks of metadata and private state slots on the GPU.

GDN [selects chunked prefill](../python/oh_my_vllm/kernels/gdn.py#L19)<!-- source: def chunked_prefill(rows, batch, max_queries, verify): --> when the longest sequence has more than 128 tokens.
You can use different sequence lengths. It divides 64-token blocks at 784-token checkpoints and sequence ends.
The [triangular solve](../python/oh_my_vllm/kernels/cuda_backend/gdn_wy_bf16_inverse.cuh#L68)<!-- source: __device__ inline void inverse_blocks(Storage &s) { --> uses FP32 diagonal solves and BF16 block products with FP32 accumulation.
Matrix products use BF16 operands with FP32 accumulation.

The [carry kernel](../python/oh_my_vllm/kernels/cuda_backend/gdn_wy_register.cuh#L87)<!-- source: __global__ __launch_bounds__(Threads, 2) void register_carry_kernel( --> keeps state in FP32 registers.
It converts register fragments directly to BF16 matrix operands.
BF16 private checkpoint stores do not change the FP32 register state.
The [model contract](operators/model.md#intermediate-precision) gives the separate rounding points for new values and state updates.

Prefill inputs with more than 32 rows and at most 128 tokens per sequence use [16-token prefetch slots](../python/oh_my_vllm/kernels/cuda_backend/gdn_kernels.cuh#L145)<!-- source: constexpr int GDNPrefetchRows = 16; --> for the token scan.
These slots keep scaled Q values in FP32 and K values in BF16.
A separate kernel applies output normalization and quantization.

FA input projection uses [64 output columns for each block](../python/oh_my_vllm/kernels/cuda_backend/fa_small_input.cuh#L225)<!-- source: fa_small_input_launch(TensorView input, TensorView input_scale, TensorView weight, -->.
The full and query-only paths use this kernel for 1, 4, or 5 input rows.
FP32 scale updates keep K-group order and the original BF16 output conversion.

The [KV-only path](../python/oh_my_vllm/kernels/cuda_backend/fa.cu#L151)<!-- source: bool split_kv = variant == 1 && (rows == 4 || rows == 16) && --> uses eight K partitions for 4 or 16 rows when the attention scratch buffer has sufficient space.
The head kernel adds the FP32 partials before BF16 conversion, K normalization, RoPE, and private KV writes.
This changes FP32 association. The named tolerances apply.
Other KV-only inputs use the groupwise implementation.

FA prefill uses [separate roles](../python/oh_my_vllm/kernels/cuda_backend/fa_prefill_pipeline.cuh#L40)<!-- source: __global__ void fa_prefill_kernel( --> for copies, matrix products, softmax, and output correction.
Large inputs use [clusters of two SMs](../python/oh_my_vllm/kernels/cuda_backend/fa_prefill_cluster.cuh#L84)<!-- source: template <bool ExportBFloat16, bool Decouple> -->.
When `Decouple` is selected, separate score and probability buffers let copies, score products, and output products operate at the same time.

FA uses a [ring schedule](../python/oh_my_vllm/kernels/cuda_backend/fa_core.cuh#L56)<!-- source: if (max_queries >= 512 && max_queries < 1024 && batch == 1) { --> for one request with 512 to 1023 queries.
The worker supplies the maximum context length from CPU metadata.
The FA buffer cache key has a separate value for the 12544-token schedule threshold.

Single-split prefill writes BF16 attention into packed projection storage after the head kernel has read it.
The following kernel applies the gate and FP8 conversion without FP32 partial reads.

[Grouped TMA descriptors](../python/oh_my_vllm/kernels/cuda_backend/fa_prefill.cuh#L60)<!-- source: uint32_t key_box[5] = {64, 64, 4, 1, 1}; --> supply complete aligned KV tiles.
Each descriptor box is in one 784-token page.
Additional 16-token TMA copies supply complete aligned tiles across page boundaries.
Vector copies handle unaligned starts and final partial tiles.

The [softmax output conversion](../python/oh_my_vllm/kernels/cuda_backend/fa_prefill_pipeline.cuh#L337)<!-- source: asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(word) : "f"(p0), "f"(p1)); --> converts two FP32 probabilities to BF16 with one RN instruction.
Each vector store writes eight BF16 probabilities. The FP32 sum keeps the original order.

FA [decode loads](../python/oh_my_vllm/kernels/cuda_backend/fa_attention.cuh#L82)<!-- source: Position base_offset = base % 784; --> calculate cache page pointers before the vector loop.
FA uses a [transposed warp matrix product](../python/oh_my_vllm/kernels/cuda_backend/fa_q1_warp_transpose.cuh#L17)<!-- source: __global__ void fa_q1_warp_partial_kernel( --> for one query per request with at most 32 input rows.
Four or more requests can use 128 KV partitions when the maximum context is at least 32768 tokens.
The FA buffer cache key separates context lengths of less than 32768 tokens from lengths of 32768 tokens or more.
This path keeps BF16 probabilities and the original scalar softmax reduction tree.

For two nonempty requests, FA can use a [separate warp kernel](../python/oh_my_vllm/kernels/cuda_backend/fa_core.cuh#L16)<!-- source: if (batch == 2 && rows > max_queries && first_token == 0 && min_context >= 32768 && -->.
Use it only with `2 <= max_queries <= 5`, `rows > max_queries`, `first_token == 0`, and `min_context >= 32768`.
Supply capacity for at least 64 partitions.
The buffer cache key includes this selection.

Other small inputs use [16-row or 32-row query tiles](../python/oh_my_vllm/kernels/cuda_backend/fa_core.cuh#L198)<!-- source: if (max_queries * 6 <= 16) -->.
FA combines two or more partial attention results with 256 threads per block.
It uses 32 threads per block when there is only one partial result.

The [numerical contract](operators/model.md#intermediate-precision) uses the pinned vLLM reference with selected FP8 or BF16 intermediate operands.
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

The acceptance audit uses [native sources from the kernel directory](../development/kernels/acceptance.py#L128)<!-- source: source = SOURCES / f"{name}.cu" -->.
It must include [greedy numerical evidence](../development/kernels/acceptance.py#L336)<!-- source: def verify_greedy(check): -->.

`copy_bytes` now synchronizes only when the destination is on the CPU.
`gather_rows_kernel` uses 16-byte vectorized loads when the row and base pointers are 16-byte aligned.

FFN uses TMA with `EVICT_FIRST` and `CU_TENSOR_MAP_L2_PROMOTION_NONE` for weight loads in [`paired_up_kernel`](../python/oh_my_vllm/kernels/cuda_backend/ffn_up.cuh#L59)<!-- source: uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST), --> and [`decode_projection_kernel`](../python/oh_my_vllm/kernels/cuda_backend/projection_decode.cuh#L76)<!-- source: uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST), -->.

The [row copy](../python/oh_my_vllm/kernels/cuda_backend/device.cu#L217)<!-- source: int column_blocks = std::clamp<int64_t>(row_bytes / 4096 + (row_bytes % 4096 != 0), 1, --> can use more than one block for each row.
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

Use the [performance plan](operators/performance-plan.md) and pinned public attention comparisons before further kernel changes.
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
