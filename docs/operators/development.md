# Kernel development

Use the [operator specification](model.md), [fusion plan](fusion.md), and [evaluation procedure](evaluation.md).
Native providers supply all seven serving boundaries. Full workload acceptance is necessary.
The inference worker loads the native modules. The placeholder is used only for `--model stub`.

## Tools

Use CUDA 13.1, Nsight Systems, and Nsight Compute.
Keep their paths in the ignored `LOCAL.md` file.
Set `CUDA_HOME` to the CUDA installation directory.

Install the kernel development packages:

```bash
scripts/with-env.sh uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/kernel-development.txt
```

The inference binding packages are in `requirements/kernels.txt`.
The development packages are not worker dependencies.

Set `CUTLASS_ROOT` to an external dependency directory.
Get the specified CUTLASS headers:

```bash
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass.git "$CUTLASS_ROOT"
export OH_MY_VLLM_CUTLASS="$CUTLASS_ROOT"
```

The build accepts only Git revision `098de2a652cf8f00fd70b2df54051c7eccbb855a` from a clean dependency checkout.
Set `OH_MY_VLLM_KERNEL_CACHE` to an external build directory.

Build the native modules:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels build
```

The inference loader uses these modules. It does not use the compiler or development tools.
The loader rejects source or binary identities that do not agree with its manifest.
Each binary has an immutable copy and build manifest in the external cache.

## Evaluation commands

Get the workload plan:

```bash
scripts/with-env.sh python -m development.kernels plan
```

Do the CPU contract checks:

```bash
scripts/with-env.sh python -m unittest development.kernels.test_contracts development.kernels.test_reference
```

Set `MODEL_ROOT` to the checkpoint directory.
Set `ARTIFACT_ROOT` to an external artifact directory.
Measure the FFN provider:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels bench --model "$MODEL_ROOT" --operator ffn --mode graph --output "$ARTIFACT_ROOT/ffn-graph.json"
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels bench --model "$MODEL_ROOT" --operator ffn --mode eager --output "$ARTIFACT_ROOT/ffn-eager.json"
```

Get a Systems trace and Compute counters for one shape:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.kernels profile --model "$MODEL_ROOT" --operator ffn --rows 1 --mode graph --output "$ARTIFACT_ROOT/ffn-profile"
```

Before the command, select a profile directory path that does not exist.
Profiling does not supply acceptance timing.

Use `--operator` to select `ffn`, `fa_attention`, `gdn_attention`, `vision_block`, `sample`, `mtp_context`, or `mtp_draft`.
Use `--rows` for a subset of the modeled timing rows.
This option does not change the [small numerical case list](evaluation.md#numerical-scope).
Each benchmark does checks of this list in the two execution modes before timing.
A subset is not sufficient for the operator coverage gate.
An incomplete workload or failed provider prevents acceptance.

Supply one full report for each operator and mode from the same frozen sources.
Use the report paths for a full acceptance audit:

```bash
scripts/with-env.sh python -m development.kernels audit --reports "$ARTIFACT_ROOT"/suite-*.json --output "$ARTIFACT_ROOT/acceptance.json"
```

The audit does checks of current sources, immutable binaries, checkpoint identities, cold weights, the small numerical cases, and all modeled timing shapes.
Version 2 reports keep numerical evidence separate from timing coverage.
Large timing cases record numerical status `not_run`. This status does not remove a shape from performance coverage.
Old reports do not qualify after a source change.

For each selected numerical case, the FFN check examines all output elements for finite values.
It compares all residual elements for exact BF16 equality.

The FFN FP64 projection reference compares all columns in the first, middle, and final rows.
Each provider identifies its numerical sample scope in the report.

GDN checks all input projections and private state writes. Sampling checks all rows and vocabulary columns.
FA and MTP check KV writes and unchanged guard storage. Vision checks raw patches, output samples, and frame isolation.
These checks do not supply tolerance evidence for elements not covered by their stated scope.

## Method references

Use external implementations as design references. Obey the [runtime library rules](fusion.md).
Use the [GDN execution plan](gdn.md) for recurrence, value tiles, and sequence partitions.

The [CUTLASS blockwise example](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_blockwise.cu) uses E4M3 operands and FP32 scales.
Its [accumulator transform](https://github.com/NVIDIA/cutlass/blob/v4.8.0/include/cutlass/gemm/collective/sm100_mma_warpspecialized_blockwise_scaling.hpp) applies FP32 updates in K-group order.
The [cuBLAS 13.1 scaling table](https://docs.nvidia.com/cuda/archive/13.1.2/cublas/index.html#scaling-mode-support-overview) limits FP32 128-element and 128-by-128 scaling to compute capability 9.0.
Current [DeepGEMM SM100 FP8 interfaces](https://github.com/deepseek-ai/DeepGEMM/blob/main/README.md) use only packed UE8M0 scales.
Do not change checkpoint scale values to use another matrix instruction.

The [FlashAttention SM100 implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py) separates loads, matrix products, softmax, and output correction.
Its paged implementation supplies a load path without TMA.
Compare this schedule with the current FA implementation for head width 256.
Keep the 784-token cache pages and BF16 probability storage.

The [PTX warp matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma) support fixed matrix shapes.
Compare transposed products when few query rows leave most matrix outputs unused.
Include score exchange, softmax reduction order, output ownership, and shared-memory access.

The SGLang [BF16 GEMV](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/jit/csrc/gemm/hopper_bf16_gemv.cuh) uses vector loads, FP32 FMA, and warp reductions.
Its [small GEMM](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/jit/csrc/gemm/tiny_gemm.cuh) reuses weight values across input rows.
Compare these methods with matrix products that compute unused rows.
Include operand conversion, input cache traffic, reductions, and register use.
Examine the compiled instructions before you use an instruction throughput model.

The [PTX tensor-copy rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor) include swizzle and transaction barrier requirements.
Use descriptor boxes with the same physical layout as K/V MMA inputs.
Keep each box in one cache page. Use vector copies for unaligned starts and partial tiles.

The [CUTLASS TMEM copy instructions](https://github.com/NVIDIA/cutlass/blob/v4.8.0/include/cute/arch/copy_sm100.hpp) supply different load widths for known row counts.
Keep all participating lanes and the necessary destination registers.
A known row count can remove unused scale and accumulator work.
Compare a narrow transposed projection with padded row tiles for small inputs.
Include the extra column blocks and scale requests.
Keep the epilogue operations and their signed-zero behavior before BF16 conversion.

The [PTX barrier rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar) distinguish counted thread groups from `.aligned` CTA barriers.
Use an explicit count and the non-`.aligned` form for independent thread groups in one CTA.
Examine each path and each barrier phase before you reuse its storage.

The [CUTLASS split-K implementation](https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/include/cutlass/gemm/device/gemm_splitk_parallel.h#L256-L307) stores accumulator partials before a separate reduction.
Compare this pattern with a merge in an existing consumer. Include scratch traffic and changes to FP32 association.

For the [FlashInfer GDN prefill interface](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_prefill.py), checkpoint intervals must be divisible by 64.
Its [Blackwell implementation](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py) converts live state to the input type for matrix products.
Do a check of these conditions against the model contract before you use its algorithm.

The [NVIDIA FP32 emulation](https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation) divides each operand into three BF16 components.
It combines nine products. Compare this method with native FP32 products for GDN chunk transforms.
Keep FP32 live state and the 784-token BF16 checkpoints.
Extra implementation work cannot increase the necessary-work lower bound.

The [pinned vLLM sampler](https://github.com/vllm-project/vllm/blob/e8e1e96196725f097eae2394cbc458b1f5a679b8/vllm/v1/sample/sampler.py) selects greedy tokens with `argmax` before top-k and top-p processing.
Compare a greedy `argmax` reduction with the full vocabulary sort.
Apply masks and penalties before token selection.
Keep the required stochastic distribution for random rows.

The [FlashInfer RoPE kernel](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/pos_enc.cuh) and [vLLM Qwen3 vision model](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_vl.py) share sine and cosine values across attention heads.
Compute the original position coefficients once per row if another cache changes their rounded values.
Keep the original bias rounding, BF16 coefficients, paired loads, and output conversion.

The [FA4 head layouts](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py) round general head widths up to multiples of 16.
Compare these layouts with the pinned CUTLASS collective before you change storage for the 72-dimensional Vision heads.

Group TMA descriptor dimensions to fit the shared layout before you decrease the logical head width.
Apply the pinned test for a contiguous prefix. Keep its specified descriptor bit update.
Examine workload correctness after raw descriptor acceptance. Keep the physical row stride. Examine each output consumer.

The [CUTLASS callbacks](https://github.com/NVIDIA/cutlass/blob/v4.8.0/include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp) can combine bias and residual operations in a projection epilogue.
Keep the intermediate BF16 rounding before the residual addition.
The [TensorRT-LLM normalization kernels](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/layernormKernels.cu) include different variance methods.
Keep the original centered variance and reduction tree when you combine residual addition and normalization.

The [vLLM vector-copy change](https://github.com/vllm-project/vllm/pull/43437) replaces scalar 16-bit accesses with 128-bit copies.
Compare generated instructions with the public alignment contract before you use wider accesses.
Fewer copy instructions do not reduce the payload bytes.

## Optimization procedure

Use [TileSight](https://github.com/tile-ai/TileSight) to compare tile schedules before GPU experiments.
Its [model limitations](https://arxiv.org/html/2607.22432v1#S7) include small-batch latency and B200 memory topology.
Compare the model with the kernel's data types, layouts, dependencies, and physical memory transactions.
Include TMEM allocation, masked loads and stores, partial results, and merge operations.
Do a check for changes to candidate order with uncertain latency parameters.
Use numerical checks and full-boundary measurements to select the implementation.

Use these lessons to select another candidate:

- Include merge work when you compare K partitions. A faster projection alone is not sufficient.
- Before you extend a fused projection to more rows, do a check of register and shared-memory use.
- Treat matrix completion, accumulator reads, and buffer reuse as separate dependencies.
- Use occupancy queries as predictions. Compare them with active-cycle counters and resource lifetimes.
- Include repeated input reads when you move normalization into projection blocks.
- Keep the original reduction order when local normalization supplies shared-memory matrix inputs.
- Compare the work for each thread before you change reduction parameters. Keep the original partial sums and addition order.
- Check compiled vector loads and stores. Pair BF16 conversions for packed checkpoints. Keep the RN conversion for each element.
- Query active clusters with the compiled kernel and its full launch configuration before a cluster fusion experiment.
- For input multicast, keep blocks on different output columns. Examine the matrix instruction group, pipeline stages, and active clusters.
- Include cluster synchronization and scheduling limits when you compare output fusion with a separate launch.
- For paged loads, calculate page bases before the vector loop. Keep page-boundary and masked-tail addresses unchanged.
- Limit column blocks when row copies already supply enough parallel work.
- Before you remove a copy, do a check of buffer lifetime and graph inputs.
- Combine fixed metadata arrays by data type. Keep aligned views and fixed graph addresses.
- Exclude invalid K groups from TMA byte counts, matrix products, and accumulator reads. Zero scales do not prevent NaN propagation.
- Use a separate process without a profiler for comparison timings.
- Make sure that the profiler collected each selected kernel.
- Examine the two measurement orders before you accept a small gain from alternating trials.
- Run HTTP comparisons without concurrent reference calculations. Keep the measured speculative acceptance trace.

Use a temporary external directory for each experiment.
Keep the current reports until the experiment's review and applicable acceptance audit are completed.
Then delete raw reports, profiler files, and obsolete experiment binaries.
Keep the current runtime binaries and their manifests.
Keep brief reusable lessons in this guide. Do not put measurement results in project documents.
