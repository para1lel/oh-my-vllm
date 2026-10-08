# Kernel development

CUDA is the default owned backend on B200.
The pinned TileLang backend is an explicit comparison.
Third-party libraries can use Triton internally.
See [acceptance](acceptance.md) for measured source identity and [audit](audit.md) for open limits.

## Backend and provenance

Set `OH_MY_VLLM_KERNEL_BACKEND=tilelang` before Python starts to select the comparison.
An invalid backend or missing CUDA implementation raises an error.
There is no automatic TileLang fallback.
Public Python wrappers keep validation and factory bindings.
Pinned bodies stay in `python/oh_my_vllm/kernels/tilelang_reference/`.

Do not tune the reference.

`development/kernels/tilelang-reference.json` binds these identities:

| Identity | Value |
|---|---|
| Accepted reference record | `02bf35f98ceb6b40a2a813aa3e0cccbfa9067600` |
| Kernel implementation | `da75c02bd9e01556436877acba3243636dec80d6` |
| Runtime lock SHA-256 | `5f02f2051db2ba92fcdc1864283d4101eba331ed9c9677f3dac627f04c37e15a` |
| TileFoundry fork | `6b1b1493efcb55f3f2f327b5cc90a01181adf690` |

The manifest also records each reference file's SHA-256.
Keep these files and the runtime lock unchanged during unrelated work.
A changed lock/reference must have an explicit new comparison decision.

Native source is `python/oh_my_vllm/kernels/cuda_backend/kernels.cu`.
Independent TVM FFI `build_inline` compiles it lazily. `load_module` loads the returned module.
The target is `sm_100a`, on the caller's current stream.
The wrapper sets `TVM_FFI_CUDA_ARCH_LIST=10.0a`. Direct launches must set it explicitly.

A sidecar binds source/configuration and module SHA for validated reuse without available nvcc.
Finish compilation before capture and formal timing.
TileFoundry is not a native build or service dependency.

## Contracts and implementation

| Operation | Current properties |
|---|---|
| FP8 | Four-element vectors, warp REDUX, exact scale division, and packed stores. |
| BF16 finite FP8 scale reciprocal | `rcp.approx` with one residual correction and two `__fmaf_rn` operations. |
| Other FP8 reciprocals | Nonfinite maxima, FP16, and FP32 use exact division through `__fdiv_rn`. |
| SiLU/multiply | Fast exponential and paired BF16 multiplication keep the two BF16 rounding points. |
| RMS | Dense width 5120 uses aligned register-resident specializations and epsilon `1e-6`. |
| Gated RMS | 48 heads, width 128, stable sigmoid, FP32 gating without an extra BF16 rounding point. |
| Q/K | Packed sixteen-head rows, token stride 10240, aligned eight-element loads. |
| Convolution | 10240 channels, token stride 16384, guarded weight alignment, and source-state snapshots. |
| GDN | Guarded vector state updates, FP32 arithmetic, BF16 MTP or FP32 ordinary storage. |
| Attention preparation | Q/K RMS/RoPE, V conversion, and KV writes with FP64 phase reduction. |
| Decode attention | BF16 `mma.sync`/`ldmatrix`, FP32 accumulation, sector swizzling, split-KV, and a different merge operation. |

Generic CUDA paths handle other supported layouts.
Fast-path guards include alignment, strides, shape, and disjoint-storage checks.
State updates can validly overwrite their own source after required snapshots.
Direct recurrence FFI must use contiguous `int32`/`int64` index tensors, monotone starts, valid reads, and valid writes or `-1`.
GPU index contents are not copied to the host on each launch.

The batch plan validates logical slots against reported capacities.

Use different contracts for fresh-output non-aliasing and in-place state/cache operations.
Reject invalid wrapper inputs at the wrapper or C++ boundary.
Negative attention slots skip KV writes and keep Q.
Native MTP4 verification has at most five queries. DSpark verification has at most eight.
The production wrapper aligns misaligned KV views by copying and reports that cost.

See [profiling](profiling.md) for clone counters and fault attribution.

Reduction order can vary with row count and alignment.
FP64-reference tolerances apply. Bitwise RMS batch invariance is not a contract.
The gated-RMS sweep included each finite BF16 gate at input 1 and weight 0.25.
This test does not include all input/weight pairs.

For finite BF16 maxima, exact scale division gives a result in `[2.232142829e-13,7.565917940e35]`.
The reciprocal's approximate range is `[1.321716729e-36,4.480000066e12]`.
Scale and reciprocal stay FP32 normal values.
[PTX reciprocal rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp) specify the approximate instruction's error bound.
The B200 production regression includes 65280 finite BF16 encodings, 32640 finite maxima, signed midpoint-derived samples, and nonfinite fallback.

It also includes 126 positive/negative FP8 midpoints at maximum 448.
Final FP8 bytes and scales match the CPU RN reference for those tests.
This finite test does not include all pairs or prove identical intermediate quotients.

## Formal cases and timing

`development/kernels/cases.py` statically derives and deduplicates each path's maximum legal invocation in the twelve workloads.
FA fixtures exclude null page 0.
Full operations include snapshots, copies, and split merges.
`development/kernels/semantics.py` gives logical behavior. `twins.py` binds the pinned full operations.

DSpark adds static supplemental shapes for target verification and draft operators.
The original case IDs and eight pinned TileLang files stay unchanged.
`development/kernels/dspark-reference.json` binds the supplemental reference and its original manifest hash.
`dspark_tilelang_reference.py` supplies a different DSpark performance comparison.
Independent references from FP64 PyTorch supply the numerical comparison.
The new cases include BF16 hidden RMS rounding, head-128 YaRN Q/K preparation, context append, and seven-row draft attention.

Each new case has the same full-output, cache-write, three-round, twenty-pair, and confidence-bound gates.
Current-source supplemental operator performance acceptance results are not available.

```bash
scripts/with-env.sh python benchmarks/kernels.py --list
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_KERNEL_BACKEND=cuda python benchmarks/kernels.py --output "$EVIDENCE_DIR/operators.json"
```

Use an external writable `EVIDENCE_DIR`.
`--operations` selects diagnostic subsets. Partial coverage or dirty source cannot pass the full matrix.
GPU contention and source changes invalidate collection.
The collector records source, module, hardware, full output verification, all samples, and independently labeled estimates.

Before timing, compare full returns and written inference caches at existing tolerances.
Include FP8 scale strides and recurrent per-slot bounds.
Restore mutable state and do warmup on the two paths.
Each case must have three rounds of twenty interleaved pairs and faster CUDA medians in each round.
The one-sided 95% hierarchical-bootstrap lower bound on mean time saved must be positive.

There is no minimum percentage gain or new TileLang-relative framework ratio.

The formal output gate must use one fast and zero generic host dispatches for these cases:

| Operation | Fast-path input |
|---|---|
| `norm` | Dense nongated 5120 rows, aligned input/weight. |
| `add_norm` | Dense 5120 rows, aligned input/residual/weight. |
| `gated_norm` | Gated 48-head,128-wide rows. |
| `qk` | Packed 16-head rows,10240 token stride, aligned Q/K. |
| `recurrent` | 16 Q heads,48 V heads, aligned Q/K/state, disjoint state pool. |
| `convolution` | 10240 channels,16384 token stride, aligned weight, disjoint state pool. |

`variant_launch_counts()` measures host dispatch, not graph replay.
`append` is not in the formal matrix and has a focused fast/generic GPU test.
Its fast path must use width divisible by eight and aligned K/V/cache.

## Analysis and profiling

Install the pinned TileFoundry tools as specified in [development](development.md).
Its fork supplies HIR sin/cos evaluation, printing, and cost classification. It does not generate device code.
Use compatible released TileLang and OR-Tools versions. Do not maintain local forks of these libraries.
The editable TileFoundry source must stay available.

```bash
scripts/with-env.sh tilefoundry analyze development/kernels/semantics.py:Operators.silu "$EVIDENCE_DIR/silu-analysis.txt" --compute-cost --memory --roofline
scripts/with-gpu.sh scripts/with-env.sh python -m development.kernels.observations --case attention-c426ebd5d5 --output "$EVIDENCE_DIR/observations.json"
```

HIR arithmetic, memory, and roofline outputs are estimates.
The analysis topology is not the device placement.
The observation tool profiles full backend operations independently with Nsight Compute.
The NVTX range does not include warmup or JIT.
Missing or nonfinite requested counters fail the observation run.

The report labels estimates, counters, resources, and source identity independently.
It supplies no acceptance timing samples.

The pinned HIR has no FP64 dtype.
Its FP32 RoPE phase cannot show maximum-context accuracy.
Use independent references with FP64 computation and model tests instead.
Keep raw profiler CSV and temporary tuning in external storage.

## Change procedure

1. Read wrappers, contracts, tests, and the relevant audit entries.
2. Keep supported shapes, strides, dtypes, alias rules, state outputs, and rounding boundaries.
3. Use analysis to select hypotheses, then measure full operations with valid metadata.
4. Keep production dispatch deterministic. Do not start a runtime autotuning search.
5. Complete affected correctness and formal operator cases.
6. Complete affected framework rows before you claim continued performance acceptance.
7. Commit portable evidence with raw hashes. Keep temporary records in an external directory.
8. Complete a review by another agent and stop owned GPU programs.
