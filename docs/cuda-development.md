# CUDA kernel development

## Vector-kernel optimization progress

Clean7437e0f completed the full173-case operator matrix without detected GPU
interference:48 pass and125 fail. The subsequent vector-kernel diagnostic covers
91 normalization/quantization cases:55 satisfy the speed decision, versus23 of
those same cases before this change. Dirty source and partial coverage make the
new collection diagnostic only. Per-source results and paired hardware summaries
are in `bench/baseline/2026-09-22-cuda-vector-progress.json`; no complete CUDA
operator or framework acceptance is claimed.

Native quantization now tiles rows and scaling groups, uses unsigned warp REDUX
on nonnegative FP32 magnitudes, and packs fused SiLU loads/FP8 stores. Small fused
rows and widths exceeding CUDA grid.y capacity use a flat grid. Exact division
and both BF16 rounding boundaries remain. Model-width5120 RMS retains values in
registers and uses aligned vector loads with guarded scalar/general-layout paths;
residual addition rounds BF16 pairs before normalization. Fixed fused-RoPE
frequencies use FP64 read-only values; phase reduction remains FP64. Fresh-output
kernels declare nonaliasing pointers; in-place state/cache kernels do not.

Full CUDA correctness passes160 tests plus28 subtests. Supplementary boundary
checks cover nondefault RMS epsilon, unaligned input/weight storage, FP16/FP32
quantization, dispatch boundaries and extremely wide quantization. Existing
long-position FP64/stream/graph checks and all-file hooks pass. Independent
review found and fixed the quantization grid.y width limit. Large RMS, small
quantization, fused SiLU and attention still need tuning; the default stays
TileLang. Test/profiler workers exited and no task-owned GPU process remains.


The accepted comparison sources are copied byte-for-byte into
`python/oh_my_vllm/kernels/tilelang_reference`. Their source hashes, original
commit, dependency lock hash and TileFoundry pin are recorded in
`development/kernels/tilelang-reference.json`. Do not tune the reference.

Set `OH_MY_VLLM_KERNEL_BACKEND=cuda` before Python starts to select the new
backend. Unsupported entries raise instead of using TileLang. During migration,
the default remains TileLang; change it only after complete CUDA acceptance.
Native code uses independent TVM FFI, the caller's current CUDA stream and SM100a.
TileFoundry is not a runtime/build dependency of native kernels. All compilation
must finish before CUDA Graph capture and formal measurement.

Current milestone: all owned kernel entries have native CUDA implementations,
including FP64-phase fused RMS/RoPE, FP32 recurrent GDN, convolution and paged
attention. The existing full CUDA suite passes153 tests plus24 subtests after
the attention-merge optimization. Independent
production-entry FP64 packed Q/K, int32/int64 positions near262144, nondefault
stream and changed-input graph replay checks pass. These are correctness results,
not model or performance acceptance. Default remains TileLang.

The formal static matrix is in `development/kernels/cases.py`; full-operation
fixtures retain required snapshots and merge operations. It includes distinct
three-sequence dispatch, proposal/catch-up metadata and eager graph-cache fallback.
Valid FA pages exclude null page0. No dynamic shape extraction is used. Run:

```bash
scripts/with-env.sh python benchmarks/kernels.py --list
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_KERNEL_BACKEND=cuda CUDA_HOME=/usr/local/cuda-13.1 TVM_FFI_CUDA_ARCH_LIST=10.0a python benchmarks/kernels.py --output /tmp/cuda-kernels.json
```

`--operations` selects diagnostic subsets; partial coverage cannot pass the full
matrix gate. Dirty source runs are diagnostic only. The collector records source
hashes, GPU identity, all paired samples, confidence decisions and separate
TileFoundry estimates. GPU contention or changing sources invalidates collection.
An initial diagnostic was invalidated by an external GPU entrant; it establishes
no acceptance result. Attention and other small-shape paths need further tuning.
The TileFoundry `TileLang` twin always binds the frozen implementation, regardless
of the selected production backend.

`development/kernels/comparison.py` requires at least three rounds and20 paired
samples per round. It uses a reproducible hierarchical bootstrap: resample rounds,
then pairs within a round, and require the one-sided95% lower bound of mean time
saved to exceed zero. Every round must also have a lower CUDA median. Formal
harness runs must alternate backend order, warm both paths, restore mutable state,
and retain complete raw timing pairs; profiling runs cannot supply these times.

Follow REQ-KERNEL-002 for static case selection and complete acceptance. Keep
hardware metrics, compiler resource reports and TileFoundry HIR estimates distinct.
The HIR currently lacks FP64; it cannot certify long-position RoPE phase accuracy.
Use unchanged independent references and full model tests for that boundary.

Verified this milestone: Nsight Compute can read B200 hardware counters;
filtered quantize_kernel collection succeeds. TileFoundry analyze and Nsight CSV
import both run. Nondefault CUDA stream and changed-input CUDA Graph replay pass
exact FP8 comparison; TileLang14-case regression also passes. Complete valid maximum-case performance and final framework acceptance remain
pending. Stop owned GPU processes after every run.

TileLang kernel bodies exist only in `tilelang_reference/`. Production modules
contain validation, public wrappers and explicit factory bindings; they do not
carry a second copy of the TileLang DSL. The frozen whole-operation wrappers
remain for fair comparison, including their original snapshots and allocations.

Native attention uses explicit `mma.sync`/`ldmatrix` BF16 fragments with FP32
register accumulation, shared-memory sector swizzling and an integer-width-safe
position specialization. Both backend suites pass154 tests plus24 subtests after
the duplicate-body cleanup. Attention performance still requires further tuning;
correctness and successful PTX compilation do not establish a speedup.

For development observations, select an ID from `benchmarks/kernels.py --list`:

```bash
scripts/with-gpu.sh scripts/with-env.sh env CUDA_HOME=/usr/local/cuda-13.1 TVM_FFI_CUDA_ARCH_LIST=10.0a python -m development.kernels.observations --case attention-c426ebd5d5 --output /tmp/kernel-observations.json
```

This runs TileFoundry HIR analysis and separate Nsight Compute replays of both
complete backend operations. Warmup/JIT stays outside the NVTX range. Each launch
must contain every requested finite counter; unavailable metrics are reported
explicitly and produce a failing exit status. The report distinguishes static
estimates from measured counters, includes register/shared-memory resources,
source hashes and hardware identity, and cannot pass any performance gate.
Temporary raw CSV files are removed. Cancellation terminates the owned profiler
process group, including GPU workers. CPU tests cover incomplete/nonfinite
metrics and cancellation cleanup; actual paired attention profiling succeeds.

Contiguous BF16 inputs may have an unaligned storage offset. Native attention
handles Q scalar loads in that case; the production wrapper aligns KV for both
backends and Q for frozen TileLang without modifying its frozen source. Existing
numerical tolerances apply to the added Q/KV alignment regression cases. Latest
CUDA full-suite result is160 tests plus28 subtests; TileLang attention passes16.
Register softmax and shape-specific KV prefetch remain under performance tuning.
