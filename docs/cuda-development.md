# CUDA kernel development

## Medium residual RMS cache policy

Only2048–4095-row residual RMS now uses a streaming store cache hint for its two
outputs. This preserves the stored BF16 bits and stream visibility while reducing
input eviction in the operator workload. The26-case dirty-source RMS diagnostic
passes all decisions; the previously failing2496-row case saves about1.30us in
paired mean latency. This is not complete operator/framework acceptance, and
possible downstream cache misses still require the final end-to-end gates.

Full CUDA correctness passes160 tests plus28 subtests, and FP64 boundary checks
at2047/2048/4095/4096 rows pass with exact residual sums. A compiler incompatibility
in the first packing intrinsic was fixed before these successful reruns; its test
process was stopped. Independent review, separate TileFoundry/Nsight observations
and all-file checks pass. Evidence is in
`bench/baseline/2026-09-22-cuda-rms-stream-progress.json`.
All owned GPU programs exited. Default remains TileLang; migration continues.

## Gated RMS specialization and complete-matrix update

Clean d45c4f2 completes all173 cases:111 pass,62 fail, with no detected GPU
interference. All16 attention cases pass. The2496-row residual RMS case fails
this complete collection despite earlier subset wins; its advantage is not yet
reliable. Remaining failures include append, small quantization/rotary, Q/K
normalization, recurrent updates, gates and convolution. Full acceptance is pending.

A new48-head/128-wide gated RMS path retains values in registers and uses a stable
sigmoid with denominator in[1,2]. It keeps FP32 gating without intermediate BF16
rounding and supports the original strides/epsilon. The dirty-source gated RMS
subset passes13/13 cases. Full CUDA correctness passes160 tests plus28 subtests.
Independent review, extra FP64 packed/nonunit-stride and large-weight/negative-gate
checks pass. All finite BF16 gate encodings were also checked with fixed input1
and weight0.25; this is not exhaustive over input/weight combinations or bitwise
proof. No original tolerances change. Separate TileFoundry/Nsight observations,
complete-matrix decisions and subset evidence are summarized in
`bench/baseline/2026-09-22-cuda-gated-rms-progress.json`.

All-file checks pass and all task-owned GPU programs exited. Default stays
TileLang; further kernel tuning and complete framework acceptance remain.

## RMS launch and vector-width tuning

Model-width RMS now selects 128 threads for medium batches, 160 for large plain
RMS and 320 for large residual RMS. Large residuals use eight-element vectors
only when input/residual are16-byte and weights32-byte aligned; otherwise the
four-element or generic path remains. Compile-time divisibility checks ensure
complete row coverage. Reduction order changes; existing tolerances are retained.

The final unchanged-source diagnostic passes26/26 RMS cases. Earlier diagnostics
pass24/26 then25/26; the latter has one unusually slow CUDA sample and fails the
confidence bound despite faster medians in all three rounds. No samples were
excluded and no interference was detected; its exact cause remains unresolved.
All attempts are summarized in `bench/baseline/2026-09-22-cuda-rms-progress.json`,
with separate TileFoundry/Nsight observations. These are dirty-source subsets,
not full operator or framework acceptance.

The full CUDA suite passes160 tests plus28 subtests. Temporary FP64 checks cover
2047/2048/4095/4096-row dispatch boundaries, nondefault epsilon, and both alignment
fallbacks at4096 rows; residual sums remain bitwise equal to the reference.
Independent review and all-file checks pass. All owned GPU programs exited.
Default remains TileLang; remaining operator tuning and full acceptance continue.

## Attention fragment reuse and merge optimization

The attention diagnostic now passes the speed decision for all 16 static cases,
with three rounds of 20 interleaved pairs and no detected GPU interference.
This dirty-source subset is not full acceptance. Q/K and probability/value MMA
fragments are reused across output tiles; static register indices eliminate
local-memory spills. The merge uses shared split weights and vector output.
Ungrouped KV double buffering is limited to four queries to preserve occupancy
for larger eager batches. The sampled final eager profile reports zero local
loads/stores; TileFoundry estimates remain separate from measured counters.

The unchanged full CUDA suite passes 160 tests plus 28 subtests. Actual-model
ordinary and MTP4 eager FP64 probes pass, including recurrent states and draft
attention. Forced-length probe text is not semantic or agentic acceptance.
Merge reduction order changes, so numerical tolerance results do not establish
bitwise equivalence. Independent review and all-file checks pass. Summarized
source identities, timings, hardware observations and probe coverage are in
`bench/baseline/2026-09-22-cuda-attention-progress.json`.

RMS, Q/K normalization, KV append and other failed cases still need tuning.
The latest complete clean matrix remains 80/173; do not add subset pass counts.
Full operator and framework performance acceptance remain outstanding; default
backend stays TileLang. All task-owned GPU programs have exited.

## Packed SiLU follow-up

Cleancb867d4 completed all173 operator cases without detected GPU interference:
80 pass,93 fail. The next diagnostic uses width-specialized FP8 kernels and
paired BF16 SiLU multiplication. All13 fused-SiLU cases satisfy the existing
three-round speed decision, including the previously failing medium shape;
this dirty-source subset is not full acceptance. Full CUDA correctness remains
160 tests plus28 subtests. Independent review, layout/dtype boundary checks and
all-file hooks pass. Separate TileFoundry/Nsight summaries and source identities
are recorded in `bench/baseline/2026-09-22-cuda-silu-progress.json`.

The fast exponential and paired multiply preserve both BF16 rounding boundaries.
Temporary SM100/CUDA13.1 exhaustive checks found zero differences for rounded
SiLU over every finite BF16 input and for BF16-pair multiplication versus FP32
multiplication followed by BF16 rounding over all finite BF16 operand pairs.
These checks include signed zero, subnormals, overflow and underflow; they do not
establish equivalence for other architectures/toolchains or arbitrary FP32 SiLU.
No temporary tuning scripts or raw traces are retained in the repository.
Attention, Q/K normalization, KV append and other failed cases remain to be tuned;
no final CUDA model/performance/service acceptance is claimed. Default remains
TileLang. All task-owned GPU programs have exited.


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
