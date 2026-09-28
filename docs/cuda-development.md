# CUDA kernel development

## Status

CUDA is the default custom-kernel backend on B200. The 147/147 operator cases
and 12/12 framework rows in [acceptance.md](acceptance.md) apply to historical
clean commit `c36d1c9`; current full-matrix and framework acceptance are pending
after audit remediation. Known
kernel-contract and observability issues are listed in
[audit-2026-09-23.md](audit-2026-09-23.md), under the `KRN-*`, `MNT-*` and `EVD-09`
entries. Read those before changing a kernel.

## Backend selection and frozen reference

- **Selecting the backend.** The backend is chosen once per process, at import.
  Setting `OH_MY_VLLM_KERNEL_BACKEND=tilelang` before Python starts selects the
  frozen comparison. An invalid name raises an error. Runtime identity reports
  the backend that was actually selected.
- **No silent fallback.** A missing CUDA entry raises `NotImplementedError`. It
  never falls back to TileLang.
- **Frozen sources.** The accepted TileLang sources are copied byte for byte into
  `python/oh_my_vllm/kernels/tilelang_reference`. The following are recorded in
  `development/kernels/tilelang-reference.json`:
  - their hashes
  - the original commit
  - the dependency lock hash
  - the TileFoundry pin

  Never tune the reference.
- **Where the kernels live.** Production modules hold validation, the public
  wrappers and explicit factory bindings. TileLang kernel bodies live only in
  `tilelang_reference/`.
- **Native build.**
  - Native code is `python/oh_my_vllm/kernels/cuda_backend/kernels.cu`. It is
    compiled lazily through independent TVM FFI `build_inline`, then loaded from
    the returned path with `load_module`. It targets `sm_100a` and runs on the
    caller's current CUDA stream.
  - `scripts/with-env.sh` isolates `TVM_FFI_CACHE_DIR` and sets
    `TVM_FFI_CUDA_ARCH_LIST=10.0a`. Direct launches must set that arch explicitly.
    A build sidecar binds the source/configuration and exact `.so` hash for
    verified cache reuse if nvcc is unavailable later.
  - Host CUDA 13.1 compiles it. Use `CUDA_HOME=/usr/local/cuda-13.1` when needed.
  - All compilation must finish before CUDA Graph capture and before formal
    measurement.
  - TileFoundry is not a runtime or build dependency.

## Implementation notes

These describe the current code and the numerical properties it relies on.

- **FP8 quantization.**
  - Rows and scaling groups are tiled. The kernel uses four-element vectors and
    an unsigned warp REDUX over nonnegative FP32 magnitudes.
  - Loads for fused SiLU are packed, and FP8 stores are packed.
  - A flat grid handles small fused rows and widths too wide for `grid.y`.
  - For finite BF16 maxima, the scale reciprocal uses `rcp.approx` plus one FMA
    residual correction. Scales themselves still use exact division.
  - Temporary SM100/CUDA 13.1 exhaustive checks over finite BF16 input/max pairs
    found zero FP8 differences. An exact-arithmetic review in the 2026-09-23
    audit agrees, assuming the `rcp.approx` error is at most 1 ulp.
- **SiLU and multiply.**
  - Uses a fast exponential and paired BF16 multiplication. Both BF16 rounding
    points are kept.
  - Exhaustive checks over every finite BF16 input, and over all finite operand
    pairs, found zero differences on SM100/CUDA 13.1.
  - Flat offsets are 32-bit (audit KRN-04).
- **RMS normalization.**
  - The model width is 5120. Values stay in registers, and loads are aligned
    vectors of width 4 or 8.
  - The thread count depends on row count and on whether a residual is added:
    ≥ 4096 rows use 320 (residual) or 160 threads; 2048–4095 rows use 128;
    fewer rows use 256.
  - Residual RMS with 2048–4095 rows uses a streaming store hint.
  - Misaligned or other layouts use the generic kernel.
  - Reduction order therefore varies with row count and alignment (audit
    MNT-03).
- **Gated RMS.**
  - Specialized for 48 heads × 128 width. Values stay in registers.
  - Uses a stable sigmoid with the denominator in [1,2], and FP32 gating with no
    intermediate BF16 rounding.
- **Q/K normalization.**
  - The packed 16-head, 10240-stride path uses aligned eight-element loads and
    paired FP32 arithmetic.
  - Other layouts use the generic kernel.
- **Convolution.**
  - Specialized for 10240 channels with token stride 16384. It uses four-element
    weight loads and a stable sigmoid.
  - Uses 4 rows per block for medium inputs and 8 for large ones.
  - Weight alignment and a conservative disjoint pool/input span check guard the
    specialized path.
  - Source states are snapshotted before candidate writes.
- **GDN recurrence.**
  - The model layout gives each of 16 lanes eight contiguous key values, uses four
    warps with two value rows per half-warp, and keeps FP32 persistent state.
  - Head, base/row alignment and disjoint-storage checks guard the restricted
    pointers. Otherwise the generic kernel runs. An in-place update of the
    request's own source is valid.
- **Full-attention preparation.**
  - One entry fuses Q/K RMS and RoPE, V layout conversion and the KV write, shared
    by the target model and MTP.
  - It keeps both BF16 rounding points and reduces the phase in FP64 before FP32
    sin/cos.
  - Negative slots skip only the KV write. Out-of-range slots are also skipped
    silently (audit KRN-02).
- **Paged decode attention.**
  - Uses explicit `mma.sync`/`ldmatrix` BF16 fragments with FP32 accumulation,
    shared-memory sector swizzling and split-KV with a separate merge.
  - Q/K and P/V fragments are reused across output tiles. There are no
    local-memory spills.
  - Ungrouped KV double buffering is limited to four queries.
  - Grouped verification assumes at most five query rows per request (audit
    KRN-01).
- **Other properties.**
  - Fresh-output kernels declare non-aliasing (`__restrict__`) pointers;
    in-place state/cache kernels do not.
  - Contiguous BF16 views may start at an unaligned storage offset. Native
    attention handles scalar Q loads; the production wrapper aligns KV for both
    backends and Q only for frozen TileLang (KRN-07 covers the KV clone).
  - The attention merge's reduction order changed during tuning; results meet
    the tolerances but are not bitwise-equal to earlier versions.
  - The gated-RMS gate sweep covered every finite BF16 gate with fixed input 1
    and weight 0.25; it is not exhaustive over input/weight combinations.
  - The TileFoundry `TileLang` twin always binds the frozen implementation,
    regardless of the selected production backend.

## Formal operator comparison

The formal static matrix is `development/kernels/cases.py`. It derives each
distinct implementation path's largest legal invocation per workload, statically
and deduplicated, as REQ-KERNEL-002 requires. Full-operation fixtures keep the
required snapshots and merges. Valid FA pages exclude null page 0.

```bash
scripts/with-env.sh python benchmarks/kernels.py --list
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_KERNEL_BACKEND=cuda CUDA_HOME=/usr/local/cuda-13.1 TVM_FFI_CUDA_ARCH_LIST=10.0a python benchmarks/kernels.py --output /tmp/cuda-kernels.json
```

- **Subsets.** `--operations` selects diagnostic subsets. Partial coverage and
  dirty-source runs cannot pass the full-matrix gate.
- **What the collector records:** source hashes, GPU identity, all paired
  samples, the decisions, and separately labelled TileFoundry estimates.
- **Invalidation.** GPU contention or a source change invalidates a collection.
- **Decision rule** (`development/kernels/comparison.py`):
  - At least three rounds of 20 pairs, with interleaved backend order.
  - Both paths warmed, and mutable state restored between runs.
  - A faster CUDA median in every round.
  - A one-sided 95% hierarchical-bootstrap lower bound on mean time saved that
    is greater than zero.
- **What the harness compares.** It compares time only (audit EVD-09).
  Numerical agreement comes from the FP64 test suite.

## Development observations

```bash
scripts/with-gpu.sh scripts/with-env.sh env CUDA_HOME=/usr/local/cuda-13.1 TVM_FFI_CUDA_ARCH_LIST=10.0a python -m development.kernels.observations --case attention-c426ebd5d5 --output /tmp/kernel-observations.json
```

- **What it runs:** TileFoundry HIR analysis, plus separate Nsight Compute replays
  of both complete backend operations. Warmup and JIT stay outside the NVTX range.
- **Required counters:** missing or nonfinite requested counters fail the run.
- **Report contents:**
  - HIR estimates and measured counters, labelled separately
  - register and shared-memory resources
  - source hashes and hardware identity
- **Not a performance gate:** the report cannot pass any performance gate.
- **Cleanup:** raw CSV files are temporary. Cancellation terminates the owned
  profiler process group.
- **HIR limitation:** the pinned HIR has no FP64. It cannot certify the precision
  of the long-position RoPE phase. Use the FP64 references and full model tests
  for that.

## Change checklist

1. Preserve the wrapper contracts and the documented rounding points. Put new
   preconditions in the wrapper or in a C++ `ICHECK`, not only in upstream
   callers.
2. Run the full GPU suite with its tolerances unchanged, plus the actual-model
   ordinary/MTP4 FP64 probes.
3. Re-run every formal operator case whose implementation changed. Report
   dirty-source subsets only as diagnostics.
4. Re-check the framework rows affected by the hot path before claiming that the
   12-row result still holds.
5. Record summarized evidence in `bench/baseline`. Keep temporary scripts and
   raw traces outside the repository. Stop all GPU programs you started.
