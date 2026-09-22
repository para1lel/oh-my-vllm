# CUDA kernel development

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
