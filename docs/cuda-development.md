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

Current milestone: FP8 quantization and SiLU have CUDA implementations. The
existing14 FP8/fused-SiLU correctness cases pass, including exact quantization.
Other native kernels, static maximum-shape cases, measurement/metric integration
and full final acceptance remain in progress. No native speedup claim yet.

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
exact FP8 comparison; TileLang14-case regression also passes. Formal maximum-case
performance has not been collected. All stage GPU processes exited.
