# MTP throughput gap investigation — 2026-09-22

## Scope and distance

The refreshed baseline is official vLLM main at
`e9f169d16b9408bb9ae44f75072b91a5521d733c`, with its official cu130 wheel,
V2 GPU runner and Torch2.13.0. The project uses independent Torch2.14.0,
Triton3.8.0 and FlashInfer0.6.18.post1; it does not import/link vLLM.
The reference workload is MTP4, batch4, input32768, output4096, one B200.
Baseline median throughput is729.621 tok/s; the95% gate is693.140 tok/s.

On committed5166885, all six ordinary rows and MTP batch1/2 passed formal
throughput/TTFT/stability/audit checks. Three prefix rows failed TTFT; MTP batch4
attempts were interrupted by external GPU arrivals, retaining diagnostics near
683 tok/s. These are not a complete final acceptance.

Subsequent GDN prefill normalization and native short-prefix attention gave
689.346 tok/s in a single measured diagnostic:0.550% more throughput, or130ms
less total batch time, was needed. MTP prefill's FA2-to-TRT ragged change then
measured692.012 tok/s. This still needs formal two-warmup/five-repetition
verification and is slightly below the gate; no estimated speedup is accepted.

## Matched-shape decode kernel comparison

Both diagnostic profiles execute20 target query tokens for four requests near
step30 of32768-input/512-output MTP runs. The table sums actual CUDA kernel
events, not CPU wrapper times or nested CUDA graph annotations. Baseline target
and draft FMHA calls are separated by execution order (16 target plus4 draft).
Our draft figure includes split-KV merge kernels.

| Work per step | Project ms | vLLM ms | Observation |
|---|---:|---:|---|
| FP8 linear GEMMs,272 calls |6.944|7.176|Project TRT GEMMs are already competitive |
| Vocabulary projections,5 calls |1.761|1.869|CuTe-DSL projection is not the main deficit |
| Target full attention,16 calls |1.397|1.386|Native target path is essentially comparable |
| Draft attention,4 calls including merges |0.964|0.338|Largest clearly isolated unfavorable kernel difference |
| All CUDA kernels |14.279|14.095|Other faster project kernels offset much of the draft loss |
| CUDA kernel count |1349|1125|Project still launches more separate pointwise/metadata kernels |

These are individual profiler diagnostics on separate B200s and CPU affinities,
not formal throughput comparisons. Profiler overhead and GPU overlap mean summed
kernel times cannot be directly substituted for wall time. The baseline profile
uses a warmed iteration; our selected step uses already-captured full-batch
graphs before its short run drains. Framework/library versions also differ.

## Why the paths differ

1. **Draft attention and cache semantics.** Project MTP row p pairs target
   hidden[p-1] with input[p]. Position0 is absent, preserving Rust's immutable
   token-prefix hash across page boundaries. Its split-KV Triton kernels explicitly
   exclude0. vLLM's autoregressive speculator pairs hidden[P] and token[P+1]
   using its own cache/metadata contract and executes native FMHA for draft too.
   Merely enabling our target's native path for draft would include an invalid
   row. This needs a correct adapter or a faster first-valid-position kernel,
   not removal of the mask or mutation of shared prefix boundaries.
2. **MTP prefill dispatch was still FA2.** Target long prefill had migrated to
   TRT ragged, but the independent MTP layer retained FA2. A controlled first0-
   excluded gather comparison measured FA2/TRT35.18/9.55ms for32K query/32K KV,
   and241.40/71.90ms for32K query/131K KV. The new path keeps the same gathered
   valid tokens, causal semantics and FP32 softmax. This is a separate prefill
   improvement, not a fix for the draft decode kernel above.
3. **Pointwise fusion differs.** vLLM has fused post-convolution GDN recurrence,
   gating and output normalization, fused Q/K RMS+RoPE+gate preparation, and
   compiler-managed residual/quantization fusions. We use smaller project-owned
   kernels and explicit immutable convolution snapshots. Our recurrent kernel
   alone is faster in this profile, so comparing it directly to vLLM's broader
   fused GDN kernel would be misleading. Compare the complete sequence of work.
4. **Accepted-token bookkeeping crosses the CPU.** Our greedy sampling returns
   `.tolist()`, verifies candidates and constructs draft metadata on CPU, then
   dispatches catch-up/proposal graphs. vLLM keeps sampled/rejected counts on GPU
   and prepares draft inputs in GPU kernels. In our trace, target output reaches
   host around12.54ms and catch-up GPU work begins around13.20ms. This exposes
   a potential critical-path gap, but does not establish an equivalent net gap
   versus vLLM. The11.12ms CPU sampling annotation mostly waits for target GPU
   execution; the D2H copy itself takes only3.52us. Do not call that11ms of
   CPU or ZMQ overhead. Rust scheduling/ZMQ is not established as the bottleneck.

MTP acceptance is not collapsing: a full-batch baseline diagnostic used1121
steps; the latest project diagnostic used1117. This small difference does not
explain the per-step draft attention cost. Token trajectories can differ with
valid BF16 backend rounding, so this is a work-count diagnostic, not an assertion
of identical generated tokens.

## Controlled checks and priorities

- Changing only draft KV storage from NHD to head-major did not help the current
  kernel: batch4/query1 measured0.25625/0.25510ms; query5 measured0.21202/0.21999ms.
  Do not undertake a layout rewrite based only on the upstream layout choice.
- Quantization warp-count scans did not improve on the existing launch settings.
- GDN prefill Q/K normalization fusion removes repeated FP32 intermediates:
  the operator diagnostic fell from1.661ms to0.081ms at32768 rows. It retains
  FP32 norm/epsilon and BF16 output and is checked against FP64.
- Native short-prefix context attention reduces the batch4/624-query/32768-KV
  operator from6.54ms to1.68ms. The project TTFT diagnostic falls from216ms to
  138ms. Long target prefills retain ragged attention because the native paged
  prototype was slower there.
- First complete formal measurements of these prefill changes. If the MTP4
  throughput gate still fails, prioritize the explicitly masked draft attention
  kernel and accepted-token GPU bookkeeping. Keep prefix isolation, constraint
  decoding, all state snapshots and fixed block784; avoid speculative broad
  rewrites of GEMM or Rust scheduling.

## Source and evidence

Project sources: `python/oh_my_vllm/worker/mtp.py`, `worker/model_runner.py`,
`worker/sampler.py`, `kernels/decode_attention.py`, `kernels/mtp_attention.py`.
Upstream source at the frozen SHA:

- [V2 autoregressive speculator](https://github.com/vllm-project/vllm/blob/e9f169d16b9408bb9ae44f75072b91a5521d733c/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py)
- [V2 model runner](https://github.com/vllm-project/vllm/blob/e9f169d16b9408bb9ae44f75072b91a5521d733c/vllm/v1/worker/gpu/model_runner.py)
- [FlashInfer attention backend](https://github.com/vllm-project/vllm/blob/e9f169d16b9408bb9ae44f75072b91a5521d733c/vllm/v1/attention/backends/flashinfer.py)
- [Fused GDN CUDA implementation](https://github.com/vllm-project/vllm/blob/e9f169d16b9408bb9ae44f75072b91a5521d733c/csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu)
- [RMS/quantization fusion](https://github.com/vllm-project/vllm/blob/e9f169d16b9408bb9ae44f75072b91a5521d733c/vllm/compilation/passes/fusion/rms_quant_fusion.py)

The [current Developer Guide](https://docs.vllm.ai/en/latest/contributing/) is
background; measurements and code comparison use the frozen SHA above.
`bench/baseline/2026-09-22-performance-gap.json` retains summaries and raw-log
hashes. Raw profiler traces stay under `/tmp` and are deliberately not committed.
