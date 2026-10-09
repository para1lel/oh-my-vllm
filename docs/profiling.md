# Profiling and diagnostics

Use measurements to identify prefill and decode latency gaps.
Keep profiling isolated from formal timing.
TileFoundry arithmetic, memory, and roofline results are estimates.
Use the phase-latency protocol in [requirements](requirements.md).

## Logs and correlation

Rust uses `RUST_LOG`. Python uses `OH_MY_VLLM_LOG_LEVEL`.
Default logs include UTC time and run identity without per-step I/O.
Debug logs add request/step correlation and host durations.
Host execution time is not CUDA kernel time.

Use CUDA Events or profiler traces for device attribution.
Keep raw logs and traces in external storage.

## Steady-state audit

Formal collection examines FlashInfer, third-party Triton, TileLang, and native CUDA cache trees, with text files.
Unset or missing required roots fail the audit.
Finish compilation and capture before measured repetitions.
Source and cache hashes identify the observed run.
The audit cannot exclude silent in-memory recompilation or arbitrarily short interference between process polls.

See [testing](testing.md) and [acceptance](acceptance.md).

## CUDA fault attribution

The default owned launch uses `cudaPeekAtLastError` and keeps prior runtime errors.
An asynchronous failure can appear at a later launch or host synchronization.
The message identifies the observation point, not a proven faulting instruction.

For a diagnostic process, set the two flags before Python starts:

```bash
scripts/with-gpu.sh scripts/with-env.sh env OH_MY_VLLM_CUDA_DEBUG_SYNC=1 OH_MY_VLLM_ENFORCE_EAGER=1 OH_MY_VLLM_KERNEL_BACKEND=cuda python -m tests.gpu_cuda_error_case eager
```

Debug mode examines existing errors and synchronizes the selected stream before and after each owned launch.
It rejects graph capture.
Prior/after-launch labels identify observation order. Earlier asynchronous faults can still appear there.
Use `compute-sanitizer --tool memcheck` from your compatible CUDA installation for device details.

Debug synchronization times cannot supply acceptance measurements.
See the [CUDA error API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__ERROR.html).

## Copies and kernel observations

Misaligned contiguous BF16 KV views use an observable alignment copy.
`oh_my_vllm.kernels.decode_attention.unaligned_cache_clone_count()` reports calls.
Warnings report copied bytes at counts 1,2,4,8 and subsequent powers of two.
Aligned views take no clone path.

`python -m development.kernels.observations` profiles full CUDA and pinned TileLang operations.
The report separates HIR estimates from Nsight counters and resources.
It verifies immutable reference identity and requested counter completeness.
NVTX ranges do not include warmup/JIT.
Raw profiler CSV stays in external storage.

Cancellation stops the owned profiler process group.
Use [kernel development](kernels.md#analysis-and-profiling) for the command and limitations.


## Large operator tuning

Use effective model shapes before a tile search.
Record elapsed time, DRAM and L2 throughput, L2 hit rate, register count, shared storage, occupancy, and warp stalls.
Use HIR estimates to select a hypothesis. Verify it with CUDA Events and Nsight counters.
An occupancy increase alone is not a latency result.

Swizzle selection used five randomized timing rounds.
SM selection and the subsequent swizzle confirmation used ten rounds with random order.
Candidates must keep rounding, scales, and persistent writes equal.
The previous 230 cases, sixteen fused cases, and twenty-eight other FP8 projection cases supply operator acceptance.
Full-model phase collection supplies framework acceptance.

| Optimization | Diagnostic observation | Selection |
|---|---|---|
| FP8 gate/up GEMM traversal | At 32144 by 34816 by 5120, DRAM was 75.04% and L2 hits were 26.10%. Swizzle 8 changed median time from 9.235 to 6.989 ms. | Swizzle 8 for M at least 2048 and N at least 32768. |
| Small FP8 gate/up GEMM | At M 624, one-SM time was about 0.141 ms and two-SM time was about 0.170 ms. | Use two-SM tiles at M at least 4096 for the wide projection. |
| Four-channel convolution | At 32144 rows, scalar time was 1.603 ms and vector time was 0.624 ms. Warp instructions decreased from 1.484 billion to 0.589 billion. | Four adjacent channels per thread, eight rows for large inputs, four rows for smaller inputs. |
| SiLU and FP8 quantization | At 32144 rows, one-row time was 1.130 ms and four-row time was 0.711 ms. At 624 rows, two-row time was 0.0165 ms. | Four rows per warp for large inputs. Use two rows for smaller model inputs. |
| Gated RMS | At 32144 token rows, one-row time was 0.399 ms and four-row time was 0.310 ms. | Four rows per warp for large inputs. Use one row for small inputs. |
| Residual RMS and FP8 quantization | At 32144 by 5120, the two operations took 0.325 ms and the fused operation took 0.195 ms. | Keep residual and normalized BF16 rounding. Remove the read and write of the normalized array. |
| Single-row normalized projection | One-row gate/up diagnostic time was about 0.03431 ms with fusion and 0.03319 ms with the previous CUDA chain. | Use the previous CUDA chain for one row. |

The table uses diagnostic operator microbenchmarks and a different profiler replay.
Short-output full-model diagnosis has a different scope.
They do not supply the fifteen-row phase verdict.
The convolution vector path had zero shared-memory bank conflicts.
Its DRAM throughput was 15.80%, SM throughput was 83.38%, and short-scoreboard stalls were 2.43%.
This profile supports the decrease in instruction and shared-memory overhead.

Residual RMS/quantization candidate measurements used ten rounds with random order and forty graph replays per timing sample.
FP8 bytes, scales, and residual sums were equal to the two operations.
The fused profile used 40 registers per thread and 64 bytes of static shared storage.
Its DRAM throughput was 46.34%, SM throughput was 74.66%, and occupancy was 57.87%.
These counters came from a different profiler replay.
Short-output model diagnosis after this fusion had a prefill median of about 1.39 s.

K=256 GEMM tiles, smaller M tiles, and two-SM down projections did not show a stable paired gain.
They are excluded from dispatch. Keep unsuccessful attempts in external evidence storage.
N=256 tiles increased the large gate/up diagnostic time from about 6.72 to 19.39 ms.
Dispatch does not use these tiles.
