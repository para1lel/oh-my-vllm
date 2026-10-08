# Profiling and diagnostics

Use measurements to identify the source of throughput or TTFT gaps.
Keep profiling isolated from formal timing.
TileFoundry arithmetic, memory, and roofline results are estimates.
The future roofline gate in [requirements](requirements.md) does not replace the current acceptance protocol yet.

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
Source and cache hashes bind the observed run.
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
