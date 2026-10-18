# GPU peak calibration

Use measured resource rates for each GPU UUID.
Operator and HTTP roofline calculations must have a matching calibration file.
There is no nominal peak fallback.

## Command

Set `GPU_UUID` to one UUID from the device inventory.
Set `PEAK_PROFILE` to a new JSON file external to the repository.
Set `CUDA_HOME` and `OH_MY_VLLM_KERNEL_CACHE` as specified in the [kernel setup](operators/development.md#tools).

```bash
nvidia-smi --query-gpu=uuid,name --format=csv
export OH_MY_VLLM_GPU_UUID="$GPU_UUID"
scripts/with-env.sh scripts/with-gpu.sh python -m development.peaks --output "$PEAK_PROFILE"
```

Do this command again for each available GPU UUID with a different output file.
The GPU wrapper reserves one idle B200 for each command.
The CUDA Driver UUID must agree with the wrapper selection.
Use that UUID and file for subsequent operator or HTTP measurements.
Do not combine different GPU calibrations in one operator audit.

## Resource measurements

| Resource | Method | Configurations | Work per invocation |
|---|---|---|---|
| HBM | CUDA vector copy with pseudorandom input | 1 or 2 GiB per buffer, 4, 8, or 16 blocks per SM | Read bytes plus write bytes |
| FP8 | Dense cuBLASLt TN GEMM, E4M3 inputs, FP32 accumulation and output, scalar scales of 1 | Square dimensions 4096, 8192, 16384, up to eight heuristic algorithms | `2 * M * N * K` FLOPs |
| BF16 | Dense cuBLASLt TN GEMM, BF16 inputs, FP32 accumulation and output | Same GEMM configurations | `2 * M * N * K` FLOPs |
| TF32 | Dense cuBLASLt TN GEMM, FP32 inputs, `CUBLAS_COMPUTE_32F_FAST_TF32` | Same GEMM configurations | `2 * M * N * K` FLOPs |
| FP32 | Native scalar FMA instructions | 16 independent accumulators, 4096 iterations, 2, 4, or 8 blocks per SM | Two FLOPs per FMA |
| FP64 | Native scalar FMA instructions | Same scalar configurations | Two FLOPs per FMA |

All custom kernels use 256 threads per block.
Each copy buffer must be more than four times the L2 capacity.
The stored result includes a value from each accumulator.
The build check must find native FP32 and FP64 FMA instructions in the binary.
The output sample checks must find no errors.

Each configuration has at least 200 ms of warmup before measurement.
Each of seven samples uses CUDA events around ten invocations on the same stream.
Allocation, input preparation, and algorithm selection occur before timing.
Calculate seconds per invocation as `elapsed_milliseconds / 10000`.
For each resource, select the highest `work / median(seconds)` across configurations.
Do not select the fastest individual sample.

## Identity and calculation

The external file keeps each configuration, work count, and timing sample.
It gives the UUID, GPU properties, driver, memory clock, power limit, and SM clock snapshots.
It binds calibration sources, compiler options, native binary, and the loaded cuBLASLt image.
The consumer checks the recorded work and recalculates the selected rates.
Missing rates, changed sources or binaries, and a different UUID prevent measurement.
After a driver, memory clock, or power limit change, do a new calibration.

These rates give the measured workload and operating conditions.
They do not prove the maximum throughput of all possible workloads.
The latency lower bound is a roofline estimate based on these measured rates.
An operator can have a higher throughput than a calibration rate. Keep efficiencies above 100% for analysis.
Do not fit resource rates to measured operator or HTTP latency.

Use the [operator formula](operators/evaluation.md#performance-calculation) and the [HTTP trace calculation](performance.md#lower-bound-calculation).
Operator efficiency has no minimum requirement. The HTTP efficiency requirement is 50% for each configuration.
Before each delivery, do the operator measurements again.
Then update the tables in the two README files.
Keep calibration files and raw timing reports external to the repository.
