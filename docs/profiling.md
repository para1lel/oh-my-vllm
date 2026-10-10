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
Parsed counter rows stay in the external report. Temporary CSV files are removed after parsing.

Cancellation stops the owned profiler process group.
Use [kernel development](kernels.md#analysis-and-profiling) for the command and limitations.


## Large operator tuning

Use effective model shapes before a tile search.
Record elapsed time, DRAM and L2 throughput, L2 hit rate, register count, shared storage, occupancy, local-memory sectors, and warp stalls.
Use HIR estimates to select a hypothesis. Verify it with CUDA Events and Nsight counters.
An occupancy increase alone is not a latency result.

Swizzle selection used five randomized timing rounds.
SM selection and the subsequent swizzle confirmation used ten rounds with random order.
Candidates must keep rounding, scales, and persistent writes equal.
The 317 cases supply operator acceptance.
They keep the previous 230 cases and add fused projection, FP8 projection, full GDN prefill, and selected-output preparation cases.
Full-model phase collection supplies framework acceptance.

| Optimization | Diagnostic observation | Selection |
|---|---|---|
| FP8 gate/up GEMM traversal | At 32144 by 34816 by 5120, DRAM was 75.04% and L2 hits were 26.10%. Swizzle 8 changed median time from 9.235 to 6.989 ms. | Swizzle 8 for M at least 2048 and N at least 32768. At M at least 32144, the 34816 by 5120 two-SM path uses swizzle 16. |
| Small FP8 gate/up GEMM | At M 624, one-SM time was about 0.141 ms and two-SM time was about 0.170 ms. | Use two-SM tiles at M at least 4096 for the wide projection. |
| Four-channel convolution | At 32144 rows, scalar time was 1.603 ms and vector time was 0.624 ms. Warp instructions decreased from 1.484 billion to 0.589 billion. | Four adjacent channels per thread, eight rows for large inputs, four rows for smaller inputs. |
| SiLU and FP8 quantization | At 32144 rows, one-row time was 1.130 ms and four-row time was 0.711 ms. At 624 rows, two-row time was 0.0165 ms. | Four rows per warp for large inputs. Use two rows for smaller model inputs. |
| Gated RMS | At 32144 token rows, one-row time was 0.399 ms and four-row time was 0.310 ms. | Four rows per warp for large inputs. Use one row for small inputs. |
| Residual RMS and FP8 quantization | At 32144 by 5120, the two operations took 0.325 ms and the fused operation took 0.195 ms. | Keep residual and normalized BF16 rounding. Remove the read and write of the normalized array. |
| Small normalized projection | One-row gate/up diagnostic time was about 0.03431 ms with fusion and 0.03319 ms with the previous CUDA chain. | Use the previous CUDA chain for at most 32 rows. |
| Narrow FP8 traversal | At M 32144, QKV/Z changed from 3.451 to 3.307 ms and down changed from 3.563 to 3.437 ms. | At M at least 32144, use swizzle 16 for N/K 16384/5120 and swizzle 8 for N/K 5120/17408. |
| Strided GDN value copy | At 32144 rows and stride 10240, the median changed from 0.42002 to 0.12376 ms. | Copy eight adjacent BF16 values per lane, with 256 threads. |
| FP32 recurrent tile | Batch 2 changed from 3.80272 to 3.05744 microseconds. Batch 4 changed from 5.21184 to 4.54240 microseconds. | Batch 2 uses eight rows and one warp. Batch 4 uses two rows and four warps. |

The table uses diagnostic operator microbenchmarks and a different profiler replay.

The [FP8 cache-traversal diagnosis](../bench/evidence/2026-10-10-fp8-cache-traversal.json) has ten randomized rounds with thirty graph replays per sample.
At M 32144, QKV/Z L2 hits changed from 75.95% to 89.92%, and DRAM bytes changed from 6.345 to 2.829 GB.
Down L2 hits changed from 76.33% to 84.95%, and DRAM bytes changed from 6.439 to 3.826 GB.
The profiles kept 168 registers per thread, 202240 shared-memory bytes, and 12.50% occupancy.
Different profiler replays supply these counters.
Candidate outputs were equal at zero absolute and relative tolerances.

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

Paired time checks rejected previous K=256 candidates with K-major scales, smaller M tiles, and two-SM down projections.
They are excluded from dispatch. Keep unsuccessful attempts in external evidence storage.
N=256 tiles increased the large gate/up diagnostic time from about 6.72 to 19.39 ms.
Dispatch does not use these tiles.

GDN copy candidate measurements used ten rounds with random order and forty graph replays per sample.
Recurrent candidate measurements used ten rounds and one hundred operations in timed graphs.
Recurrent outputs and states stayed bitwise equal.
The copy profile changed DRAM throughput from 16.00% to 58.00%.
The selected copy used 18 registers per thread and 65.56% occupancy.

Long-scoreboard stalls were 68.14%, and L2 hits were 0.01%.
Different profile replays supply these counters.

Selected-output paths keep all required K/V and remove unused historical attention and MLP output.
Complete-operation comparison passed twenty-one selected cases with three rounds and twenty pairs per round.
The source was dirty, so these measurements are diagnostic.
The full 317-case regression stays active.
Reports record different loaded module identities for owned pointwise CUDA and FP8 GEMM.

Short-output diagnosis uses 32768 input tokens and sixteen outputs.
Ordinary prefill had a median of about 1.323 s after the selected FP8 traversal.
Its revised required-work bound was about 0.442 s.
The ratio was about 2.994, and the wall spread was about 1.07%.
The source stayed the same during this diagnostic.
The source-contract check passed.

MTP short-output prefill had a median of about 1.379 s.
Formal phase verdicts use the fixed 4096-output workloads.


The [dispatch diagnosis](../bench/evidence/2026-10-10-dispatch-diagnosis.json) keeps source identities, profile counters, and original hashes.


## Subsequent operator diagnosis

Residual RMS at 128 to 2047 rows uses streaming stores with the same 256-thread, four-value tile.
At 624 rows, diagnostic time changed from about 4.028 to 3.717 microseconds.

The different full-operation profile kept about 12.8 MB of DRAM traffic.
Its warp instruction count changed from 1412736 to 1178112. Register count changed from 48 to 44 against pinned TileLang.

GDN gates at 128 to 4095 rows use 128 threads. Small and large dispatch keep their previous tiles.

The small-row chain comparison used three rounds with twenty pairs and one hundred operation repetitions inside each timed graph. Each sample replays that graph once.
At 8, 16, and 32 rows, the previous CUDA chain saved about 0.672, 0.538, and 0.692 microseconds against fusion.

All three time-saved confidence lower bounds were positive. Full outputs were equal.

A two-SM gate/up candidate changed stage count and swizzle together. Its mean time saved was about 0.210 ms.
A subsequent diagnosis kept automatic seven-stage storage and changed only swizzle from 8 to 16.

It used three rounds with twenty pairs and ten graph repetitions per sample.
Mean time saved was about 0.104 ms, with one-sided 95% lower bound about 0.094 ms. Outputs were equal at zero absolute and relative tolerances.

The profiles kept 168 registers, 210432 shared-memory bytes, and 11.72% occupancy.
L2 hits changed from 81.78% to 88.73%. DRAM traffic changed from 8.161 to 5.507 GB.

The two profiles have all thirteen requested metrics. Profiler replay durations do not supply the timing verdict.

A full prefix-hit graph profile used 32768 input tokens, a 32144-token prefix, and sixteen outputs.
Across fourteen graph replays, the sum of FP8 projection kernel times is about 21.707 ms per replay.

The sum of attention kernel times is about 6.337 ms per replay.
These sums do not measure a critical path because branches can overlap.

They identify projection and attention work for subsequent tuning. Formal workloads keep 4096 outputs.

FP8-to-BF16 conversion with BF16 GEMM increased full projection time. Dispatch keeps FP8 checkpoint weights and FP32 scales.
The [phase progress record](../bench/evidence/2026-10-10-phase-progress.json) keeps the failed full-output phase and operator measurements.


## Large gate/up scale and stage selection

The current two-SM gate/up path uses K256 tiles and three pipeline stages at M at least 32144, N 34816, and K 5120.
The previous K128 path used five stages.
Swizzle stays 16.

A stage comparison kept this swizzle and changed seven automatic stages to five explicit stages.
Three rounds used twenty pairs and ten graph repetitions per sample.
Mean time saved was about 0.120 ms. The one-sided 95% lower bound was about 0.114 ms.
Outputs stayed bitwise equal.

Column-major scales decrease scale-transfer overhead for this large shape. Other projections keep K-major scales.
Checkpoint weights and public FP32 scales keep their values and layout.
Each call packs 43520 bytes of weight scales, so graph replay reads subsequent scale changes.
The fused quantization kernel writes into physical storage with row capacity rounded up to a multiple of four.

Each call clears at most three tail rows and returns only the logical output rows.
Padding, packing, and tail clearing do not increase the theoretical bound.

A full-operation candidate comparison included residual RMS, quantization, scale packing, padding, and GEMM.
Three rounds used twenty pairs and one hundred operation repetitions inside each timed graph. Each sample replays that graph once.

At M 32144, mean time saved was about 0.187 ms, with a one-sided 95% lower bound of about 0.185 ms.
At M 32290, the values were about 0.046 ms and 0.044 ms.
The candidate outputs and residual sums stayed bitwise equal.
The candidate used a copy into padded storage. The production fused path writes directly into the aligned storage.

Thirty GPU checks passed. These include changed input and scales, side-stream graph replay, and rejected scale layouts and pitches.
Four affected full-operation cases passed numerical and speed checks with the full operator timing protocol.
These dirty-source diagnostics do not supply the full current-source 317-case or fifteen-row verdict.

The previous K128 GEMM profiling run used 168 registers and 160256 dynamic shared-memory bytes, with 11.72% occupancy.
Its L2 hit rate is 88.60%, and DRAM traffic is about 5.489 GB.
The previous automatic seven-stage, swizzle-16 profile used 210432 dynamic shared-memory bytes and the same occupancy.

These different replays identify resource use. The timing comparisons supply the timing verdict.
The TileFoundry report uses a representative HIR shape. It does not show the full native FP8 GEMM shape or layout.
The [scale and stage diagnosis](../bench/evidence/2026-10-10-fp8-scale-stage-diagnosis.json) keeps each launch's thirteen counters, estimates, and source identities.

## Projection CTA clusters at 624 to 2496 rows

The five shape-selected one-SM projections use a two-block CTA cluster at 624 to 2496 rows.
[Architecture](architecture.md#target-computation-and-attention) specifies their N/K pairs.
TMA multicast supplies a shared activation tile to the two output-column blocks.
The previous arithmetic, K-major scales, automatic stages, PDL, and caller stream stay the same.

The [cluster diagnosis](../bench/evidence/2026-10-10-fp8-cluster-diagnosis.json) compares thirty-five full operations and ten GEMM primitives.
Full operations include RMS or SiLU, quantization, allocation, and GEMM.
The measurements use three rounds, twenty pairs per round, and one hundred operation repetitions inside each timed graph. Each sample replays that graph once.
All full-operation outputs were bitwise equal. The rows include 625, 1023, 2047, and 2048, across the swizzle boundary.

For gate/up, mean time decreased by about 8.544, 20.496, and 32.252 microseconds at 624, 1248, and 2496 rows.
The one-sided 95% lower bounds were about 7.988, 19.321, and 30.663 microseconds.

Fifteen affected static cases passed numerical and speed checks with the full per-case protocol.
Eighteen GPU checks passed, with changed graph inputs, scales, PDL settings, and row boundaries.
A subsequent eight-test collection also checked that activation scales stay the same.

A different R624 profiling run shows 168 registers, 201216 dynamic shared-memory bytes, and 11.83% occupancy for the owned GEMM.
Its L2 hit rate is 72.90%, with 215003648 DRAM bytes and 36.52% long-scoreboard stalls.
The record keeps thirteen counters for each launch of the CUDA and pinned TileLang operations.
The TileFoundry HIR uses R2 and N256. It does not show the full native GEMM resource layout.

Sixteen-output prefix-hit diagnosis has prefill medians of 36.778 and 122.464 ms for batches 1 and 4.
The bound ratios are about 3.657 and 3.044. Spreads are about 6.537% and 1.206%.
These diagnostics do not supply the fifteen-row, 4096-output verdict.
Prototype build records do not contain the library hash at measurement time.

Hybrid scale layouts and N64 tiles did not pass the primitive speed checks. Those experiments did not include scale packing.
The hybrid candidate preallocated output. Only the previous operation included output allocation.

Compact-KV attention decreased time by small amounts at batches 1 and 2, and increased time at batch 4. Native paged attention stays selected.
TRT increased full gate/up time at 624, 1248, and 2496 rows. The FP16 softmax candidate has an SM107 guard and failed that guard on B200.

## Gated RMS and FP8 output projection

The GDN output path fuses gated RMS and activation quantization, then uses its previous GEMM.
The BF16 rounding point stays before FP8 conversion. Packed inputs and checkpoint scales keep their values.
This removes one intermediate BF16 array and one launch.

The [gated projection diagnosis](../bench/evidence/2026-10-10-gated-projection-diagnosis.json) keeps twenty-four full-operation candidate rows and sixteen affected formal cases.
Candidate timing uses three rounds, twenty pairs per round, and one hundred operations inside each timed graph.
Each sample replays that graph once. All candidates kept FP8 bytes, scales, and outputs bitwise equal to the previous CUDA chain.

At 32144 rows, four head rows per warp decreased mean full-operation time by about 0.153 ms.
The one-sided 95% lower bound was about 0.151 ms.
At 624, 1248, and 2496 rows, two head rows per warp decreased mean time by about 4.816, 8.666, and 18.187 microseconds.
Their lower bounds were about 4.806, 8.578, and 17.911 microseconds.

Column-major scales use one head row per warp. Four head rows per warp increased time at 1, 8, and 32 token rows.

All sixteen affected cases passed numerical, dispatch, and speed checks with the full per-case protocol.
The matrix keeps the previous 301 cases and adds sixteen gated output cases, for 317 cases.
Twenty GPU checks passed, with packed views, changed graph inputs, bitwise output equality with the previous CUDA chain, and rejected storage.
One subsequent GPU twin check passed at the existing TileLang tolerances.

The different Nsight replay used 32 registers, zero dynamic shared storage, and 94.19% occupancy for fused gated quantization.
It had 979841280 DRAM bytes and 23.63% L2 hits.
The GEMM used 168 registers, 201216 dynamic shared-memory bytes, 12.51% occupancy, and 546286080 DRAM bytes.
Its L2 hit rate was 95.20%. Each launch has all thirteen requested counters.

TileFoundry estimates use R2, N256, 48 heads, and K6144. They do not show native GEMM resource layout.

HIR twins supply adapters for all HIR functions. The GDN recurrence example takes normalized Q/K and keeps FP32 state.
Full GDN timing keeps its existing frozen chunked reference. Prototype records do not contain the library hash at measurement time.

Sixteen-output ordinary prefill ratios were about 2.897 and 2.984 at batches 1 and 4.
Wall spreads were about 0.294% and 0.695%.

Prefix-hit ratios were about 3.699 and 3.027, with spreads about 12.598% and 1.070%.
The batch-1 prefix group failed the two diagnostic gates. Keep that full group for investigation.
All four groups used two full warmups and five measurements. Full 4096-output phase acceptance stays pending.

## K tiles and local-memory traffic

The [K-tile and scheduling diagnosis](../bench/evidence/2026-10-10-fp8-k-tile-scheduling-diagnosis.json) keeps candidate timing, source identities, counters, and original hashes.

The large MN gate/up path uses K256 tiles and three stages. Its scale groups keep 128 elements.
Each tile computes, scales, and adds two groups in the previous order.
The K-major layout reference keeps K128 tiles and five stages.
Other projections keep their previous configuration.

The measurements compare full operations with residual RMS, FP8 quantization, scale packing, padding, and GEMM.
Three rounds use twenty pairs and one hundred operations inside each timed graph.
Each sample replays the graph once. Full outputs and residual sums stay bitwise equal.

At 32144 and 32290 rows, mean time decreased by about 0.487 and 0.487 ms.
The one-sided 95% lower bounds are about 0.480 and 0.478 ms.
Two-stage candidates increased time at each shape. Three stages passed each round at each shape.

The next counter values come from the prototype replay. That replay has no baseline loaded-library hash at measurement time.
A different CUDA and pinned TileLang replay records the loaded CUDA libraries.

A different Nsight replay keeps fifteen counters for each launch, with local-memory read and write sectors.
Each GEMM uses 168 registers and 11.72% occupancy.
Dynamic shared storage increases from 160256 to 185344 bytes.
Local reads decrease from 39207168 to 1096704 sectors. Local writes decrease from 4983712 to 14208 sectors.

DRAM traffic changes from 5501504256 to 5492587776 bytes.
L2 hit rate changes from 88.57% to 88.06%.
These observations identify local-memory traffic changes. Paired full-operation measurements supply the timing result.
The canonical model keeps the same required rows, scale groups, and parameter ranges.

A large-shape TileFoundry report uses 32144 rows, width 5120, and 34816 output columns.
Its logical FP32 dequantization model has 11459845160960 matmul operations.
The report counts logical intermediate arrays. It does not model native FP8 group accumulation, cache reuse, or CUDA placement.
Its roofline estimates do not supply framework acceptance bounds.

Smaller M64 tiles decreased registers and shared storage but kept about 11.8% observed occupancy and increased time.
N256, different two-SM tiles, K-split reduction, and sequential or parallel KV splits also increased time.
MN scale and stage candidates at 624 to 2496 rows keep their primitive scope.
Scale packing and quantization are not in those measurements.

## Aligned prefill and mixed requests

After a successful prefill chunk of at least one block, a later rounded one-token prefill can wait for the next step.
The checks compare the available budget with the checkpoint distance and `prefill_end - start`.
The budget must be less than each distance.
The checks run before cache allocation and keep queue order.

Decode-only steps and speculative verification keep the positive-input fallback.
Multi-step tests include new and continued prefill with ordinary decode and four or seven drafts.
Reachable checkpoints and final tails keep progress.
Configured token budgets less than one block also keep progress.
The production token budget stays 32768 and block size stays 784.

## TMEM and output-stage tuning

Use observed kernel counters to distinguish tensor work, scale access, accumulator waits, and output conversion.
The observability command requests 21 metrics per launch.
It includes Tensor activity, issue activity, eligible warps, DRAM/L2 bytes, TMA transfers, resources, local-memory sectors, and stalls.
It keeps worker output and load identities from stdout and stderr.
Each counter launch must match the recorded worker process, case, and backend.

The necessary CUDA modules must match source, templates, project headers, build inputs, and loaded library bytes.
Only the three FP8 projection operations at more than 32 rows must load the owned GEMM module.

TMA bytes specify transfers through the TMA path. L2 can supply those transfers.
They do not equal compulsory HBM traffic.

SourceCounters show sampled warp states at source instructions. Their proportions do not measure instruction wall-time fractions.
Warp-specialized kernels can change register budgets by role. Examine SASS and local-memory traffic with launch resource counts.

A source-C9 prefix graph contained 1196 kernels, with 252 FP8 projection launches.
Their elapsed-time sum was about 19.470 ms. Attention's sum was about 5.767 ms.
These sums identify work for tuning. Overlapping branches prevent their direct use as a critical path.

SourceCounters showed about 87% of reported excessive regular-load sectors at four activation-scale load instructions.
That fraction excludes TMA traffic. It cannot show the fraction of HBM bandwidth used by scales.

The current five projections at 624 through 2496 rows use an owned accumulator derived from the pinned CUTLASS mainloop.
Two independent output tiles issue TMEM loads before one load wait.
Each output keeps ascending K-group order, FP32 scale multiplication, and FP32 accumulation.
The final wait is before accumulator-slot release.

The epilogue directly rounds FP32 values to BF16, with the same round-to-nearest-even result.

External candidate diagnostics kept full outputs bitwise equal.
The production path passed 26 boundary, replay, changed-weight, PDL, stream, and storage checks.
All 18 affected full-operation cases passed against pinned TileLang, with three rounds and twenty pairs per round.

Each graph has 100 full operations. The smallest time-saved confidence lower bound was about 6.486 microseconds.
These dirty-source filtered records have diagnostic scope.
The [accumulator diagnosis](../bench/evidence/2026-10-10-fp8-accumulator-observations.json) keeps selected cases, original hashes, counters, and review scope.

The native gated projection observation uses 624 rows, output width 5120, and input width 6144.
Its representative HIR uses two rows and output width 256. Keep these shapes and purposes different.
The full native chain includes gated normalization, FP8 quantization, allocation, and GEMM.
The profiler has three TileLang launches and two CUDA launches, with all 21 counters per launch.
Profiler timing does not supply the interleaved speed verdict.

A vector activation-scale prefetch kept outputs equal but increased full-operation latency. Dispatch excludes it.
An output-tile change failed numerical checks. Static scheduling and different input layouts also did not decrease full-operation time.
Their compile failures, timeouts, and source-stability failures stay in external records.
The unstable external accumulator-equivalence attempt supplies no performance conclusion.

Four TMEM loads before one wait and MN scales for the two operands kept outputs bitwise equal.
Each candidate increased time in all 18 full-operation cases against the current production path.
Dispatch keeps two loads per wait and K-major scales for these shapes.
N64 compilation failed because the checkpoint scale granularity is 128 output columns.

The last production short-output diagnosis used input 32768, prefix 32144, and output 16.
With two warmups and five measurements, prefill medians were about 0.037355 and 0.063120 seconds at batches 1 and 2.
Their required-work bounds were about 0.010057 and 0.020114 seconds.
More tuning is necessary. Formal phase workloads keep 4096 outputs.
