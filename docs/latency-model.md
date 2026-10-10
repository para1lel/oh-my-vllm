# Theoretical phase latency

The [performance requirements](requirements.md#req-perf-001-phase-latency) use the work of each measured request.
The model reads semantic execution records. CUDA Event durations supply diagnostic data only.

## Request boundaries

Prefill starts at submission of a request with token IDs and ends at its first kept output token.
Decode starts at that token and ends at its last kept output token.
The wall intervals include Rust scheduling, Python execution, ZMQ, and sampling.
Model loading, tokenization, HTTP, and warmup do not contribute to these intervals.

Each repetition uses the longest request prefill interval and the longest request decode interval.
The two maxima can come from different requests.
Batch duration and cleanup time have different fields.

The worker records completed work at existing token readbacks.
An endpoint excludes work after its last readback.
The analysis validates request IDs, output counts, execution steps, mode, and state precision against Rust results.

## Semantic work

`performance/model.py` specifies target layers, MTP, DSpark, vocabulary projection, greedy selection, and persistent state.
Each projection uses its effective rows and inner/output dimensions.
Matrix multiplication counts `2*M*N*K` arithmetic operations.
FP8 group scales, attention KV lengths, GDN chunks, and conditional draft paths have different counts.

The model includes the full seven-row DSpark backbone when that backbone executes.
It includes rejected verification rows and necessary draft computation.
Padding and duplicate implementation work do not increase the bound.

Parameter metadata specifies loaded tensors, with converted scales.
The model counts each required parameter buffer and the selected embedding rows.
It does not use checkpoint disk size as GPU traffic.

### Required output rows

Previous rows supply K/V for future attention.
Only selected rows supply final hidden features and logits.
Selection keeps all verification samples, with rejected candidates.
Native MTP also keeps target rows used by its context input and registered boundary features.

At the target's last FA layer, all required rows perform input normalization and K/V projection.
Selected rows perform Q/gate projection, attention, output projection, MLP, and final normalization.
DSpark feature taps precede this layer.
A configured feature tap at layer 63 keeps the full layer output.

Persistent MTP normalizes target-hidden and embedding inputs, then applies FC to each required context row.
Each such row supplies K/V. Only selected endpoints supply draft hidden output.
Subsequent context rows receive target hidden, so discarded previous draft hidden has no consumer.
Temporary proposal rows keep their full output.

For `R` context rows and `S` output rows, QKV Tensor work is `2*5120*(2048*R+12288*S)`.
FP8 input quantization counts once for `R` rows.
A second implementation quantization for selected rows is overhead.
Q/gate and K/V parameter ranges have different, disjoint read identities.

Terminal requests keep registered full prefix pages.
The model removes draft context and hidden work with no subsequent consumer.
Restored MTP boundary features have a named 10 KiB read per physical page.
Features produced previously in the same prefill phase have free ideal retention.
MTP incoming KV reads use physical positions `[1,end)`, with shared ranges merged by page.

The equation version is `qwen38-dspark-b200-semantic-v4`.
Typed FX metadata identifies shape queries and host-known proposal page addresses.
These metadata operations have zero required GPU work in the relaxed implementation.
Their execution time stays in wall time.

## Dependencies and shared resources

Semantic nodes keep activation, token, and state dependencies.
Independent parameter reads can start early.
CUDA stream order does not add a semantic dependency.

Whole-matrix completion barriers would prevent ideal streaming in an operation.
The current path analysis thus relaxes all arithmetic service to zero, with GEMM and GDN.
Each incoming read node keeps dependencies and receives its own complete optimistic storage credit.
Complete arithmetic work still contributes to the shared-resource bound.
The phase uses the maximum step path, so parameter prefetch can overlap work across steps.
This path is a relaxation, not the latency of complete atomic matrix operations.

The phase bound is the maximum of the relaxed critical path, mandatory HBM time, and shared execution-resource time.
This maximum lets branches with no dependency overlap while they share one GPU.
FP8 and BF16 Tensor Core times add because these operations use the same execution resource.
Different execution resources can overlap.

The model removes submission or phase-start offsets when requests enter one interval at different times.
This adjustment gives a bound for the longest individual request interval.

Decode also has a bound from its common tail.
Let `T` be the maximum first-token timestamp and `E` the maximum kept-token endpoint.
Rust records those first-token timestamps before it schedules the next worker step.
The common tail keeps only steps after the maximum first-token step.
Each request still stops at its own validated endpoint.
This work is in `[T,E]`, and `E-T` is at most the longest request decode interval.

The tail uses the same parameter-range unions, feedback intervals, and hardware peaks.
It starts a new final-write ledger and keeps the full initial cache credit.
Subsequent allocation changes can invalidate its writes.
Its resource bound has no phase-start offset adjustment.
The decode bound is the maximum of the offset-adjusted bound and the common-tail bound.
The two bounds do not add.

## Hardware limits

The profile uses the official dense peaks for one 1000 W B200.

| Resource | Peak |
|---|---|
| FP8 Tensor Core | 4.5 PFLOP/s |
| BF16 Tensor Core | 2.25 PFLOP/s |
| FP32 | 75 TFLOP/s |
| FP64 | 37 TFLOP/s |
| HBM | 7.7 TB/s |

See Table 3 in the [NVIDIA Blackwell technical brief](https://resources.nvidia.com/en-us-blackwell-architecture).
The table rounds dense BF16 to 2.2 PFLOP/s.
The model uses the optimistic value `4.5/2 = 2.25` from its sparse peak.
The [CUDA instruction throughput table](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions) supplies published integer, conversion, shuffle, and SFU limits.

SM count and maximum SM clock come from the selected GPU.
The reviewed full B200 profile has 148 SMs, 132644864 L2 bytes, and a maximum SM clock of 1965000000 Hz.
The collector reads the hardware clock ceiling separately. It must match the worker record.
Other SM, cache, or clock profiles fail before cost analysis.
Each repetition must also have a bound at most its observed phase time, with 1 microsecond timestamp allowance.

Measured kernel throughput and efficiency factors do not change these peaks.

The model uses ideal L2, registers, unified L1/shared storage, TMEM, and constant cache.
Each SM contributes `(3*256+8)*1024` bytes to this optimistic storage credit.
The terms are registers, unified storage, TMEM, and constant cache: 256 KiB, 256 KiB, 256 KiB, and 8 KiB.
See [SM100 storage](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-programming-guide/index.html#compute-capability-10-0) and [TMEM](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory).

The register and compute-capability checks validate the selected profile.
The model lets this storage keep values across operation boundaries.
Unpublished on-chip bandwidth has an infinite limit in the model.

Packed BF16/FP8 arithmetic and rounding with unresolved execution-resource mapping have zero service time.
The report lists these recognized relaxed operations.
An unknown operation blocks acceptance.

## Mandatory HBM traffic

Read accounting uses unique incoming tensor ranges and one storage credit per feedback interval.
The prefill interval uses parameter reuse across its chunks.
Each decode feedback interval gives one new ideal storage credit.
Required parameters that exceed this credit must come from HBM.

`performance/memory.py` tracks physical KV pages, recurrent states, and persistent draft data.
It counts final live writes after accepted progress, checkpoint registration, and terminal release.
Rejected snapshots and overwritten versions do not become mandatory final writes.
Page reuse and writes by other requests invalidate old physical versions.

The read and final-write calculations have separate credits.
This gives ideal initial and final cache contents.
Temporary activations can stay on chip through ideal fusion.
The model thus counts mandatory traffic rather than observed kernel traffic.

## Evidence and verdict

`benchmarks/framework.py` collects the fifteen fixed workloads with two complete warmups and five measurements.
Each measurement has its own trace-derived prefill and decode bounds.
Except for prefix-hit prefill, each phase must satisfy `median(wall) <= 3*median(bound)`.
Record the prefix-hit prefill ratio and check result without a latency gate.
Each phase must satisfy `(max(wall)-min(wall))/median(wall) <= 0.10`.
TPS, acceptance rate, GPU time, and cleanup stay diagnostic.

The collector checks unchanged committed source, the binary Rust build stamp, checkpoints, loaded CUDA modules, and steady-state cache trees.
The reviewed operation contract records project sources, provider sources, template headers, build flags, and package versions.
The assessment records the model version and equation-source hashes.
Original failed attempts stay in external storage.
Portable summaries identify their original hashes and source scope.

Use [profiling](profiling.md) for operator diagnosis and [testing](testing.md) for commands.
