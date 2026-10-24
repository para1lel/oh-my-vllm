# HTTP performance measurement

Use one NVIDIA B200 and the [model contract](model.md).
Use the full Rust service and Python inference worker.
Placeholder outputs do not supply inference evidence.
The [GPU execution rules](../../execution.md) control execution.

## Configurations

Each batch starts with concurrent HTTP requests.
The input and output lengths apply to each request.

| Mode | Batch sizes | Input tokens per request | Output tokens per request |
|---|---|---|---|
| Ordinary decode | 1, 2, 4 | 32768 | 64, 8192 |
| MTP4 | 1, 2, 4 | 32768, 65536, 131072 | 64, 8192 |

These are 24 configurations.
The 64-token output makes the measurement sensitive to prefill time.
Keep the 8192-token output for long decode measurement.
Use a token budget of 32768 and a maximum of 4 running requests.
Use requests without a cached prefix.

Use workload `serving-tools-v1` with checkpoint and service sampling defaults.
The current defaults are temperature 1.0, top-k 20, top-p 0.95, presence penalty 0.5, and the repetition guard.
Use the client ordinal as the random seed. A fixed seed does not select greedy sampling.

Keep reasoning off.
Set `tool_choice=auto` with the [benchmark tool schema](../../../development/serving/workload.py).
Ordinary text is permitted. The grammar checks tool names and JSON arguments.
Target and proposal sampling use the same grammar and penalty rules as serving.

Load the model before timing.
Complete initialization before timing.
Construct the prompt text before timing.
Start timing before the client creates its request threads.
Stop timing after the client reads all responses as JSON and its request threads stop.
Include submission, request preparation, queue time, prefill, decode, constraints, sampling, process communication, and output transport.

Exclude model loading, initialization, and trace analysis from timing.
Do not use time to first token as the full latency.

## Measurement command

Set `MODEL_ROOT` to the checkpoint directory.
Select one GPU UUID with `OH_MY_VLLM_GPU_UUID`.
Set `ARTIFACT_ROOT` to a new directory external to the repository.
Build the native modules and release executable with the [service setup](../../serving.md#start).
Run the full comparison:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.benchmark --model "$MODEL_ROOT" --artifacts "$ARTIFACT_ROOT"
```

The command starts and stops the service for each mode.
It completes short initialization requests before timing.
It also checks a complete tool call in each mode before timing.
It checks prompt lengths, output lengths, and uncached prefixes in HTTP usage.

The worker uses fixed FA defaults and the startup checks in the [service memory budget](cache.md#cache-memory).
The default GDN capacity is 128 slots.
Use `--blocks` for a smaller explicit FA capacity and `--mamba-blocks` to change the GDN capacity.
The trace identity records the resolved capacities and memory budget.

Use `--case ordinary:1:32768` to select its two output lengths.
Use `--case ordinary:1:32768:64` to select one configuration.
Use `--output-tokens 64` for measurements sensitive to prefill time.
Use `--output-tokens 32` for a short run.

The command accepts configuration subsets and other output lengths.
It rejects duplicate configurations.
Request failures or incorrect artifacts cause an error exit.

The worker writes an execution trace when `OH_MY_VLLM_EXECUTION_TRACE` is set.
The trace path must be external to the repository and must not exist before the run.
The benchmark sets this variable for its service process.

## Nominal resources

Use the dense HGX B200 values from [NVIDIA's data sheet, page 8](https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf).
The rates are HBM 7.7 TB/s, FP8 4.5 PFLOP/s, BF16 2.25 PFLOP/s, and TF32 1.1 PFLOP/s.
Scalar rates are FP32 75 TFLOP/s and FP64 37 TFLOP/s.
Do not substitute sparse rates, GB200 rates, or measured calibration.

```text
group_seconds = max(
    group_bytes / HBM_rate,
    group_FP8 / FP8_rate + group_BF16 / BF16_rate + group_TF32 / TF32_rate,
    group_FP32 / FP32_rate,
    group_FP64 / FP64_rate,
)
reference_seconds = sum(group_seconds for each serial group)
efficiency = reference_seconds / HTTP_seconds
```

This is a conservative resource estimate, not a minimum physical latency proof.

## Lower-bound calculation

The [reference calculation](../../../development/serving/kernels.py) uses version `http-reference-v4`.
It uses the current kernel boundaries and precision rules. It has no measured calibration input.

The [trace validator](../../../development/serving/trace.py) first checks full prefill, continuous accepted positions, proposal dependencies, and output counts.
The single trace identity must match the service mode.
Prompt lengths must match HTTP usage.
The benchmark manifest maps client ordinals to distinct response and trace IDs, mode, and input/output lengths.
Missing IDs, duplicate IDs, unknown IDs, and inconsistent lengths prevent accounting.

The reference scheduler admits requests by client ordinal, with at most 4 active requests.
It schedules decode first, then shares the unused part of the 32768-token budget between prefill requests.
Integer remainders go to earlier ordinals.
Partial prefill stops at a 784-token boundary.
Unused space goes to earlier pending requests when another aligned chunk fits.
A final prompt chunk can end between boundaries.

Physical arrival order and call boundaries do not control this schedule.
MTP verification consumes the previous proposals and keeps its recorded acceptance.
Context computation stops at the accepted endpoint, with one additional row when new proposals follow.
The scheduler cannot create proposals or combine different steps of one request.

The calculation uses the maximum resource time for each kernel group.
It then adds the times across serial groups.
Prefill compute time and decode memory time are added.
A fused kernel is one group.
PDL producers and consumers use one group when their necessary work can occur at the same time before the consumer wait.

FFN rows 65 to 1024 use one group for normalization, fused upper computation, and down projection.
Each consumer can read constant weights and weight scales before its wait.
Activation values and activation scales are read after those waits.
Other prefill row counts keep their previous groups.

The down16dp projection for at most eight rows reads weights after that wait.
Its credited work stays in a different group.
Layer completion boundaries stay serial.

The [group definitions](../../../development/serving/kernels.py) use different groups for FA projection, heads, attention, gate, and output work.
They also use different groups for GDN convolution, recurrence stages, and projections.
FFN stages, MTP input stages, and vocabulary projection have different groups.
Count each necessary object's memory bytes in the first group that uses that object.
The groups use the fixed canonical schedule. Trace launch counts do not change these groups.

Extra tiles, padding, launches, repeated computation, or temporary copies cannot increase credited work.

Artifacts contain responses, the manifest and its digest, the selected trace digest, resource totals, group times, and acceptance counters.
The manifest also records sampling defaults, the tool schema, seeds, request hashes, and tool validation counts.
Tool arguments must match the schema. A final call stopped at the output limit can contain a valid argument prefix.

The manifest binds the full request timer to the artifact identity.
That identity binds source, executable, native modules, checkpoint, tokenizer, device, and accounting version.
The [artifact check](../../../development/serving/benchmark.py) rejects changes during the run.
Measurement artifacts apply only to their recorded source and accounting version.

## Memory

Count matrix weights once per logical projection pass, shared across its batch.
FP8 matrices use one byte per element and FP32 scales for each 128-by-128 region.
BF16 matrices use two bytes per element.
Count each normalization vector once per pass.
Intermediate hidden states, logits, and activation buffers have no memory credit.

FA reads prior KV history once per request and pass.
New query KV can pass directly to attention without another HBM read.
Count accepted KV writes once, with 4 KV heads, width 256, and BF16 K and V.
MTP draft attention starts at position 1. The first draft reads its cached context endpoint.
Subsequent drafts compute new KV.

GDN reads one FP32 recurrent state and one BF16 convolution state when a request starts above position zero.
Count unique 784-token checkpoints and the accepted endpoint in that pass.
A checkpoint at the endpoint counts once.
Verification advances by the accepted proposal count plus one.
Each GDN layer has 48 recurrent heads of 128-by-128 elements and 10240 convolution channels with 3 stored values.

Private candidate writes and subsequent state copies have no credit.
This omission does not remove the private-state correctness checks.

## Arithmetic

Matrix products count two operations per multiply-add at their specified operand precision.
The calculation covers all 64 target layers and the full vocabulary projection.
It counts all verification rows, also when some proposals are rejected.
FA counts the full causal attention pairs and three FP32 operations per pair and query head.
These scalar operations represent score scaling, probability summation, and probability normalization.

GDN prefill with more than 128 queries per request uses 64-row chunks, split at 784-token checkpoints.
Preparation, state carry, and output use three serial groups.
Their products use BF16. Diagonal inversion and state updates use FP32.
The W/U triangular products include the diagonal after beta multiplication enters the inverse matrix.

Short prefill, decode, and verification use FP32 recurrence.
Small BA projections use FP32 arithmetic on BF16 inputs when fused into convolution or recurrence.
Other BA projections use BF16 matrix products.
Use a new accounting version when these fixed dispatch and fusion rules change.

RMS normalization counts four operations per element and two scalar operations per row.
Residual addition counts one operation per element.
The FFN activation counts two products per intermediate element.
GDN convolution counts seven operations per token and channel.

FA head normalization uses width 256. Rotary products use the specified 64 dimensions.
The attention output gate counts one product per element.
MTP context, draft, and vocabulary projections follow their recorded causal positions.

## Omissions

Sampling, penalties, grammar processing, sorting, integer indexing, random generation, and MTP acceptance have no resource credit.
Their runtime cost stays inside the HTTP timer.
The estimate does not include sigmoid evaluation, transcendental instructions, datatype conversion, quantization, and scale application arithmetic.
It also does not include GDN head normalization, gates, input normalization, and prefix/ratio construction beyond the listed terms.
There is no empirical throughput factor for operations that have no credit.

The trace does not include all input token IDs for embedding-region deduplication.
Embedding memory reads therefore have no credit.
Checkpoint hidden storage, MTP endpoint storage, metadata, and returned token traffic also have no credit.
These conservative exclusions can reduce reported efficiency.
They cannot increase it when an implementation adds work.

Vision and initialization do not occur inside these text-only HTTP timers.
These timers cannot check their correctness.
Changes to these assumptions must change the accounting version and invalidate earlier performance evidence.
