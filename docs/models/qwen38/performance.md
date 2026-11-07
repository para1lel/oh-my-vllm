# HTTP performance measurement

Use one NVIDIA B200 and the [model contract](model.md).
Use the full Rust service and Python inference worker.
The [GPU execution rules](../../execution.md) control execution.

## Configurations

Each batch starts with concurrent HTTP requests.
The input and output lengths apply to each request.

| Mode | Batch sizes | Input tokens per request | Output tokens per request |
|---|---|---|---|
| Ordinary decode | 1, 2, 4 | 32768 | 64, 8192 |
| MTP4 | 1, 2, 4 | 32768, 65536, 131072 | 64, 8192 |

These are 24 configurations.
The new cache layout must complete the full matrix and the approved AgentX workload.
Each of the 12 configurations with 8192-token outputs must have efficiency at least 40%.
No new-layout HTTP or AgentX acceptance is complete.

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
scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.benchmark --model "$MODEL_ROOT" --artifacts "$ARTIFACT_ROOT" --reference-version qwen-fp8-http-reference-v1
```

The [CLI default](../../../development/serving/benchmark_options.py#L69)<!-- source: choices=(VERSION, FP8_VERSION), default=VERSION --> is still `http-reference-v5`.
The [case dispatcher](../../../development/serving/benchmark_case.py#L81)<!-- source: def case_bound(service, tokenizer, events, manifest, responses, rust_previous): --> returns `blocked` for the new layout with that version.

The command starts and stops the service for each mode.
It completes short initialization requests before timing.
It also checks a complete tool call in each mode before timing.
It checks prompt lengths, output lengths, and uncached prefixes in HTTP usage.

The worker resolves capacity through the [cache byte budget](cache.md#cache-memory).
Usable page and checkpoint counts must satisfy `P = 64 * C`.
The current plan does not prove measured GPU capacity.
The trace identity records the resolved capacities and memory budget.

Use `--case ordinary:1:32768` to select its two output lengths.
Use `--case ordinary:1:32768:64` to select one configuration.
Use `--output-tokens 64` for measurements sensitive to prefill time.
Use `--output-tokens 32` for a short diagnostic run.
The new reference returns `blocked` for lengths other than 64 or 8192.

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

Efficiency is reference seconds divided by full HTTP seconds.
This is a conservative resource estimate, not a minimum physical latency proof.

## New-layout HTTP accounting

The full cold-HTTP calculator, [`lower_bound`](../../../development/serving/qwen_fp8_roofline.py#L292)<!-- source: def lower_bound(events, manifest, artifacts, responses, *, tokenizer=None): -->, covers all 24 configurations.
Its [accounting identity](../../../development/serving/qwen_fp8_evidence.py#L11)<!-- source: VERSION = "qwen-fp8-http-reference-v1" --> is `qwen-fp8-http-reference-v1`.

The [evidence validator](../../../development/serving/qwen_fp8_trace.py#L492)<!-- source: def validate(events, manifest, artifacts, responses, tokenizer=None): --> binds loaded identities, Rust-validated execution, parser results, sampling, response tokens, and HTTP usage.
Missing or inconsistent evidence returns `blocked` without efficiency.
This calculator rejects prefix reuse and AgentX tree accounting.

The [canonical schedule](../../../development/serving/qwen_fp8_schedule.py#L122)<!-- source: def canonical_passes(states, mode): --> and [resource calculation](../../../development/serving/qwen_fp8_math.py#L225)<!-- source: def resource_reference(passes, component_function, device): --> define the new layout's credited work.
Only necessary accepted work receives credit.
Resident device data has a conservative resource allowance.
Logical KV traffic alone earns no HBM credit.
Excluded operations still contribute to full HTTP time.

The partial [`resource_floor`](../../../development/serving/qwen_fp8_roofline.py#L211)<!-- source: def resource_floor(workload): --> keeps `qwen-fp8-components-v1`.
It is not the full HTTP calculator.
Component estimates and old v5 results cannot prove the new 40% target.

## Offline HTTP acceptance

Stop the services.
Set `TOKENIZER_ROOT` to recorded tokenizer assets.
Set `ACCEPTANCE_REPORT` to an external file.
Run the [offline check](../../../development/serving/qwen_fp8_acceptance.py#L437)<!-- source: def main(argv=None): -->:

```bash
scripts/with-env.sh python -m development.serving.qwen_fp8_acceptance "$ARTIFACT_ROOT" --tokenizer "$TOKENIZER_ROOT" --output "$ACCEPTANCE_REPORT"
```

All 24 configuration keys occur one time per run.
All 12 long-output rows must reach at least 40% efficiency.
Exactly 40% passes.
The check compares raw ranges, manifests, HTTP responses, and full reference results.

Rust complete records contain prepared responses and do not prove client receipt.
The checker compares prepared responses with archived responses from successful HTTP calls.
Requests must use nonstreamed HTTP.

Only `status=passed` with `target_passed=true` and scope `qwen-fp8-http-24-long-12` meets this HTTP target.
Exit codes are 0 for `passed`, 1 for `failed`, and 2 for `invalid`.
New runs must match the [calculator binding](../../../development/serving/qwen_fp8_identity.py#L36)<!-- source: def calculator_identity(): --> at `identity.json` field `accounting.calculator`.
The checker adds no bindings to old runs.
This check does not prove AgentX acceptance, DeepSeek acceptance, numerical precision, or GPU capacity.

## Lower-bound calculation

The historical [reference calculation](../../../development/serving/kernels.py) uses version `http-reference-v5` for the [recorded matrix](results.md).
The following calculation, memory, arithmetic, and omission sections specify that version's older BF16 layout and 784-token checkpoints.
They do not define accounting for the new FP8 layout.
The calculation has no measured calibration input.

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
Subsequent drafts compute new KV. Their temporary KV writes have no credit.

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

A chunk that starts at token position zero has zero initial state.
Its W/state and Q/state products have no credit.

Short prefill, decode, and verification use FP32 recurrence.
Small BA projections use FP32 arithmetic on BF16 inputs when fused into convolution or recurrence.
Other BA projections use BF16 matrix products.
Use a new accounting version when these fixed dispatch and fusion rules change.

RMS normalization counts four operations per element and two scalar operations per row.
Residual addition counts one operation per element.
MTP counts target final normalization for all context rows, also for rows without vocabulary projection.
Rows shared with vocabulary projection count once.

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
