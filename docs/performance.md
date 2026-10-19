# End-to-end requirements

Use one NVIDIA B200 and the [model contract](operators/model.md).
Use the full Rust service and Python inference worker.
Placeholder outputs do not supply inference evidence.

## Configurations

Each batch starts with concurrent HTTP requests.
The input and output lengths apply to each request.

| Mode | Batch sizes | Input tokens per request | Output tokens per request |
|---|---|---|---|
| Ordinary decode | 1, 2, 4 | 32768 | 64, 8192 |
| MTP4 | 1, 2, 4 | 32768, 65536, 131072 | 64, 8192 |

These are 24 configurations. Each configuration must pass separately.
The 64-token output makes the measurement sensitive to prefill time.
Keep the 8192-token output for long decode measurement.

Use a token budget of 32768 and a maximum of 4 running requests for this comparison.
Use requests without a cached prefix.
Load the model before timing.
Complete initialization before timing.

Start timing before the client creates its request threads.
Stop timing after the client reads all responses as JSON and its request threads stop.
Include client submission, service request preparation, queue time, prefill, decode, constraints, sampling, process communication, and output transport.
Construct the exact prompt text before timing.
Exclude model loading, initialization, and trace analysis from timing.
Do not use time to first token as the full latency.

## Performance limit

Each configuration must have roofline efficiency of at least 50%.
The measured batch latency must be at most 2 times the roofline time for its recorded execution trace.
Use the selected GPU UUID's [measured peaks](calibration.md) and [precision rules](operators/evaluation.md#performance-calculation).

Calculate each serving operator's necessary work, with MTP context, proposals, verification, and acceptance.
Use the full sequence lengths for attention work.
Keep the measured MTP acceptance counts. Do not assume a fixed acceptance rate.

Do not count padding, repeated computation, or unnecessary copies as additional necessary work.
Do not add measured launch time or an empirical latency factor to the lower bound.
Keep source, binary, checkpoint, device, configuration, and timing identities in external artifacts.
Keep the lower-bound calculation available for review.

## Execution

Select decode requests first. Divide the available token budget between prefill requests.
Use CUDA Graph execution for batches that contain only decode requests.
Use ordinary execution for batches that contain prefill requests.

Apply the constraint mask before sampling in target and proposal computation.
Keep constraint state consistent with the accepted token history after speculative rejection.
Use MTP4 acceptance and private cache state selection from the model contract.

Use fusion for producer and consumer work when it decreases the full execution time.
Multiple CUDA streams are permitted for independent work with explicit dependencies.
Do not use PDL in the inference worker.

Use operator measurements and full HTTP measurements to select changes.
Use the [operator workload](operators/evaluation.md) for operator acceptance.
Operator measurements have no minimum efficiency requirement.

## Measurement command

Set `MODEL_ROOT` to the checkpoint directory.
Set `PEAK_PROFILE` to the selected GPU UUID's calibration file.
Select this UUID with `OH_MY_VLLM_GPU_UUID`.
Set `ARTIFACT_ROOT` to a new directory external to the repository.
Build the native modules and release executable with the [service setup](serving.md#start).
Run the full comparison:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.benchmark --peaks "$PEAK_PROFILE" --model "$MODEL_ROOT" --artifacts "$ARTIFACT_ROOT"
```

The command starts and stops the service for each mode.
It completes short initialization requests before timing.
It checks prompt lengths, output lengths, and uncached prefixes in the HTTP usage fields.
The default cache capacities are 1024 FA blocks and 1024 GDN slots.
Use `--blocks` and `--mamba-blocks` to change these capacities.

Use `--case ordinary:1:32768` to select its two output lengths.
Use `--case ordinary:1:32768:64` to select one configuration.
Use `--output-tokens 64` for measurements sensitive to prefill time.
Use `--output-tokens 32` for a short run.

A partial set or other output lengths cannot pass the full gate.
Repeated configurations, missing configurations, or a failed configuration prevent full acceptance.
The command rejects duplicate selections and cases that do not match the selected output lengths.

The worker writes an execution trace when `OH_MY_VLLM_EXECUTION_TRACE` is set.
The trace path must be external to the repository. It must not exist before the run.
The benchmark sets this variable for its service process.

## Lower-bound calculation

The [calculation](../development/serving/roofline.py) uses the specified precision of each matrix operation.
It includes full causal attention lengths, GDN work for the selected operand types, vocabulary projections, and MTP context and draft work.
It includes acceptance arithmetic and the measured accepted token counts.

Each trace event keeps its recorded batch, prefill chunk, query lengths, and proposal stage.
The calculation keeps separate calls. The calculation does not combine requests by their verification round numbers.
Each prefill chunk includes the weights and cache accesses necessary for that call.
Sampling follows completed prefill requests and recorded verification rows.

The calculation excludes intermediate tensor copies and padding.
Small operations without a specified throughput model do not increase the bound.

Each serving operator's time is the maximum of its HBM time, tensor-core time, FP32 time, and FP64 time.
Tensor-core time adds FP8, BF16, and TF32 work at the peak for each type.
The current worker executes these operators in sequence on one CUDA stream.
The gate uses the sum of these operator times, with the recorded call counts.
It does not permit resource overlap between dependent calls.

The global resource estimate is supplied for analysis. It does not set the gate.

The external artifacts contain responses, usage, execution shapes, MTP counters, bound components, and source identities.
The [artifact check](../development/serving/benchmark.py#L218)<!-- source: def verify_artifact_identity(arguments, expected): --> prevents acceptance if source, executable, native build, or checkpoint identities change.

The trace must have full prefill coverage, continuous accepted positions, and output counts that agree with HTTP usage.
Its single identity record must match the service mode. Its prompt lengths must match HTTP usage.
Each HTTP response must identify a different request in the measured batch.
Computation of the same work more than once or accepted positions that decrease prevent acceptance.
