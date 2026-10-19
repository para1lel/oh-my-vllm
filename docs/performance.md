# End-to-end requirements

Use one NVIDIA B200 and the [model contract](operators/model.md).
Use the full Rust service and Python inference worker.
Placeholder outputs do not supply inference evidence.
The [GPU execution rules](execution.md) control execution and final acceptance.

## Configurations

Each batch starts with concurrent HTTP requests.
The input and output lengths apply to each request.

| Mode | Batch sizes | Input tokens per request | Output tokens per request |
|---|---|---|---|
| Ordinary decode | 1, 2, 4 | 32768 | 64, 8192 |
| MTP4 | 1, 2, 4 | 32768, 65536, 131072 | 64, 8192 |

These are 24 configurations. Each configuration must pass separately at final acceptance.
The 64-token output makes the measurement sensitive to prefill time.
Keep the 8192-token output for long decode measurement.
Use a token budget of 32768 and a maximum of 4 running requests.
Use requests without a cached prefix.

Load the model before timing.
Complete initialization before timing.
Construct the prompt text before timing.
Start timing before the client creates its request threads.
Stop timing after the client reads all responses as JSON and its request threads stop.
Include submission, request preparation, queue time, prefill, decode, constraints, sampling, process communication, and output transport.

Exclude model loading, initialization, and trace analysis from timing.
Do not use time to first token as the full latency.

## Performance limit

Use [nominal resources and the global formula](execution.md#http-performance).
Intermediate changes can report failed or unmeasured configurations.
Final acceptance must have efficiency of at least 50% in each of the 24 configurations.
The full batch latency must be at most 2 times its reference time.
Operator timing and public comparisons have no independent acceptance gate.

## Measurement command

Set `MODEL_ROOT` to the checkpoint directory.
Select one GPU UUID with `OH_MY_VLLM_GPU_UUID`.
Set `ARTIFACT_ROOT` to a new directory external to the repository.
Build the native modules and release executable with the [service setup](serving.md#start).
Run the full comparison:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.benchmark --model "$MODEL_ROOT" --artifacts "$ARTIFACT_ROOT"
```

The command starts and stops the service for each mode.
It completes short initialization requests before timing.
It checks prompt lengths, output lengths, and uncached prefixes in HTTP usage.
The default cache capacities are 1024 FA blocks and 1024 GDN slots.
Use `--blocks` and `--mamba-blocks` to change these capacities.

Use `--case ordinary:1:32768` to select its two output lengths.
Use `--case ordinary:1:32768:64` to select one configuration.
Use `--output-tokens 64` for measurements sensitive to prefill time.
Use `--output-tokens 32` for a short run.
A partial set or other output lengths cannot pass the full gate.
Duplicate configurations, missing configurations, or a failed configuration prevent full acceptance.

The worker writes an execution trace when `OH_MY_VLLM_EXECUTION_TRACE` is set.
The trace path must be external to the repository and must not exist before the run.
The benchmark sets this variable for its service process.

## Lower-bound calculation

The [reference calculation](../development/serving/accounting.py) uses version `http-reference-v1`.
It has no runtime algorithm selector or measured calibration input.
The [accounting assumptions](accounting.md) specify its arithmetic and traffic that has no credit.

The [trace validator](../development/serving/trace.py) first checks full prefill, continuous accepted positions, proposal dependencies, and output counts.
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

The calculation adds all resource work before it takes the maximum.
It does not add independent operator times.
Extra tiles, padding, launches, repeated computation, or temporary copies cannot increase credited work.

Artifacts contain responses, the manifest and its digest, the selected trace digest, resource totals, and acceptance counters.
The manifest binds the full request timer to the artifact identity.
That identity binds source, executable, native modules, checkpoint, tokenizer, device, and accounting version.
The [artifact check](../development/serving/benchmark.py) rejects changes during the run.
Old measurement artifacts cannot supply acceptance evidence for this version.
