# Public attention comparison

These tools compare the FA and GDN attention cores with pinned public implementations.
The comparison includes necessary input changes and state exports.
It does not measure the seven full serving boundaries.
Use the [operator evaluation](../../docs/operators/evaluation.md) for those boundaries.
Use the [HTTP procedure](../../docs/performance.md) for the 24 HTTP configurations.

## Environment

Use an environment that is separate from inference.
Keep its path and the GPU settings in the ignored `LOCAL.md`.
The [dependency input](../../requirements/baselines.in) gives the direct versions and source commits.
The [lock file](../../requirements/baselines.txt) gives the package versions and archive hashes.
The [header file](headers.json) gives the public source header commits and archive hashes.

Set these variables from your local configuration:

```bash
export BASELINE_ENV=/external/path/environment
export BASELINE_CACHE=/external/path/cache
export BASELINE_PYTHON=/path/to/python3.12
export CUDA_HOME=/path/to/cuda
export OH_MY_VLLM_CUTLASS=/path/to/project-cutlass
export OH_MY_VLLM_GPU_UUID=GPU-uuid
```

Install the environment:

```bash
python3 -m development.baselines.bootstrap \
  --environment "$BASELINE_ENV" --cache "$BASELINE_CACHE" \
  --python "$BASELINE_PYTHON"
```

The installer checks each source archive hash before extraction.
It does not change public source code.
Public source archives do not contain their Git submodules.
The launcher supplies the pinned header paths through public compiler flags.
The compiler does not use the project's CUTLASS headers for public implementations.
The public build disables unrelated distributed expert builds and automatic package installation.

Use the [launcher](run.sh) for GPU work.
It keeps the project GPU wrapper lock and sets separate compiler cache paths.
These paths include the cuDNN compiled plans and the CUDA driver cache.

The launcher keeps CuTe PTX and CUBIN artifacts. The report saves cuDNN GDN binaries from memory with filenames that contain their content hashes.
These files identify the kernels used for the measurement. The timed interval does not include artifact collection.

The report also saves compiled CuTe CUBIN files with content hashes. Later compilation cannot replace these copies with different contents.
Compilation does not use the CuTe or cuDNN disk cache. Cached executables can omit the CUBIN artifact.

## Input contract

The [input module](inputs.py) defines `CoreInputs` and `Prepared`.
A local factory has the signature `prepare(case, inputs) -> Prepared`.
Pass its module and function as `--own MODULE:prepare`.
Keep the factory in the project source tree.

The factory must use the supplied tensors.
It must not change them.
Its `outputs` dictionary contains an `output` tensor.
GDN also supplies its private `state_pool` output tensor.
Its `reset` callback restores mutable storage before each invocation.
The callback runs before the timed interval.

Its `run` callback includes each necessary launch and storage change.
The factory metadata supplies `source_files` and `binaries` as nonempty path lists.
The report records their file hashes.

The report checks these hashes again after timing and at the end of the run.
It also checks the public package contents and compiled public kernels.

FA uses these input tensors:

| Tensor | Shape and type |
|---|---|
| `q` | `[total_queries, 24, 256]`, BF16 |
| `cache` | `[pages, 2, 784, 4, 256]`, BF16, NHD |
| `sequence_offsets` | `[batch + 1]`, INT32 query offsets |
| `page_offsets` | `[batch + 1]`, INT32 page offsets |
| `page_indices` | Flat INT32 physical page indices |
| `last_page_lengths` | `[batch]`, INT32 |
| Output | `[total_queries, 24, 256]`, BF16 |

The cache already contains the complete context.
Attention is causal, with query positions at the end of each context.
The softmax scale is `1 / sqrt(256)`.
The report records each public implementation's intermediate precision.
All implementations must pass the same numerical checks for this boundary.
FA planning runs before timing because its metadata is fixed for each case.

`inputs.sequence_offsets` contains host offsets.
Use `inputs.tensors["sequence_offsets"]` for the device tensor.

The FA candidates use FlashInfer's paged prefill and decode APIs.
They include the automatic, FA2, FA3, cuDNN, CuTe DSL, and TensorRT-LLM backends.
FA3 can run only on an SM90a device.
CUDA-core decode uses FP32 probabilities and supports one query per request.
The Tensor Core decode path uses BF16 probabilities.

Decode APIs can process balanced prefill batches when the API supports the query length.
The adapter clears output before each CuTe run that uses atomic reduction.
The timed call includes this work.
The report records API, architecture, and shared-memory limits as unsupported paths.

GDN uses these input tensors:

| Tensor | Shape and type |
|---|---|
| `q`, `k` | `[total_queries, 16, 128]`, BF16, before normalization |
| `v` | `[total_queries, 48, 128]`, BF16 |
| `a`, `b` | `[total_queries, 48]`, BF16 gate inputs |
| `A_log`, `dt_bias` | `[48]`, FP32 |
| `state_pool` | `[slots, 48, 128, 128]`, BF16, `[slot, head, value, key]` |
| `read_slots` | `[batch]`, INT32 |
| `checkpoint_rows`, `checkpoint_slots` | INT32 export rows and destination slots |
| Output | `[total_queries, 48, 128]`, BF16 |

The initial state is nonzero.
Read slots and unused guard slots must stay unchanged.
Verification exports each private row.
Prefill exports each 784-token boundary and the final row.
The position of each sequence controls its first boundary.
The live state stays FP32 across exports.

The state head uses key head `head // 3`.

The public prefill adapters divide calls at the required checkpoint rows.
They keep FP32 carry between calls and export BF16 checkpoints inside timing.
They batch sequences when segment starts and lengths match.
They include packing, normalization, gates, state conversion, and output copies.
FlashQLA uses `auto_cp=False` to keep the complete history.

FlashInfer candidates use prefill with context parallelism enabled, disabled, and selected automatically.
They also include its cuDNN prefill path and BF16 state decode path.
The Cake paths must support the supplied head counts and state type.
Cake CP uses one public direct-composite plan per checkpoint segment.
The caller supplies graph capture for these plans.

cuDNN plans keep the complete history.
The adapter uses the default piece chain or value split only when the plan selects that exact path.
The report rejects default plans with permitted decay warmup cuts.
A second candidate sets `batch_invariant=True` to keep the complete history for those shapes.

Chunk APIs can round normalized recurrent Q/K to BF16 when this conversion agrees with the precision policy.
Each candidate must pass independent numerical checks and keep all private exports.

## Cases and reports

Print the modeled interest set:

```bash
PYTHONPATH=python:. "$BASELINE_ENV/bin/python" -m development.baselines plan
```

Supply the full execution trace and its HTTP result files for recorded HTTP shapes:

```bash
PYTHONPATH=python:. "$BASELINE_ENV/bin/python" -m development.baselines plan \
  --trace /external/path/ordinary-execution.jsonl \
  --http-result /external/path/ordinary-1-32768-64.json
```

Repeat the arguments for all required files.
Supply one result file per HTTP configuration.
The trace checks must show complete generation with the specified output length.

The fixed selection includes each prefill shape.
It also includes the first, middle, and final decode shape for each configuration and batch pattern.
This selection includes verification and draft calls.
It does not claim coverage of each decode context.

The report keeps the full file hashes and selected source lines.
It records the complete selected manifest before numerical checks or timing.

Measure each mode in a separate run:

```bash
development/baselines/run.sh "$BASELINE_ENV" "$BASELINE_CACHE" \
  python -m development.baselines run \
  --own development.baselines.gdn_native:prepare \
  --operator gdn_attention --mode eager \
  --output /external/path/gdn-eager.json
```

The report starts with `passed: false`.
Independent CPU FP64 references check the fixed small numerical shapes.
The limits come from the existing named tolerances.
Large shapes compare public outputs with the local implementation on identical inputs.
These comparisons include each exported state and unchanged state storage.
Unchecked or failed public paths cannot supply the comparison time.

The default cuDNN candidate also uses a small 705-token numerical case.
This additional case checks an exact piece chain with nonzero incoming state.
It does not change the fixed numerical cases or supply a modeled performance result.

Each public candidate must pass at least one supported independent check for the selected prefill or recurrent path.
All supported checks must pass.
Unsupported checks stay explicit and do not count as passes.
Errors and failed checks prevent eligibility.
The default cuDNN candidate also checks each measured segment's exact plan selection.

Each measured invocation starts after state reset and L2 eviction.
The eviction buffer is larger than twice the selected GPU L2 capacity.
CUDA events measure all necessary operations on the caller stream.
The procedure uses two warmups and five measured runs.
The report keeps each duration, its median, and its spread.

Each shape compares the local median with the fastest eligible public median.
The local time must be less than or equal to that time.
A shape passes only when its checked, applicable public comparison passes.
For overall acceptance, supply the selected representatives from all 24 HTTP configurations.

Source hashes, package identities, GPU settings, input hashes, and selection rules stay in the report.
The GPU check compares the selected UUID with the device UUID that Torch reports.
Keep all reports external to the repository.
