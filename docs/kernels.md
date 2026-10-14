# Kernel development

Use the [operator contracts](requirements/kernels.md), [harness instructions](../development/kernels/AGENTS.md), and [performance protocol](requirements/performance.md).
The current provider migration is in the [handoff](handoff.md). Do not treat old measurements as new acceptance.
Read [algorithm research and measured lessons](kernels/optimization.md) before selecting a new numerical or scheduling path.

## Implementation sources

[CUDA backend](../python/oh_my_vllm/kernels/cuda_backend/) owns device code and native bindings.
[Semantic IR](../python/oh_my_vllm/ir/) owns operation contracts, references, mutation schemas, and provider selection.
[Target model](../python/oh_my_vllm/models/qwen.py) supplies call-site context.

Schedules, layout guards, launch parameters, and specialization thresholds belong in those sources.
Use repository CUDA for operations that Torch does not supply. Native CUDA libraries are permitted.
Python bindings are transport into owned kernels, not external operator providers.

Keep each backend `.cu`, `.cuh`, and `.py` file at most 800 lines.
Use pinned clang-format and the definition-spacing hook. Keep necessary numerical and lifetime explanations concise.

```bash
scripts/with-env.sh python scripts/format_cuda.py
scripts/with-env.sh python scripts/format_cuda.py --check
scripts/with-env.sh python scripts/check_definition_spacing.py
```

## Numerical and state contracts

Each operation has input/output shape, dtype, stride, alignment, and mutation contracts.
Use an independent Torch reference with the specified precision and rounding boundaries.
Higher precision does not replace prescribed FP32 division or observable BF16 casts.
Keep written-state witnesses and untouched storage checks, with returned-output comparisons.

Fusion must keep mathematical semantics, accepted-state boundaries, and numerical tolerances.
Cache and state storage cannot become temporary donation targets.
Check changed inputs and metadata during graph replay. Reject unsupported layouts or invalid bindings before launch.

## Formal cases

Derive the largest legal invocation of each implementation path from all fifteen framework workloads.
Keep phases distinct when their paths differ. Deduplicate identical configurations.
Do not independently maximize dimensions that cannot occur together.
Keep formal cases, independent references, the harness, and summarized evidence in the repository.

Operator timing diagnoses bottlenecks and compares candidates. The full framework supplies the final performance gate.
Compare complete operations, with required copies, reductions, and conversion.
Use interleaved candidate/baseline measurements. Profile separately from acceptance timing.
Keep original failed attempts and unavailable counters visible.

## Tool and build identity

TileFoundry is a development tool. Its updates must not silently change production build dependencies.
Pin native headers independently and record their source identities with compiler options and loaded module hashes.
Finish compilation and graph capture before timing.
Keep temporary tuning and raw traces in external storage. Stop owned GPU processes after each run.

Read [profiling](profiling.md) for host/device attribution and [test commands](testing/performance.md) for collection.
