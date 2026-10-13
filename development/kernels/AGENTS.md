# Operator harness

Read [kernel contracts](../../docs/kernels.md), [requirements](../../docs/requirements.md), and [profiling](../../docs/profiling.md).
Derive formal cases statically from legal invocations in the complete workload matrix. Deduplicate identical configurations.
Keep independent mathematical references, shape/dtype contracts, mutation witnesses, and numerical tolerances.

FP64 is not a substitute for specified FP32 quantization or observable BF16 casts.

## Measurement

Compare complete operations, with required copies and reductions. Use interleaved candidate/baseline pairs.
Profile separately from timing acceptance. Report unavailable counters and failed attempts.
Fusion and padding must not inflate the theoretical bound.

Framework phase latency is the final performance gate.
Operator timings guide diagnosis and candidate selection.
Keep source/header/compiler identities, formal cases, harness code, and portable summaries.

Keep raw traces and temporary tuning in external storage. Release owned GPU workers after each run.
