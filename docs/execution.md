# GPU execution

## Operator scheduling

Rust controls requests, scheduling, logical KV identities, and state commit.
Python executes the model through CUDA operators and the existing ZMQ DEALER and msgpack boundary.
The [operator boundaries](operators/fusion.md) specify computation and private state ownership.

The worker selects execution once for each Execute message.
If any scheduled row contains prefill, all fragments of that Execute use Eager execution.
A decode-only Execute uses CUDA Graph recording and replay.
This rule also controls sampling, MTP context, proposals, and acceptance.
Vision and weight preparation use Eager execution.

Eager and CUDA Graph execution use the same native operators.
Python calls these operators in model dependency order.
CUDA kernels can fuse operations and use matrix tiles internally.
There is no tile task scheduler, task DAG, or persistent task engine.

## Graph resources

A workspace owns weights, cache references, metadata, and temporary tensors for its recorded operations.
Warmup prepares storage before recording.
Sampling and MTP acceptance restore their RNG state after warmup.
Buffer addresses stay fixed while a graph exists.
Metadata changes on the caller's stream before replay.
Replay uses that same stream.

Recording uses a different nonblocking stream and restores the caller's stream on exit.
A failed recording releases the resources created before failure.
A completed recording transfers its resource leases to the workspace.
Closure waits for GPU work, destroys the graph executable, and then releases its owners and leases.
An execution failure keeps the existing worker removal and state cleanup rules.

## Numerical and state contracts

Keep checkpoint values, FP8 scales, operand types, and rounding points from the [model contract](operators/model.md).
Keep full attention contexts, the 262144-token total limit, and 784-token checkpoints.
Candidate writes stay private until validation permits state commit.
Acceptance selects state slots without copying the full recurrent state.
CPU constraint state follows accepted tokens.

Use the PyTorch FP64 [full-model logits comparison](operators/evaluation.md).
Do not use single-operator tolerances.
Keep service, graph resource, memory safety, and state ownership checks.
The 24 HTTP cases do not replace these checks.

## HTTP performance

Keep the 24 configurations and full request timer in [the HTTP procedure](performance.md).
Keep a token budget of 32768 and a maximum of 4 running requests for this comparison.
Each configuration must have efficiency of 50% or more at final acceptance.
Intermediate changes can report unmeasured or failed configurations.

Use the dense HGX B200 values from [NVIDIA's data sheet, page 8](https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf).
The rates are HBM 7.7 TB/s, FP8 4.5 PFLOP/s, BF16 2.25 PFLOP/s, and TF32 1.1 PFLOP/s.
Scalar rates are FP32 75 TFLOP/s and FP64 37 TFLOP/s.
Do not substitute sparse rates, GB200 rates, or measured calibration.

```text
reference_seconds = max(
    necessary_bytes / HBM_rate,
    FP8_work / FP8_rate + BF16_work / BF16_rate + TF32_work / TF32_rate,
    FP32_work / FP32_rate,
    FP64_work / FP64_rate,
)
efficiency = reference_seconds / HTTP_seconds
```

This is a resource lower-bound estimate with specified mathematical and memory assumptions.
It is not a proof of the minimum physical latency.
The calculation adds work across the full trace before it takes the maximum.
It does not add latency estimates for different operators.

A versioned reference schedule determines logical model passes.
Its request order, decode priority, prefill allocation, short tails, and checkpoint rules are deterministic.
An identity-bound benchmark manifest maps request ordinals to response and trace request IDs.
Validate unique IDs, full membership, and configured input/output lengths before accounting.
HTTP arrival order must not change reference grouping.

The trace supplies token positions, full contexts, proposals, verification rows, and measured acceptance counts.
The reference schedule must obey their causal dependencies.
It cannot invent proposals or combine incompatible steps.

Count each necessary weight and scale region once per reference pass, shared across that pass's batch.
Count each necessary KV or state access once at its reference position.
Subsequent decode steps read the history again.
Fixed reference algorithms and precision determine arithmetic counts.
Numerically valid implementation algorithms can differ from the reference accounting algorithms.

Extra tiles, split-K partitions, launches, padding, repeated computation, and temporary copies do not increase credited work.
Exclude intermediate storage that fusion can remove.
Specify ordinary scalar arithmetic and all omitted operations explicitly.
Do not invent throughput factors for operations without a resource model.

Artifacts bind each row to source, binary, checkpoint, device, accounting version, and timing identities.
Old results cannot supply acceptance evidence for the new calculation.
The two README files contain only the required 24-row performance table.
