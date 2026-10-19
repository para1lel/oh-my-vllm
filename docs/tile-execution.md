# Tile execution design

## Scope and authority

This design controls the `nightly` migration.
It replaces older operator performance gates, measured-peak rules, and mandatory operator measurements.
The [runtime contracts](requirements.md) still control service functions, process ownership, and numerical behavior.

The final implementation must execute all GPU tensor computation through tile dependencies and MegaKernel execution.
This applies to preparation, initialization, ordinary decode, prefill, MTP4, vision, sampling, and state computation.
Putting an ordinary operator launch behind a new Python interface does not complete its migration.

Migration can proceed in stages without a performance threshold.
Each delivered stage must pass applicable correctness checks and independent review.
Final acceptance has separate coverage, correctness, and [HTTP performance](#http-performance) conditions.

## Computation boundary

| Area | Final coverage |
|---|---|
| Language model | Embedding, 64 layers, normalization, quantization, projections, attention, and vocabulary projection |
| Prefill | FA, GDN chunk computation, metadata, reductions, and checkpoints |
| MTP4 | Context, four proposals, verification, acceptance, state selection, and endpoint gathers |
| Sampling | Penalties, masks applied on GPU, temperature, sorting, distributions, random selection, and RNG counters |
| Vision | GPU pixel resizing, normalization, patches, positions, 27 layers, merge, and embedding gathers |
| Preparation | GPU weight conversion, scale preparation, repacking, and activation tables |
| Auxiliary computation | Indexed gather/scatter, tensor initialization, state export, and GPU metadata computation |

CPU tokenization, media decoding, shape selection, grammar, stopping rules, and request control can stay on the CPU.
Any subsequent GPU implementation of these calculations must use tile tasks.

CUDA allocation, views prepared on the host, streams, events, graphs, and synchronization are infrastructure.
Unchanged-byte H2D, D2H, and D2D transfers can use CUDA transfer operations.
TMA and DMA can supply task data.
These operations must obey data dependencies and buffer lifetimes.
Model tensor initialization is computation, even when its output contains only zeros.

## Ownership and execution

Rust controls admission, request scheduling, logical KV identities, and state commit.
The Python worker owns device tensors, execution plans, and private state slots.
ZMQ DEALER and msgpack keep the existing process boundary.

Preparation builds a static or parameterized directed acyclic graph (DAG) of tasks.
Each plan has fixed buffer addresses for its lifetime.
Runtime metadata supplies lengths, page tables, state slots, valid rows, and the execution epoch.
Task implementations are shared across layers.
Layer numbers and weight addresses are task parameters.

The initial engine uses CUDA C++ and CUTLASS device components.
CUDA C++ or CuTe DSL can supply subsequent task implementations.
One persistent kernel is a starting point, not a restriction.
Several specialized persistent kernels can share the same task and event contract.
Kernel families, algorithms, and scheduling can change with research and performance evidence.

All inference submissions end at the related Execute boundary.
Preparation submissions can run during initialization or preparation.
There is no permanent GPU service loop in this design.
Decode-only execution keeps CUDA Graph submission with stable plan buffers.

The first full-model milestone covers ordinary decode from embedding through vocabulary projection.
Its CPU constraint processing and separate GPU sampling can initially use the existing path.
Sampling must join tile execution before final acceptance.
Prefill, MTP4, and vision follow the same migration rule.

## Tasks, memory, and progress

Each task identifies its operation, input and output regions, precision rules, state permissions, and resource class.
The graph has data dependencies, state-version dependencies, and buffer-lifetime dependencies.
Physical address overlap alone cannot determine these dependencies.

Fused projection tiles, attention partitions, state-head tiles, and reductions are initial task units.
Small local operations can stay in their producer or consumer task.
Tile dimensions, task ranges, queue layout, and CTA roles are implementation choices.
They must not change the model contract.

Ready queues have fixed capacity.
An executor CTA receives only a ready task.
The planner rejects dependency cycles before launch.
It checks buffer capacity and resource residency.
The scheduler and producers must have a progress path when consumers cannot execute.
Multiple persistent kernels must have an explicit residency and progress plan.

Task completion follows all related TMA, MMA, and ordinary writes.
Device-scope release/acquire ordering makes these writes visible to consumers.
An epoch identifies reused events, descriptors, and queue entries.
Completion counters distinguish queued, running, and finished tasks.
Queue overflow, invalid metadata, and incomplete execution produce an error.

State commit is not permitted after failure.
The existing worker removal and epoch cleanup procedure discards possibly modified state after execution failure or timeout.
Resetting a queue does not make that state safe for reuse.
Cancellation keeps the existing operation boundaries.

## Numerical and state contracts

Keep checkpoint values, FP8 scales, permitted operand types, rounding points, and named numerical tolerances.
Do not change tolerances to get performance acceptance.
Keep full attention contexts, the 262144-token total limit, and 784-token checkpoints.
Keep FP32 live carry and BF16 persistent state where the model contract specifies them.

Candidate writes stay private until state commit is permitted after validation.
Acceptance selects state slots instead of copying the full recurrent state.
CPU constraint state follows accepted tokens.
Inference behavior must agree across task schedules, plan reuse, and legal batch shapes.

The 24 HTTP performance cases do not replace numerical, state, media, or random-sampling checks.
Existing independent references and state checks must survive removal of the old timing tools.
New checks must cover failures that those references cannot detect, such as event reuse or scheduler deadlock.
State why each new test is necessary before adding it.

## HTTP performance

Keep the 24 configurations and full request timer in [the HTTP procedure](performance.md).
Keep a token budget of 32768 and a maximum of 4 running requests for this comparison.
The final threshold is at least 50% for each configuration.
Migration commits can report unmeasured or failed configurations.

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
It does not add independent operator latency estimates.

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
History reads repeat when subsequent decode steps use that history.
Fixed reference algorithms and precision determine arithmetic counts.
Numerically valid implementation algorithms can differ from the reference accounting algorithms.

Extra tiles, split-K partitions, launches, padding, repeated computation, and temporary copies do not increase credited work.
Exclude intermediate storage that fusion can remove.
Specify ordinary scalar arithmetic and all omitted operations explicitly.
Do not invent throughput factors for operations without a resource model.

Artifacts bind each row to source, binary, checkpoint, device, accounting version, and timing identities.
Old results cannot supply acceptance evidence for the new calculation.
The two README files contain only the required 24-row performance table.

## Research inputs

These sources inform implementation choices, not additional acceptance gates.

- [MPK, revision 2026-06-10](https://arxiv.org/html/2512.22219v2): task/event graphs and combined static and dynamic scheduling. Its register limits motivate specialized kernel families.
- [Event Tensor, revision 2026-04-21](https://arxiv.org/html/2604.13327v2): parameterized events and runtime task ranges for variable shapes.
- [Cohere, 2026-09-08](https://cohere.com/blog/megakernels): task descriptors, CTA roles, and finite decode submissions. Its H100 BF16 implementation keeps ordinary prefill.
- [TileRT, 2026-05-21](https://www.tilert.ai/blog/speed-as-the-next-scaling-law.html): persistent execution and tile pipelines. The full internal scheduler is not public source.

These works do not prove full model coverage or a speedup for this service.
Use their mechanisms where measurements and numerical checks support the choice.
The [migration plan](tile-migration.md) defines the implementation sequence.
