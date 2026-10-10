# ADR-010: Phase latency and dependency execution

Date: 2026-10-09. Status: implementation in acceptance.

## Decision

Use per-request framework wall time for prefill and decode.
Use two full warmups and five measured repetitions for each of fifteen fixed workloads.
Except for prefix-hit prefill, each phase must have median wall time at most three times its median theoretical lower bound.
Its spread, `(max-min)/median`, must be at most 10%.

Record the prefix-hit prefill latency ratio without a three-times gate.
This short cached phase keeps the 10% spread check. Prefix-hit decode keeps the two gates.
Use [requirements](../requirements.md) for the fixed shapes and capacity tests.

Derive the bound from the observed effective schedule and fixed model semantics.
Count parameter reads, accepted and rejected verification rows, required draft work, sampling, KV, and recurrent state.
Do not add padding or redundant implementation work to the bound.

Use official single-GPU dense B200 peaks, with HBM at 7.7 TB/s.

Branches with no dependency can execute at the same time.
The bound combines a relaxed dependency path and shared-resource work.
Finite ideal cache credit applies across the resource accounting interval.

The current path relaxation gives compute nodes zero service and gives each read the full cache credit.
These relaxations decrease the bound. They do not model atomic full-matrix serialization.
See [the latency model](../latency-model.md) for equations, source links, and limits.

Use manual CUDA Graph capture, multiple streams, and PDL for model execution.
Capture must restore persistent writes after warmup and on failure.
A side stream must join before its operation returns, with exception paths.
A PDL consumer waits for its producer before dependent reads.
Source, build-input, loaded-binary, and runtime identities are recorded with the execution contract.

## Consequences

The gate measures the complete framework interval against GPU resource limits.
Host, transport, launch, and redundant computation overhead decrease the available margin.
GPU events and profiler counters identify this overhead. They do not change the theoretical rates.

Owned FP8 GEMM supplies current-stream and PDL control with the checkpoint's arbitrary FP32 group scales.
GDN preparation forks its separate gate projection from its QKVZ and convolution branch.
Prefill graphs use different bounded cache families and allocator topology reuse.
Large operators use measured tile and traversal selection. See [profiling](../profiling.md).

Each original attempt stays in external storage, with failures and interruptions.
Portable summaries identify their source and diagnostic or formal scope.
A source change must have numerical, capacity, service, and applicable performance checks before complete acceptance.
