# ADR-002: Historical GPUWorker adapter

Date: 2026-09-19. Status: superseded by [ADR-006](ADR-006-independent-runtime.md) and [ADR-007](ADR-007-ttft-and-cache-capacities.md).

## Original decision

The first prototype used a vLLM GPUWorker adapter while Rust controlled scheduling and logical KV.
The adapter translated Rust block tables, scheduled tokens, and request state into worker inputs.
Its cache translation included a physical-stride mapping and a logical capacity based on `floor(num_gpu_blocks/3)`.
This allowed early scheduler and MTP tests with the target checkpoint.

## Replacement

The independent runtime removed the adapter and its vLLM dependency.
Block IDs now directly address FA/GDN slots.
The historical CLI capacity unit stays, but the old physical mapping does not.
GDN capacity has its own limit to prevent the allocation of unused state again and again.
