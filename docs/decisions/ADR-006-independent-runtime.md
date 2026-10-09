# ADR-006: Independent GPU runtime

Date: 2026-09-21. Status: accepted and implemented.
This decision supersedes [ADR-002](ADR-002-gpuworker-adapter.md) and [ADR-005](ADR-005-v2-model-runner.md).

## Decision

Keep Rust service, scheduling, logical KV, and Python GPU execution.
Use small project implementations and third-party libraries instead of a vLLM runtime dependency.
Port selected code only with provenance and licenses. Do not copy the framework wholesale.
Builds, tests, and inference must work without vLLM packages, checkout, old environment, or compiled caches.


Select compatible stable dependencies in dependency order and pin the combination with satisfactory compatibility tests.
Use the host driver and working CUDA/compiler tools, with higher-level libraries in the project environment.
Keep host locations in `LOCAL.md`.

## Consequences

The project controls FP8 weight/scales, GQA/GDN, MTP rollback, loading, cache state, sampling, graphs, and service adapters.
Keep inference-path independent references with FP64 computation and all existing features.
Review each milestone and measure affected hot paths.

[Requirements](../requirements.md) and [acceptance](../acceptance.md) define active scope and measured identity.

## Future boundaries

New architectures and GPU backends must have concrete contracts and tests before support claims.
Multiple GPUs must have rank-local state and collectives while Rust keeps scheduling ownership.
A DSpark implementation must have its own draft state, target hidden features, and variable verification/commit/rollback.
Its five-layer BF16 design, feature indices, and block distinctions are in [architecture](../architecture.md#extension-boundaries).
Do not add empty interfaces or multi-node support for this task.
