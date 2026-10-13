# Measurement and extension boundaries

## Phase model

`performance/` maps observed semantic work to official compute and HBM limits.
Rust records request submission, first-token, and last-token times and the corresponding steps.
The binary has a build stamp for Cargo inputs and Rust sources.
The collector validates this stamp, source identities, checkpoints, loaded libraries, and separate repetition intervals.
Unknown GPU work blocks acceptance.
See the [latency model](../latency-model.md).

## Extension boundaries

New models must have validated loading, semantic contracts, and cache layouts.
Multiple GPUs must have new process ownership and collective contracts.
Other NVIDIA backends must have device-specific providers and measured acceptance.
Current interfaces make no such support claim.

DSpark runtime support applies to the validated local checkpoint architecture.
Different draft checkpoints must have new loading, semantic, numerical, and acceptance tests.
Use the [acceptance index](../acceptance.md) for source identities and measured scope.

## Agentic test ownership

The inference service and GPU worker are separate from the agent sandboxes.
The host collector owns recording proxies, SDK trace endpoints, context checks, and resource measurements.
Each sandbox owns a new CPU project, OMP state, tools, and temporary files.
The collector removes these owners after completion, failure, or cancellation, and keeps logs in a separate evidence directory.
[ADR-017](../decisions/ADR-017-disposable-native-goal-soaks.md) specifies native goals and cleanup.
