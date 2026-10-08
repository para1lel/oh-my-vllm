# Document index

English files are authoritative. Each has a full same-directory Chinese translation.
Start with [requirements](requirements.md), [architecture](architecture.md), and [current work](handoff.md).

## Guides

| Document | Purpose |
|---|---|
| [Development](development.md) | Environment, configuration, dependencies, and build. |
| [Tests](testing.md) | CPU/GPU checks and formal acceptance procedures. |
| [Service](serving.md) | HTTP subset, masks, thinking, state, and limits. |
| [Kernels](kernels.md) | CUDA, pinned TileLang, formal cases, and development tools. |
| [Profiling](profiling.md) | Diagnostics, counters, provenance, and timing limits. |
| [Acceptance](acceptance.md) | Sole index of measured source and evidence scope. |
| [Audit](audit.md) | Open findings and compact closed-fix index. |
| [Writing](writing.md) | ASD-STE100 rules, translations, and automated checks. |
| [Glossary](glossary.md) | Project technical nouns, verbs, and definitions. |
| [Tutorial](../code-journey/README.md) | Twelve interactive Chinese chapters and project source coverage. |
| [Contribution](../CONTRIBUTING.md) | Review, checks, and commit procedure. |

## Decision records

| Record | Status and purpose |
|---|---|
| [ADR-001](decisions/ADR-001-zmq-socket-type.md) | Active: DEALER transport. |
| [ADR-002](decisions/ADR-002-gpuworker-adapter.md) | Superseded: initial GPUWorker adapter. |
| [ADR-003](decisions/ADR-003-mtp-state-slots.md) | Active: BF16 MTP state and block 784. |
| [ADR-004](decisions/ADR-004-openai-serving.md) | Active: Rust HTTP and OMP acceptance with the target checkpoint. |
| [ADR-005](decisions/ADR-005-v2-model-runner.md) | Superseded: transitional V2 runner. |
| [ADR-006](decisions/ADR-006-independent-runtime.md) | Active: independent runtime and extension boundaries. |
| [ADR-007](decisions/ADR-007-ttft-and-cache-capacities.md) | Active: TTFT protocol and different FA/GDN capacities. |
| [ADR-008](decisions/ADR-008-semantic-ir.md) | Active: semantic operations and compiled GPU units. |

Architecture includes the former design document.
Current work includes the former plan.
Kernel development combines the former CUDA and TileLang guides.
The audit keeps open problems and useful fix identities. Detailed task history stays in Git.

Keep host-specific maintenance in ignored `LOCAL.md`.
