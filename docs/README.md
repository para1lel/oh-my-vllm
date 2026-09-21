# Documentation index

Read [AGENTS.md](../AGENTS.md) and [handoff.md](handoff.md) first. AGENTS.md governs
project constraints; handoff.md records current evidence, blockers and pending runs.

- [Requirements](requirements.md): model, ownership, functionality and acceptance.
- [Architecture](architecture.md): current runtime and module responsibilities.
- [Design](design.md): wire messages, logical/physical cache and scheduler lifecycle.
- [Development](development.md): conda wrappers, build and timestamped logging.
- [Testing](testing.md): actual-path FP64, text, feature and performance commands.
- [Acceptance](acceptance.md): complete matrix, raw evidence and coverage limits.
- [Profiling](profiling.md): host timings and optional kernel diagnostics.
- [Plan](plan.md): completed milestones and remaining work.
- [Serving](serving.md): implemented OpenAI-compatible subset, client commands and real GPU acceptance evidence.
- [Contributing](../CONTRIBUTING.md): reviews, checks and commits.
- [ADR001](decisions/ADR-001-zmq-socket-type.md): DEALER socket choice.
- [ADR002](decisions/ADR-002-gpuworker-adapter.md): installed Worker and physical IDs.
- [ADR003](decisions/ADR-003-mtp-state-slots.md): speculative states and BF16 SSM.
- [ADR004](decisions/ADR-004-openai-serving.md): Rust-first dual-API serving direction.

Ordinary/MTP text, actual-path probes and feature combinations have passed checks.
All nine performance rows passed. Consult acceptance.md for exact configuration
and handoff.md for history rather than relying on old commit-time status entries.
