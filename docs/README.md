# Documentation index

[中文阅读版](README.zh.md). Agents use the English originals; update each
same-directory `.zh.md` companion whenever its English source changes.

Read [AGENTS.md](../AGENTS.md) and [handoff.md](handoff.md) first. AGENTS.md governs
project constraints; handoff.md records current state and open work.

- [Requirements](requirements.md): model, ownership, functionality and acceptance.
- [Architecture](architecture.md): current runtime and module responsibilities.
- [Design](design.md): wire messages, logical cache and scheduler lifecycle.
- [Development](development.md): conda wrappers, build and timestamped logging.
- [Testing](testing.md): CPU/GPU suites, FP64 probes, features and performance commands.
- [Acceptance](acceptance.md): current evidence and a dated index of historical artifacts.
- [Code audit 2026-09-23](audit-2026-09-23.md): open findings and remediation batches.
- [Plan](plan.md): completed stages and next work.
- [CUDA development](cuda-development.md): native kernels, frozen comparison, operator gates.
- [TileLang development](tilelang-development.md): frozen reference and TileFoundry tools.
- [Profiling](profiling.md): host timings and kernel diagnostics.
- [Serving](serving.md): OpenAI-compatible subset, client commands and service checks.
- [Contributing](../CONTRIBUTING.md): reviews, checks and commits.
- Decisions: [ADR-001](decisions/ADR-001-zmq-socket-type.md) DEALER socket;
  [ADR-002](decisions/ADR-002-gpuworker-adapter.md) GPUWorker adapter (superseded);
  [ADR-003](decisions/ADR-003-mtp-state-slots.md) MTP state slots and BF16 SSM;
  [ADR-004](decisions/ADR-004-openai-serving.md) Rust-first serving;
  [ADR-005](decisions/ADR-005-v2-model-runner.md) V2 runner (superseded);
  [ADR-006](decisions/ADR-006-independent-runtime.md) independent runtime;
  [ADR-007](decisions/ADR-007-ttft-and-cache-capacities.md) TTFT gate and separate capacities.
