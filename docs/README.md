# Documentation index — oh-my-vllm

## Quick reading paths

**First time on the project** → `../README.md` then `architecture.md` then `development.md`

**Picking up an in-progress task** → `handoff.md` first, then `plan.md`

**Fixing a Python/ZMQ runtime error** → `development.md#troubleshooting` and `handoff.md#known-issues`

**Understanding a design decision** → `decisions/`

**Adding a feature or fixing a bug** → `../CONTRIBUTING.md` for process, `requirements.md` for scope, `testing.md` for what to verify

---

## Documents

| File | What it covers |
|---|---|
| `../README.md` | Project overview, architecture diagram, quick start, performance target |
| `architecture.md` | Implemented module structure, data flow, Qwen3.5 invariants, ZMQ topology |
| `requirements.md` | REQ-* IDs with acceptance criteria and current status |
| `development.md` | Conda env setup, build/lint/test commands, troubleshooting |
| `testing.md` | Test layers, current verification status per requirement, change-triggered matrix |
| `plan.md` | TASK-* IDs, phase status, what is done vs pending |
| `handoff.md` | Working tree state, open blockers, known traps, next approved task |
| `design.md` | ZMQ protocol schema, KV cache data structures, scheduler logic, MTP integration |
| `profiling.md` | Flamegraph + torch profiler workflow, gap analysis guide |
| `decisions/ADR-001-zmq-socket-type.md` | Why DEALER instead of PAIR for the ZMQ channel |
| `../CONTRIBUTING.md` | Commit conventions, definition of done, what is out of scope |
| `../AGENTS.md` | Rules for AI agents working on this repo |
| `../CLAUDE.md` | Claude-specific additions to AGENTS.md |
| `../benchmarks/compare_vllm.py` | Script to measure throughput vs vLLM baseline |

---

## Status at last update (2026-09-19)

- Rust unit tests: 55 passing
- GQA accuracy: 3/3 cases verified
- End-to-end smoke test: **not yet verified** (fixes committed, awaiting run)
- Throughput benchmark: **not yet run**

See `handoff.md` for the current blocker and next step.
