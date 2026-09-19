# Implementation Plan — oh-my-vllm

_Phases 1–2 complete. Phase 3 mostly complete. Phase 4 in progress. Phase 5 docs done._

---

## Phase 1 — Minimal viable prototype ✓

**Status:** complete (commits `7b0fbbb`, `ed30ef9`, `4a2e520`, `60b8e81`)

| TASK | Description | Status |
|---|---|---|
| TASK-1-01 | Rust workspace + crate structure | ✓ done |
| TASK-1-02 | kv-cache crate: BlockPool, chain-hash, HybridCoordinator | ✓ done, 47 tests |
| TASK-1-03 | scheduler crate: FCFS, chunked prefill, continuous batching | ✓ done, 8 tests |
| TASK-1-04 | Python OhMyVllmWorker wrapper around vllm GPUWorker | ✓ done |
| TASK-1-05 | ZMQ bridge (Python DEALER server) | ✓ done |

---

## Phase 2 — KV cache + continuous batching ✓

**Status:** complete, included in Phase 1 milestones above

Prefix cache reconciliation for the Qwen3.5 hybrid (is_simple_hybrid=True),
two-phase block allocation (vLLM issue #33775 ordering), watermark enforcement.

---

## Phase 3 — Preemption + MTP ✓ (code complete, E2E unverified)

**Status:** code complete; end-to-end not verified

| TASK | Description | Status |
|---|---|---|
| TASK-3-01 | Preemption by recompute in Rust scheduler | ✓ done (`f60bbc5`) |
| TASK-3-02 | MTP spec decode Python helpers | ✓ done (`658e78c`) |
| TASK-3-03 | Swap-based preemption (CPU offload) | **explicitly deferred** (REQ-OUT-SCOPE-001) |

---

## Phase 4 — Performance alignment ✓/⏳

**Status:** partially complete

| TASK | Description | Status |
|---|---|---|
| TASK-4-01 | GQA accuracy test | ✓ verified (`7131da9`) |
| TASK-4-02 | Release binary builds clean | ✓ verified |
| TASK-E2E-001 | End-to-end smoke test | ⏳ fixes on disk, not committed or verified |
| TASK-BENCH-001 | >=95% throughput benchmark vs vLLM | ⏳ not started |

---

## Phase 5 — Docs + CI

**Status:** docs complete; CI not implemented

| TASK | Description | Status |
|---|---|---|
| TASK-5-01 | README.md | ✓ done (`0301c28`) |
| TASK-5-02 | docs/design.md, docs/profiling.md | ✓ done (`0301c28`) |
| TASK-5-03 | benchmarks/compare_vllm.py | ✓ done (`0301c28`) |
| TASK-5-04 | AGENTS.md, CLAUDE.md, CONTRIBUTING.md, docs/ suite | ✓ done (this commit) |
| TASK-5-05 | CI (GitHub Actions or similar) | **not started** — low priority until E2E verified |

---

## Scope boundary

Out of scope for all phases (REQ-OUT-SCOPE-001):
- Swap-based preemption
- Multi-GPU
- HTTP/gRPC server
- LoRA, multimodal

These must not be added without explicit user confirmation.

## Takeover milestones (2026-09-19)

Environment/logging foundation is under independent review. Next: actual-path
accuracy and E2E, full feature/combination testing, then fair performance
acceptance. Each milestone requires sub-agent review and fixes before commit.
