# Plan

## Completed

1. **2026-09-19 — Adapter.** Rust scheduling and logical KV cache, with an
   adapter to the installed vLLM GPUWorker (ADR-001–003). All 9 EngineCore rows
   passed.
2. **2026-09-21 — Serving and V2 runner.**
   - OpenAI-compatible Chat/Responses serving, with real MTP4 and oh-my-pi
     acceptance (ADR-004).
   - V2 Model Runner (ADR-005).
3. **2026-09-21 — Independent runtime.** Project-owned model, state and kernels
   built on independent libraries. vLLM is removed from builds, tests and
   inference (ADR-006).
4. **2026-09-22 — TTFT, long context and TileLang.**
   - Refreshed and froze the official vLLM baseline.
   - Added a per-request TTFT gate and separate FA/GDN capacities (ADR-007).
   - Added 262144-token context support.
   - Migrated all owned Triton kernels to TileLang.
5. **2026-09-22/23 — CUDA migration.**
   - Migrated all owned kernels to B200 CUDA C++ and inline PTX, keeping a
     frozen TileLang comparison.
   - All 147 operator cases and all 12 framework rows pass.
   - CUDA is now the default (REQ-KERNEL-002).

Evidence for each stage is indexed in [acceptance.md](acceptance.md).

## Next

Fix the findings in the [2026-09-23 code audit](audit-2026-09-23.md), batch by
batch:

1. Serving availability
2. Evidence integrity
3. Kernel and worker contracts
4. Scheduler hardening
5. Measured performance
6. Cleanup

Every batch must keep the 12-row performance gates, the numerical tolerances,
block size 784 and the Rust/Python ownership split.

## Not planned

- **Out of scope:** CPU KV swap, PD/multi-node, LoRA and multimodal execution.
- **Documentation only (ADR-006):** future model architectures, other NVIDIA
  backends, single-node multi-GPU and the local DSpark draft model. They are
  described there without placeholder interfaces.
