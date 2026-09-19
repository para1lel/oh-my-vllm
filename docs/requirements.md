# Requirements — oh-my-vllm

_Confirmed status: user-confirmed unless noted. Last reviewed: 2026-09-19._

---

## REQ-GOAL-001 — Primary objective

**Status:** user-confirmed
**Priority:** must-have

Build a Rust-first inference framework for Qwen3.5-27B-FP8 on a single B200 GPU
that reaches at least 95% of vLLM EngineCore throughput on the benchmark workloads
below. Rust owns the scheduler and KV cache; Python wraps vLLM's `GPUWorker`.

**Acceptance:** `benchmarks/compare_vllm.py` records passed=true for every required mode/workload row.

---

## REQ-ARCH-001 — Rust/Python split

**Status:** user-confirmed
**Priority:** must-have

Rust owns: request queue, token budget, KV block allocation, prefix cache, preemption.
Python owns: model loading, forward pass, attention kernels, CUDAGraph capture.
The boundary is a single ZMQ DEALER socket; no shared memory, no Tensor data crossing.

**Constraint:** Do not move scheduling logic into Python. Do not call vLLM's Python
scheduler. The split is the design, not an implementation convenience.

---

## REQ-MODEL-001 — Target model

**Status:** user-confirmed
**Priority:** must-have

Model: `/data0/shared/Qwen3.8-27B-FP8` (Qwen3.5-27B, FP8 quantised)
Architecture: 48 GatedDeltaNet (Mamba) layers + 16 full-attention layers
Block size: **784 tokens** — this is a hard constraint from the hybrid architecture.
KV groups: FA group (`group_id=0`, 16 layers) + Mamba group (`group_id=1`, 48 layers,
`mamba_cache_mode="align"`).

**Must not change:** block_size. Any code that parameterises block_size for other
values is fine, but the default and all production paths use 784.

---

## REQ-PERF-001 — Throughput target

**Status:** user-confirmed
**Priority:** must-have

| bs | input tokens | output tokens | metric | target |
|---|---|---|---|---|
| 1 | 32768 | 4096 | output tok/s | >=95% of matched vLLM |
| 2 | 32768 | 4096 | output tok/s | >=95% of matched vLLM |
| 4 | 32768 | 4096 | output tok/s | >=95% of matched vLLM |

Run each row in ordinary decoding, MTP, and controlled prefix-hit modes against
matching vLLM modes. Identical token inputs, fixed output counts, sampling,
execution settings, memory budgets and timing boundaries are required. Exclude
loading, compilation and warmup; include scheduling and transport. Report at
least three measurements and their median, rerunning if variance is material.
Faster than 105% is a pass. All nine required rows passed; raw measurements and configuration are linked in acceptance.md.

---

## REQ-FUNC-001 — Chunked prefill

**Status:** user-confirmed; **implemented** (code-observed)

Long prompts are processed in chunks bounded by `max_num_batched_tokens=32768`.
The scheduler applies the token budget and an explicit aligned_prefill split to
materialize Mamba checkpoints at block boundaries.

---

## REQ-FUNC-002 — Continuous batching

**Status:** user-confirmed; **implemented** (code-observed)

Decode-step requests are scheduled ahead of new prefills within the same step.
See `crates/scheduler/src/lib.rs` phase-1 (running queue) / phase-2 (waiting queue).

---

## REQ-FUNC-003 — Prefix caching

**Status:** user-confirmed; **implemented** (code-observed)

Chain-hash prefix cache (SHA-256 truncated to 64 bits) with LRU eviction.
Coordinated across both KV groups. See `crates/kv-cache/src/`.

---

## REQ-FUNC-004 — Preemption by recompute

**Status:** user-confirmed; **implemented** (code-observed)

When the block pool is exhausted during scheduling, the last-admitted running
request is evicted back to the waiting queue with `num_computed_tokens=0`.
See the schedule/update lifecycle in `crates/scheduler/src/lib.rs`.

**Explicitly deferred:** swap-based preemption (CPU KV offload). Not planned for
the current scope because it requires Python-side CPU tensor management.

---

## REQ-FUNC-005 — MTP speculative decoding

**Status:** user-confirmed; implemented and exercised end-to-end.

EngineArgs enables MTP when num_speculative_tokens>0. The actual Worker returns
all sampled/accepted output tokens and next drafts via take_draft_token_ids.
Rust schedules drafts, reserves target states, and rolls back scheduled rejections.
ADR003 records BF16 SSM to retain block784. Coherent MTP text, actual-path FP64
and feature combinations have passed; MTP performance passed at batch sizes 1/2/4.

---

## REQ-ACC-001 — Actual-path GQA/GDN accuracy

Compare the actual inference GQA and GDN kernels against independent CPU FP64
references, covering prefill, decode, chunk boundaries and recurrent state.
Document dtype-dependent tolerances. Existing standalone GQA results do not
establish actual-path coverage. Also demonstrate coherent real-text inference,
including MTP and prefix hits. Full-model token equality is not an acceptance gate.

## REQ-OBS-001 — Timestamped diagnostic logging

Both processes need UTC timestamps, monotonic durations, run/request/step
correlation, and configurable levels. Default logging must avoid per-step I/O;
debug mode exposes scheduler, KV, transport and worker timing. Host execution
time must not be presented as CUDA kernel time. Profiling is opt-in.

---

## REQ-OUT-SCOPE-001 — Explicitly excluded

**Status:** user-confirmed

- Multimodal inputs (text-only for the current scope; architecture leaves hooks)
- Swap-based preemption (CPU KV offload)
- Multi-GPU / tensor parallel > 1
- HTTP/gRPC server (binary currently drives inference directly via ZMQ)
- LoRA adapters
- Production deployment or containerisation
