# Requirements — oh-my-vllm

_Confirmed status: user-confirmed unless noted. Last reviewed: 2026-09-19._

---

## REQ-GOAL-001 — Primary objective

**Status:** user-confirmed
**Priority:** must-have

Build a Rust-first inference framework for Qwen3.5-27B-FP8 on a single B200 GPU
that reaches within ±5% of vLLM EngineCore throughput on the benchmark workloads
below. Rust owns the scheduler and KV cache; Python wraps vLLM's `GPUWorker`.

**Acceptance:** `benchmarks/compare_vllm.py` prints ✓ for every workload row.

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

| Workload | bs | input tokens | output tokens | metric | target |
|---|---|---|---|---|---|
| prefill-heavy | 1 | 32768 | 128 | TTFT | ±5% of vLLM |
| decode-heavy | 4 | 2048 | 4096 | output tok/s | ±5% of vLLM |
| mixed | 2 | 32768 | 4096 | output tok/s | ±5% of vLLM |

**Not yet verified** — smoke test and benchmark runs were initiated in the last
session but blocked by Python config errors (see `docs/handoff.md`).

---

## REQ-FUNC-001 — Chunked prefill

**Status:** user-confirmed; **implemented** (code-observed)

Long prompts are processed in chunks bounded by `max_num_batched_tokens=32768`.
This is a natural consequence of `to_schedule = remaining.min(token_budget)` in
`crates/scheduler/src/lib.rs`; no dedicated branch is required.

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
See `crates/scheduler/src/lib.rs:160-172`.

**Explicitly deferred:** swap-based preemption (CPU KV offload). Not planned for
the current scope because it requires Python-side CPU tensor management.

---

## REQ-FUNC-005 — MTP speculative decoding

**Status:** user-confirmed; **implemented but not end-to-end verified**

Activates via `SpeculativeConfig` passed to `VllmConfig` when
`num_speculative_tokens > 0`. Uses vLLM's `Step3p5MTPProposer` internally.
`parse_mtp_output` in `python/oh_my_vllm/worker/spec_decode.py` extracts accepted
draft counts from `sampled_token_ids`. Rust scheduler handles rollback.

**Not yet verified:** MTP path has not been exercised end-to-end against the model.

---

## REQ-ACC-001 — Single-layer GQA accuracy

**Status:** user-confirmed; **verified** ✓

GPU FP16 GQA output must be within atol=1e-2, rtol=1e-2 of a CPU FP64 reference.
Verified in `tests/test_gqa_accuracy.py`; 3/3 cases pass, max_err=8.4e-4.

---

## REQ-OUT-SCOPE-001 — Explicitly excluded

**Status:** user-confirmed

- Multimodal inputs (text-only for the current scope; architecture leaves hooks)
- Swap-based preemption (CPU KV offload)
- Multi-GPU / tensor parallel > 1
- HTTP/gRPC server (binary currently drives inference directly via ZMQ)
- LoRA adapters
- Production deployment or containerisation
