# Requirements — oh-my-vllm

_Confirmed status: user-confirmed unless noted. Last reviewed: 2026-09-21._

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

## REQ-RUNNER-002 — Independent runtime (active migration)

Replace the vLLM implementation dependency with code maintained in this repository
and independent libraries. Preserve Rust serving/scheduling/logical KV and Python
GPU execution. Final builds, tests and services must neither install/import/link
vLLM nor depend on its source, conda environment or cached compiled artifacts.
Transitional milestones may retain the V2 adapter described below.

Use recent compatible stable dependencies selected in dependency order, with the
verified versions pinned. Higher-level dependencies live in conda oh-my-vllm;
working system CUDA/compiler tools are allowed. Selected code may be ported with
provenance and licensing, but copying the framework wholesale is not acceptable.

Keep all existing functionality and FP64 tolerances. Run targeted regressions at
each milestone and all nine original-baseline >=95% checks at final acceptance.
Do not rerun native vLLM or require a new V2-relative threshold. Future model
architectures, single-node multi-GPU, other NVIDIA GPUs and local DSpark need brief
extension documentation only. See ADR-006 for the user-confirmed boundaries.

## REQ-RUNNER-001 — V2 Model Runner (completed predecessor)

Use GPUWorker's V2 Model Runner exclusively, with no legacy implementation or
fallback. Preserve all existing offline and serving features, including MTP4
with constrained decoding, prefix caching, recompute preemption and cancellation.
Do not modify vLLM source; implement required adaptations in this project.
Rust owns accepted history and may send its complete contents on admission and
resumption, while normal decode remains incremental and drafts remain separate.

For this migration, reuse the frozen nine-row EngineCore baseline in
bench/baseline/2026-09-19-acceptance.json. Do not rerun vLLM baseline. Retain the
95% performance gate, existing FP64 tolerances and HTTP observability checks.
Record current editable vLLM revision separately from its package version and
the historical baseline identity. See ADR-005.

## REQ-OBS-001 — Timestamped diagnostic logging

Both processes need UTC timestamps, monotonic durations, run/request/step
correlation, and configurable levels. Default logging must avoid per-step I/O;
debug mode exposes scheduler, KV, transport and worker timing. Host execution
time must not be presented as CUDA kernel time. Profiling is opt-in.

---

## REQ-SERVE-001 — OpenAI-compatible local service

**Status:** user-confirmed 2026-09-21; implemented; real MTP4 acceptance passed
**Priority:** must-have for the serving extension

Support both Chat Completions and Responses, with streaming and non-streaming
responses, model discovery, function tool calls, tool-result history, usage,
termination and error semantics. Both the service and oh-my-pi run on the current
machine. Prefer Rust for HTTP entry points, protocol adaptation and request
lifecycle; preserve Rust scheduler/KV ownership and the existing GPUWorker split.
This extends the former HTTP exclusion; it does not replace the EngineCore
performance requirements or claim HTTP-level performance acceptance.

Responses must support full-history requests and stored responses with
`previous_response_id`, retrieval and deletion. Use bounded, expiring in-memory
storage; continuation across service restarts is not required. Defaults are
store=true, one-hour TTL, 1,000 records and 256 MiB serialized response/history
payload; see serving.md for configurable limits, errors and compatibility.

## REQ-SERVE-002 — Configurable thinking strength

**Status:** user-confirmed 2026-09-21; implemented; real MTP4 acceptance passed

Expose off/low/medium/high/xhigh with default medium. high aliases native xhigh;
OpenAI none aliases off. Use the native template and separate reasoning from text
and tool arguments. OMP mappings, replay and validated template extensions are
specified in serving.md. These are prompt controls, not hard thinking budgets.

## REQ-SERVE-003 — Real oh-my-pi acceptance on both APIs

**Status:** user-confirmed 2026-09-21; both real MTP4 agentic tasks passed

Run oh-my-pi in this repository against each API separately. Ask it to read README,
architecture documentation and necessary source, then describe project goals,
architecture, operation and current completion status with file references.
Require actual tool calls, tool results returned to the model, and a grounded final
answer; do not modify repository files during that task. Tools execute in oh-my-pi.
Test both APIs, plus non-streaming, cancellation, errors and thinking mapping.
Detailed planned coverage is in [serving.md](serving.md).

The implementation request supersedes the earlier documentation-only discussion.
Both real agentic runs must enable MTP. Scripted worker/client protocol tests do
not satisfy real model acceptance. The earlier host CUDA/NVML fault recovered;
real evidence and limitations are recorded in acceptance.md and handoff.md.

## REQ-SERVE-004 — Constrained decoding with MTP

Support JSON object, JSON Schema and strict function arguments using token-level
masks, including the MTP verification path. No silent non-MTP fallback and no
post-generation-only validation substitute. Reject unsupported schemas explicitly.
Support tool choice auto/none/required/named and parallel_tool_calls=false.
Supported schema and XML encoding limits are specified in serving.md.

## REQ-SERVE-005 — Serving observability

No additional strict HTTP throughput gate. Inspect actual queue/preparation time,
TTFT, output rate, steps/cache behavior and MTP proposed/accepted counters. Diagnose
abnormal logs and resource retention. Do not use old EngineCore results as serving
performance evidence. Existing REQ-PERF-001 remains unchanged.

---

## REQ-OUT-SCOPE-001 — Explicitly excluded

**Status:** user-confirmed

- Multimodal inputs (text-only for the current scope; architecture leaves hooks)
- Swap-based preemption (CPU KV offload)
- Multi-GPU / tensor parallel > 1
- gRPC server (OpenAI-compatible HTTP is in scope, REQ-SERVE-001)
- LoRA adapters
- Production deployment or containerisation
