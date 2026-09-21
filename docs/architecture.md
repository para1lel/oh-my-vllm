# Architecture — oh-my-vllm

_Updated: 2026-09-21. See handoff.md for verification evidence._

---

## 1. Implemented architecture (code-observed)

```
oh-my-vllm-zmq-worker (Rust binary)
│
├── Scheduler  (crates/scheduler/)
│     FCFS waiting queue → running queue
│     Per-step token budget  →  chunked prefill naturally
│     Phase 1: running requests (decode / continued prefill)
│     Phase 2: waiting requests (new prefills, prefix cache lookup)
│     Preemption by recompute when pool exhausted
│     MTP rollback: corrects num_computed_tokens for rejected drafts
│
├── HybridCoordinator  (crates/kv-cache/)
│     BlockPool: flat Vec<Block> + hand-written doubly-linked LRU free queue
│     chain-hash prefix cache: SHA-256[0..8] per block, parent-chained
│     Two GroupManagers: FA (group 0, 16 layers) + Mamba (group 1, 48 layers)
│     Two-phase allocation sequence (vLLM issue #33775 ordering)
│
└── WorkerClient  (crates/zmq-worker/src/client.rs)
      DealerSocket bound at ipc:///tmp/oh-my-vllm.ipc
      Retry loop: polls send until Python peer connects
      Msgpack-encoded protocol (see docs/design.md for message schema)
      Spawns Python subprocess: python -m oh_my_vllm.worker.zmq_bridge

                  ZMQ DEALER ↕ msgpack

Python process (oh_my_vllm.worker.zmq_bridge)
│
├── zmq_bridge.serve()  — event loop: recv → dispatch → send
│     "init"      → _handle_init() → builds VllmConfig, inits GPUWorker
│     "register"  → worker.register_request()
│     "execute"   → worker.execute_model()  → "execute_result"
│     "abort"     → worker.unregister_request()
│     "shutdown"  → worker shutdown + distributed cleanup
│
└── OhMyVllmWorker  (wraps vllm.v1.worker.gpu_worker.Worker)
      SchedulerAdapter.convert(): adapts Rust SchedulerOutput → vLLM type
      execute_model(): calls Worker.execute_model(), then sample_tokens() if needed
      MTP: schedule actual drafts, take_draft_token_ids(), return all accepted tokens
```

### Key data flow (one step)

1. `Scheduler.schedule()` → `SchedulerOutput` (scheduled requests, block tables)
2. `WorkerClient.execute_one_step()` → msgpack `ExecuteMsg` over ZMQ
3. Python decodes, calls `Worker.execute_model`, then `sample_tokens` if needed
4. Python encodes `ExecuteResultMsg` with per-request next tokens
5. Rust decodes, calls `Scheduler.update(WorkerOutput)`
6. Finished requests freed; MTP draft tokens rolled back if rejected

---

## 2. Module responsibilities and entry points

| Module | Language | Entry | Owns |
|---|---|---|---|
| `crates/kv-cache/` | Rust | `HybridCoordinator::new` | Block pool, prefix cache, two-phase allocation |
| `crates/scheduler/` | Rust | `Scheduler::new` | Request queues, token budget, preemption |
| `crates/zmq-worker/` | Rust binary | `main()` in `src/main.rs` | CLI, ZMQ client, inference loop |
| `python/oh_my_vllm/worker/zmq_bridge.py` | Python | `serve()` | ZMQ server, message dispatch |
| `python/oh_my_vllm/worker/model_runner.py` | Python | `OhMyVllmWorker` | vLLM GPUWorker wrapper, output adaptation |
| `python/oh_my_vllm/worker/v2_runner.py` | Python | `RustDraftTokensHandler` | Real V2 draft IDs transferred to Rust |

---

## 3. Qwen3.5 hybrid-specific invariants

These are not configurable for the current target model:

- `block_size = 784` — enforced at production CLI/worker boundaries; generic unit tests use smaller blocks
- Mamba group uses `mamba_cache_mode="align"`: running/checkpoint state plus
  temporary protected previous state and K speculative state slots when MTP is enabled.
  Slots migrate across chunk boundaries; null placeholders preserve logical positions.
- `is_simple_hybrid = True`: prefix cache reconciliation converges in one iteration
  (FA hit first, Mamba hit bounded by FA hit, combined = `min(fa_len, mb_len)`)
- Full prefix hit still requires recomputing the last token for logits
  (`max_hit = num_tokens - 1` in `HybridCoordinator.get_computed_blocks`)

---

## 4. ZMQ topology

Both sides use `DEALER` sockets. Rust binds; Python connects. This is valid for
a 1:1 topology and was chosen because zeromq 0.6.0 (the crate in use) does not
expose a `PairSocket` type. See `docs/decisions/ADR-001-zmq-socket-type.md`.

Socket path: `ipc:///tmp/oh-my-vllm.ipc` (default; overridable via `--socket`).
Use unique IPC paths for concurrent runs; remove a stale socket only after its owner exits.

---

## 5. Not yet implemented

- gRPC server — outside scope
- Swap-based preemption (CPU KV offload) — explicitly deferred (REQ-OUT-SCOPE-001)
- Multi-GPU / tensor parallel > 1
- LoRA, multimodal

---

## 6. Rust OpenAI serving

`serve` in crates/zmq-worker/src/serving uses Axum for both OpenAI adapters,
model discovery and stored Responses endpoints. A bounded channel feeds one Rust
engine owner between inference steps; it owns online admission, response delivery,
request deadlines, disconnect cancellation and the existing Scheduler/KV objects.
A bounded, expiring in-memory store retains Responses snapshots. Each API has its
own JSON/SSE representation backed by a shared output/parser representation.

A `prepare` ZMQ RPC lets Python tokenize/template and validate per-request sampling
and schemas before Rust admission. Python serving.py uses the installed XGrammar
compiler and keeps per-request grammar/detokenization state. Speculative masks are
built for each draft prefix plus the bonus position, rolled back after simulation,
and passed to GPUWorker.sample_tokens. Only accepted tokens advance persistent
grammar state. No masks/tensors cross ZMQ and no vLLM scheduler is called.

Execute replies optionally include incremental text, finish reason and reasoning
counts. Rust parses model reasoning/XML calls and routes protocol events, then
releases finished/cancelled requests through the existing finished-only lifecycle.
OMP executes tools locally and sends results back. See serving.md for the exact
compatibility limits; acceptance.md records the completed real MTP4 verification.

## Python side: V2 GPUWorker adaptation (2026-09-21)

ADR-002 records the installed API and physical group mapping.
The adapter uses Worker, CachedRequestData, execute_model then sample_tokens,
and preserves request IDs and every returned token. Rust sends completion-only
steps. Logical FA/Mamba tables map into the worker's actual three Mamba groups
plus one FA group using disjoint physical block IDs (ADR-002). Rust chunk boundaries preserve Mamba checkpoint alignment.

Only `vllm.v1.worker.gpu.model_runner.GPUModelRunner` is supported. Initialization
sets `VLLM_USE_V2_MODEL_RUNNER=1`, rejects an explicit conflicting setting and checks
the actual runner class. There is no V1 fallback and no change to vLLM source.
ADR-005 describes the V2 boundary and project-owned draft-transfer adaptation.

Rust sends `prefill_token_ids` containing all accepted history on first admission
and recompute resumption, including cached prefixes and previous accepted output.
Original prompt length remains separate. Unverified drafts never enter this
history. Python removes preempted worker state and resumes via `NewRequestData`;
ordinary running updates omit history and append only block-table deltas.
The worker output count is restored from history length minus original prompt
length, preserving draft positions and output limits across resumption.

V2 normally returns placeholder draft IDs for unconstrained requests. Our local
`RustDraftTokensHandler` reuses its asynchronous D2H/event path for every MTP batch,
so the unsigned Rust protocol always receives actual draft IDs. A copied batch
flag controls transfer only; sampling and grammar flags remain unchanged.

Hybrid cache zeroing is mandatory in V2: the Rust block pool reports genuinely
new allocations, including recycled pages, and Python maps each to its reserved
physical stride for V2's pre-execution zeroer. Prefix hits and live state slots
are excluded. Startup clears dummy warmup cache contents in place before requests
arrive. These prevent stale mixed-layout bytes from becoming NaN on later reads.

## MTP and benchmark lifecycle

MTP reserves K extra Mamba state slots for target verification. Rejection rolls
back only drafts actually scheduled. Worker admission, prefix hits and resumed
requests remain distinct. ADR-003 documents BF16 SSM storage in MTP mode.

Bench uses one persistent worker, unique request IDs, cold cache reset before
warmup and measured iterations, and optional explicit prefix seeding outside the
timer. Arrival interval admits requests while existing requests decode. Reduced
scheduler-blocks exercises recompute preemption without altering physical cache.
The timer includes registration through completion notification. Results include
cache hits, preemptions and generated/accepted draft counts.

## Independent runtime migration — staged implementation

ADR-006 replaces the Python GPUWorker dependency while preserving Rust ownership
and the ZMQ contract. The current service still uses the V2 adapter above. New
`kernels/` modules implement explicit recurrent state snapshots, causal
convolution, normalization and rotary embedding, with independent FlashInfer
operators for FP8 GEMM, paged attention and chunked GDN prefill. `models/qwen.py`
loads the concrete checkpoint directly from safetensors and composes these
operators; it is currently an eager diagnostic path, not the production runner.

Service preparation/detokenization now uses Hugging Face Tokenizers and XGrammar
directly, with framework-owned sampling configuration. The transitional adapter
converts that configuration at its vLLM boundary. The standalone target sampler
applies penalties and grammar masks per draft prefix, accepts deterministic
drafts until the first target-sample mismatch, and commits only retained tokens.
Production cache planning, MTP execution and graph integration are still pending.
