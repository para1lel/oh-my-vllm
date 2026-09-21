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
| `python/oh_my_vllm/worker/spec_decode.py` | Python | Legacy helpers | Not used by the current inference path |

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

- OpenAI-compatible HTTP server — accepted future scope, not implemented
- gRPC server — outside scope
- Swap-based preemption (CPU KV offload) — explicitly deferred (REQ-OUT-SCOPE-001)
- Multi-GPU / tensor parallel > 1
- LoRA, multimodal

---

## 6. Accepted serving direction (not implemented)

Both Chat Completions and Responses will feed the existing Rust-owned inference
engine. Prefer Rust for HTTP, protocol adaptation, response storage and request
lifecycle. Python continues to wrap GPUWorker; exact placement of tokenizer,
template and model-specific parsing is pending. A shared internal generation
representation is recommended, not an implemented interface.

Responses supports full-history input and bounded, expiring in-memory stored
responses; restart persistence is not required. oh-my-pi executes tools locally
and sends their results back to the service. See [serving.md](serving.md) for the
confirmed scope, observed gaps and unresolved details, and
[ADR-004](decisions/ADR-004-openai-serving.md) for the accepted direction.

## Installed GPUWorker adaptation (2026-09-19)

ADR-002 records the installed API and physical group mapping.
The adapter uses Worker, CachedRequestData, execute_model then sample_tokens,
and preserves request IDs and every returned token. Rust sends completion-only
steps. Logical FA/Mamba tables map into the worker's actual three Mamba groups
plus one FA group using disjoint physical block IDs (ADR-002). Rust chunk boundaries preserve Mamba checkpoint alignment.

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
