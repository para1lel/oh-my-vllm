# Architecture — oh-my-vllm

_Last reviewed: 2026-09-19. HEAD at review: `0301c28` plus two unstaged changes._

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
│     "shutdown"  → break
│
└── OhMyVllmWorker  (wraps vllm.v1.worker.gpu_worker.GPUWorker)
      _to_vllm_scheduler_output(): adapts Rust SchedulerOutput → vLLM type
      execute_model(): calls GPUWorker.execute_model(), reads sampled_token_ids
      MTP branch: parse_mtp_output() when num_speculative_tokens > 0
```

### Key data flow (one step)

1. `Scheduler.schedule()` → `SchedulerOutput` (scheduled requests, block tables)
2. `WorkerClient.execute_one_step()` → msgpack `ExecuteMsg` over ZMQ
3. Python decodes, calls `GPUWorker.execute_model(VllmSchedulerOutput)`
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
| `python/oh_my_vllm/worker/spec_decode.py` | Python | `build_speculative_config`, `parse_mtp_output` | MTP config and output parsing |

---

## 3. Qwen3.5 hybrid-specific invariants

These are not configurable for the current target model:

- `block_size = 784` — used throughout the KV cache and asserted at construction
- Mamba group uses `mamba_cache_mode="align"`: at most one live block per request
  (the checkpoint after the most recent 784-token boundary)
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
The IPC file must not exist when the binary starts — clean it up between runs.

---

## 5. Not yet implemented

- HTTP/gRPC server (the binary is a direct CLI driver, not a serving endpoint)
- Swap-based preemption (CPU KV offload) — explicitly deferred (REQ-OUT-SCOPE-001)
- Multi-GPU / tensor parallel > 1
- LoRA, multimodal

---

## 6. Proposed / pending (not accepted)

None outstanding at handoff time.

## Installed GPUWorker adaptation (2026-09-19)

ADR-002 supersedes the old API names and assumed physical group ordering above.
The adapter uses Worker, CachedRequestData, execute_model then sample_tokens,
and preserves request IDs and every returned token. Rust sends completion-only
steps. Logical FA/Mamba tables map into the worker's actual three Mamba groups
plus one FA group using disjoint physical block IDs (ADR-002). Rust chunk boundaries preserve Mamba checkpoint alignment.
