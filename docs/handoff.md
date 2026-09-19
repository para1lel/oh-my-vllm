# Handoff — oh-my-vllm

_Snapshot time: 2026-09-19. Code baseline: HEAD=`0301c28` plus two unstaged changes (see below)._

---

## Working tree state

```
 M crates/zmq-worker/src/client.rs        (unstaged)
 M python/oh_my_vllm/worker/zmq_bridge.py (unstaged)
```

These two files contain fixes that are **required for the smoke test to pass** and
must be committed before handoff is complete. They are not yet committed because
the smoke test was not yet verified when this documentation was written.

---

## Current task state

### TASK-E2E-001 — End-to-end smoke test (Phase 4 gate)

**Status:** in progress, blocked on vllm API mismatch (now fixed, pending verification)

The smoke test (`oh-my-vllm-zmq-worker run --tokens 1 2 3 4 5 --max-tokens 16`)
failed in the last session with two `pydantic.ValidationError` errors in `zmq_bridge.py`:

1. `CacheConfig(swap_space=0)` — `swap_space` is not a valid parameter in the
   installed vllm version. **Fixed** in the unstaged `zmq_bridge.py`:
   removed `swap_space`, added `mamba_cache_mode="align"`.

2. `SchedulerConfig(max_num_batched_tokens=32768, max_num_seqs=256, max_model_len=...)`
   — this vllm version's `SchedulerConfig` requires `is_encoder_decoder` as a
   mandatory `InitVar`. **Fixed** in the unstaged `zmq_bridge.py`:
   added `is_encoder_decoder=False` and reordered to match the signature.

**To verify:**
```bash
git add python/oh_my_vllm/worker/zmq_bridge.py crates/zmq-worker/src/client.rs
# commit first (see CONTRIBUTING.md for message format)
rm -f /tmp/oh-my-vllm.ipc
export PATH="/data0/shared/dongwu.chen/conda-envs/vllm/bin:$PATH"
export PYTHONPATH="/data0/shared/dongwu.chen/oh-my-vllm/python:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
./target/release/oh-my-vllm-zmq-worker \
    --model /data0/shared/Qwen3.8-27B-FP8 \
    --num-gpu-blocks 2048 --block-size 784 --max-model-len 8192 \
    run --tokens 1 2 3 4 5 --max-tokens 16
# Expected: "output token ids: [<16 integers>]"
```

**Acceptance condition (REQ-GOAL-001):** binary exits 0 and prints `output token ids:`.

---

### TASK-BENCH-001 — Throughput benchmark (Phase 4 gate)

**Status:** not started. Blocked on TASK-E2E-001.

Run after smoke test passes:
```bash
python benchmarks/compare_vllm.py \
    --model /data0/shared/Qwen3.8-27B-FP8 \
    --num-gpu-blocks 4096 \
    --batch-sizes 1 2 4 \
    --input-len 32768 --output-len 4096
```

**Acceptance condition (REQ-PERF-001):** all rows print ✓ (ratio within ±5% of vLLM).

---

## What has been verified

| Item | Evidence |
|---|---|
| Rust unit tests pass (55 total) | Pre-commit hook runs `cargo test --all` on every commit |
| GQA accuracy (atol=1e-2) | Commit `7131da9`, 3/3 cases, max_err=8.4e-4 |
| Release binary builds clean | Verified 2026-09-19 with cargo 1.97 |
| Python lint passes | ruff format + ruff check clean on every commit |

---

## What has NOT been verified

- End-to-end model forward pass (TASK-E2E-001)
- ±5% throughput target (TASK-BENCH-001)
- MTP speculative decoding path end-to-end
- Continuous batching with two real concurrent requests

---

## Known issues / traps for the next developer

**Q-001 — vllm CacheConfig/SchedulerConfig API stability**
- Impact: zmq_bridge.py may break again if the vllm conda env is updated
- Blocking: yes for smoke test
- Safe boundary: check the actual signatures with `python -c "import inspect; from vllm.config import CacheConfig; print(inspect.signature(CacheConfig))"` before running
- Fix approach: keep `_handle_init()` defensive; match only fields present in the running vllm

**Q-002 — ZMQ "Address already in use" after failed run**
- Impact: next run will fail immediately with EADDRINUSE
- Blocking: only if not cleaned
- Fix: `rm -f /tmp/oh-my-vllm.ipc` before every run

**Q-003 — num_gpu_blocks must be set correctly for the model**
- Impact: too low → OOM during cache init; too high → vllm rejects the override
- Not verified: exact safe value for `/data0/shared/Qwen3.8-27B-FP8` on B200 with 176 GB
- Suggestion: start with `--num-gpu-blocks 2048` for smoke test, higher for benchmarks

**Temporary vs. final design:**
- `_to_vllm_scheduler_output()` in `model_runner.py` constructs `NewRequestData` and
  `ResumedRequestData` using field names from vLLM's `SchedulerOutput`. These are
  vLLM internals and may change. This is a known brittleness point.
- The Rust binary currently drives inference synchronously (send one step, wait for
  result). A future async overlap of scheduling and execution would improve throughput
  but is not planned.

---

## Next approved task

**TASK-E2E-001** — commit the two unstaged fixes, run smoke test, verify exit 0.
Then **TASK-BENCH-001** — run `benchmarks/compare_vllm.py`, confirm ±5%.
Both are required to close Phase 4.
