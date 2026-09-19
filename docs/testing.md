# Testing Guide — oh-my-vllm

## Test layers

| Layer | Location | Command | Env needed |
|---|---|---|---|
| Rust unit tests | `crates/*/src/tests.rs` | `cargo test --workspace` | oh-my-vllm conda env |
| GQA accuracy | `tests/test_gqa_accuracy.py` | `python tests/test_gqa_accuracy.py` | vllm conda env + GPU |
| Smoke test (E2E) | Manual (CLI) | see below | vllm env + GPU + model weights |
| Throughput benchmark | `benchmarks/compare_vllm.py` | see below | vllm env + GPU + model weights |

## Rust unit tests

```bash
export PATH="/data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin:$PATH"
export CARGO_TARGET_DIR=/data0/shared/dongwu.chen/oh-my-vllm/target
cargo test --workspace
```

**Current status (code-observed, not re-verified this session):**
- `kv-cache` crate: 47 tests pass (as of commit `ed30ef9`)
- `scheduler` crate: 8 tests pass (as of commit `4a2e520`)

These tests cover: block pool LRU operations, chain-hash prefix cache, two-phase
allocation, scheduler FCFS ordering, chunked prefill, continuous batching,
preemption by recompute, MTP draft rollback.

## GQA accuracy test

Verifies that GPU FP16 attention output is within atol=1e-2, rtol=1e-2 of a
CPU FP64 reference. Standalone — no model weights, no ZMQ.

```bash
export PATH="/data0/shared/dongwu.chen/conda-envs/vllm/bin:$PATH"
python tests/test_gqa_accuracy.py
```

**Verified result** (commit `7131da9`, 2026-09-19): 3/3 cases pass.
- `(1, 40, 8, 128, 128)` max_err=7.68e-4
- `(2, 16, 4, 64, 64)` max_err=8.37e-4
- `(1, 40, 8, 512, 128)` max_err=5.21e-4

## End-to-end smoke test

Requires: GPU, model weights at `/data0/shared/Qwen3.8-27B-FP8`, ~3 min to load.

```bash
rm -f /tmp/oh-my-vllm.ipc
export PATH="/data0/shared/dongwu.chen/conda-envs/vllm/bin:$PATH"
export PYTHONPATH="/data0/shared/dongwu.chen/oh-my-vllm/python:$PYTHONPATH"
# Use scripts/with-gpu.sh for every single-GPU test
scripts/with-gpu.sh scripts/with-env.sh ./target/release/oh-my-vllm-zmq-worker \
    --model /data0/shared/Qwen3.8-27B-FP8 \
    --num-gpu-blocks 2048 --block-size 784 --max-model-len 8192 \
    run --tokens 1 2 3 4 5 --max-tokens 16
```

**Status:** NOT YET VERIFIED. The historical config fixes are already committed;
the actual installed GPUWorker API still requires adaptation. See handoff.md.

## Throughput benchmark (REQ-PERF-001)

```bash
python benchmarks/compare_vllm.py \
    --model /data0/shared/Qwen3.8-27B-FP8 \
    --num-gpu-blocks 4096 \
    --batch-sizes 1 2 4 \
    --input-len 32768 --output-len 4096
```

**Status:** NOT YET RUN. This is the final Phase 4 gate.

## Requirements coverage

| Requirement | Test | Status |
|---|---|---|
| REQ-ACC-001 GQA accuracy | `tests/test_gqa_accuracy.py` | ✓ verified |
| REQ-FUNC-001 Chunked prefill | `crates/scheduler/src/tests.rs` | ✓ unit tested |
| REQ-FUNC-002 Continuous batching | `crates/scheduler/src/tests.rs` | ✓ unit tested |
| REQ-FUNC-003 Prefix caching | `crates/kv-cache/src/tests/` | ✓ unit tested |
| REQ-FUNC-004 Preemption | `crates/scheduler/src/tests.rs` | ✓ unit tested |
| REQ-FUNC-005 MTP spec decode | — | code exists, not E2E verified |
| REQ-GOAL-001 Single request returns answer | smoke test | **not run** |
| REQ-PERF-001 >=95% throughput | `benchmarks/compare_vllm.py` | **not run** |

## Change-triggered verification

| Change area | Must run |
|---|---|
| `crates/kv-cache/` | `cargo test --workspace` |
| `crates/scheduler/` | `cargo test --workspace` |
| `python/oh_my_vllm/worker/` | `ruff check python/` + smoke test |
| Any | pre-commit hooks (`cargo fmt`, `clippy`, `ruff`) |
| Performance-affecting | `benchmarks/compare_vllm.py` |

## Runtime foundation tests

`scripts/with-env.sh python -m unittest discover -s tests -p test_runtime_tools.py`
checks structured timestamps/correlation and GPU selection with mocked nvidia-smi.
All actual GPU tests must use scripts/with-gpu.sh, including the older examples
above. The old standalone SDPA GQA test is historical evidence only, not actual
vLLM-path coverage. Actual GQA/GDN reference tests remain required.
