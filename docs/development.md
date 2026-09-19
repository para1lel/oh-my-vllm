# Development Guide — oh-my-vllm

## Environment setup

### Two conda environments in use

| Purpose | Path | Contains |
|---|---|---|
| Rust build + Python lint | `/data0/shared/dongwu.chen/conda-envs/oh-my-vllm` | cargo, ruff, pre-commit |
| Model runner | `/data0/shared/dongwu.chen/conda-envs/vllm` | Python 3.12, torch, vllm, CUDA |

The Python worker subprocess (`oh_my_vllm.worker.zmq_bridge`) must run under the
**vllm** env. The Rust binary spawns `python` — it finds whatever `python` is on
`PATH`. Always set both before any end-to-end run:

```bash
export PATH="/data0/shared/dongwu.chen/conda-envs/vllm/bin:$PATH"
export PYTHONPATH="/data0/shared/dongwu.chen/oh-my-vllm/python:$PYTHONPATH"
export CARGO_TARGET_DIR=/data0/shared/dongwu.chen/oh-my-vllm/target
export CUDA_VISIBLE_DEVICES=0   # single B200
```

The oh-my-vllm Python package is **not** pip-installed into the vllm env — only
`PYTHONPATH` makes it importable. `msgpack` and `pyzmq` are required in that env:

```bash
# Run once in the vllm env if missing:
pip install msgpack pyzmq
```

### Rust build

```bash
# Builds the inference binary (release, ~30s on first build):
export PATH="/data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin:$PATH"
export CARGO_TARGET_DIR=/data0/shared/dongwu.chen/oh-my-vllm/target
cargo build --release --bin oh-my-vllm-zmq-worker
# Output: target/release/oh-my-vllm-zmq-worker
```

Rust edition: 2024. Workspace root: `Cargo.toml`. Three crates:
- `crates/kv-cache` — KV cache, prefix cache
- `crates/scheduler` — request scheduler
- `crates/zmq-worker` — binary, ZMQ client

### Pre-commit hooks

All hooks run via `scripts/with-env.sh` which activates the oh-my-vllm env:

```bash
cargo fmt --all               # Rust formatting
cargo clippy … -D warnings    # Rust linting (warnings = errors)
cargo test --all               # Rust tests
ruff format --force-exclude    # Python formatting
ruff check --force-exclude --fix  # Python linting
scripts/fix_whitespace.py      # trailing whitespace + final newline
```

**Known trap:** the hook stashes unstaged files before formatting. If `Cargo.lock`
is unstaged, it will be stashed and then the hook may fail to pop it because
`cargo fmt` touched it. Always `git add Cargo.lock` alongside Rust changes.

Run hooks manually:

```bash
pre-commit run --all-files
# or per-hook:
scripts/with-env.sh cargo fmt --all
scripts/with-env.sh ruff format python/
```

## Build commands (source of truth: `.pre-commit-config.yaml`, `Cargo.toml`)

```bash
# Format all Rust
cargo fmt --all

# Lint all Rust (strict)
cargo clippy --all-targets --all-features -- -D warnings

# Run all Rust tests
cargo test --workspace

# Format Python
ruff format python/

# Lint Python
ruff check python/

# Build release binary
cargo build --release --bin oh-my-vllm-zmq-worker
```

## Smoke test (requires GPU, ~3 min to load model)

```bash
rm -f /tmp/oh-my-vllm.ipc   # clean up stale socket if any
export PATH="/data0/shared/dongwu.chen/conda-envs/vllm/bin:$PATH"
export PYTHONPATH="/data0/shared/dongwu.chen/oh-my-vllm/python:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
./target/release/oh-my-vllm-zmq-worker \
    --model /data0/shared/Qwen3.8-27B-FP8 \
    --num-gpu-blocks 2048 \
    --block-size 784 \
    --max-model-len 8192 \
    run --tokens 1 2 3 4 5 --max-tokens 16
# Expected: "output token ids: [...]"
```

## Benchmark comparison

```bash
python benchmarks/compare_vllm.py \
    --model /data0/shared/Qwen3.8-27B-FP8 \
    --num-gpu-blocks 4096 \
    --batch-sizes 1 2 4 \
    --input-len 2048 --output-len 512
```

## External dependencies (not in repo)

| Dependency | Location | Notes |
|---|---|---|
| Qwen3.5-27B-FP8 weights | `/data0/shared/Qwen3.8-27B-FP8` | ~55 GB, do not copy |
| vLLM (Python) | conda vllm env | version pinned in that env |
| CUDA | system | B200 requires CUDA 12.6+ |

## Troubleshooting

**"Address already in use"** — stale IPC socket from a previous run: `rm -f /tmp/oh-my-vllm.ipc`

**"Not connected to peers"** — Rust sent Init before Python connected. The retry
loop in `client.rs:launch()` handles this; if it still fails, the Python worker
likely crashed on import. Check Python stderr.

**`CacheConfig` or `SchedulerConfig` validation errors** — the vllm conda env's
vLLM version may differ from what the code expects. Check `_handle_init()` in
`python/oh_my_vllm/worker/zmq_bridge.py` and compare against the actual signatures:
```bash
python -c "import inspect; from vllm.config import CacheConfig; print(inspect.signature(CacheConfig))"
```

**Pre-commit hook stash conflict** — run `cargo fmt && cargo clippy --fix --allow-dirty`,
then `ruff format python/ && ruff check --fix python/`, then stage everything
including `Cargo.lock` before committing.
