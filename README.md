# oh-my-vllm

A Rust-first inference framework for Qwen3.5-27B-FP8 on a single B200 GPU.
Rust owns scheduling and KV cache bookkeeping. Python wraps the installed vLLM
GPUWorker (class `Worker`) for model execution. They exchange msgpack messages
over ZMQ DEALER; the framework does not use vLLM's scheduler.

The model at `/data0/shared/Qwen3.8-27B-FP8` has 16 full-attention and 48 GDN layers.
Rust tracks two logical KV groups; Python maps these into disjoint physical
hybrid groups. Production block size is 784. See [architecture](docs/architecture.md),
[wire protocol](docs/design.md) and [decisions](docs/decisions/).

## Quick start

Use conda oh-my-vllm for framework commands and conda vllm for GPUWorker/baseline.
The wrappers set PYTHONPATH and CARGO_TARGET_DIR; no editable package install or
uv is needed. Every GPU test waits for an idle B200 and selects its UUID.

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python scripts/smoke-text.py --socket /tmp/oh-my-vllm-text.ipc --max-tokens 64
scripts/with-gpu.sh scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python benchmarks/compare_vllm.py --mode ordinary --batch-sizes 1 2 4 --output /tmp/ordinary.json
```

Use unique sockets for concurrent runs. Add `--num-speculative-tokens 4` to the text script for MTP.
Benchmark modes are ordinary, mtp and prefix. Defaults are 1024 physical blocks,
input32768, output4096, one warmup and three measured repetitions.

## Local OpenAI service

```bash
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --socket /tmp/oh-my-vllm-serve.ipc --num-speculative-tokens 4 serve
```

Chat Completions and Responses are served at `http://127.0.0.1:8000/v1`, with
thinking, tools, JSON constraints and stored Responses. See [serving](docs/serving.md)
for supported schemas, OMP configuration and acceptance commands. CPU/client tests
and real MTP4 acceptance pass; see [serving evidence](docs/acceptance.md).
The EngineCore performance results below predate this serving extension.

## Verification and performance

The target is at least 95% of matched vLLM EngineCore throughput for batch1/2/4
in ordinary, MTP and controlled prefix-hit modes. Faster than 105% also passes.
The nine required performance rows passed (97.13%–103.78% of matched vLLM).
[Acceptance evidence](docs/acceptance.md) records measurements, exact configuration
and correctness coverage; [handoff](docs/handoff.md) preserves the work history.

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
```

[Testing](docs/testing.md) explains actual-path GQA/GDN FP64 probes, coherent text,
prefix/MTP/arrival/preemption combinations and matched performance measurements.
[Development](docs/development.md) covers environment and timestamped multilevel
logs; [profiling](docs/profiling.md) distinguishes host and GPU timing.
Every milestone requires independent code review and a commit.

## Layout

- crates/kv-cache: logical hybrid pool, prefix cache, state reservations.
- crates/scheduler: requests, token budgets, chunk alignment, preemption, MTP.
- crates/zmq-worker: CLI, model-worker lifecycle and ZMQ client.
- python/oh_my_vllm/worker: model adapter, bridge, structured logs.
- tests/probe_worker.py: actual-path FP64 diagnostic (never for throughput).
- benchmarks/compare_vllm.py: paired measurements with source/binary identity.
- python/oh_my_vllm/worker/spec_decode.py: legacy helpers, outside current execution.
