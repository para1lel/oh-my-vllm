# oh-my-vllm

A Rust-first inference framework for Qwen3.8-27B-FP8 on a single B200 GPU.
Rust owns HTTP serving, scheduling and logical KV cache. Python runs the
project-owned Qwen model and GPU state/compute kernels using independent libraries.
They exchange msgpack over ZMQ DEALER. Runtime, builds and tests use conda
`oh-my-vllm`, without installing or importing vLLM.

The model at `/data0/shared/Qwen3.8-27B-FP8` has 16 full-attention and 48 GDN layers.
Block size is 784. See [architecture](docs/architecture.md),
[wire protocol](docs/design.md) and [decisions](docs/decisions/).

## Quick start

The wrappers select the framework environment, Python path, build directory and
independent kernel caches. GPU tests wait for an idle B200 and pin its UUID.

```bash
uv pip install --python /data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python -r requirements/runtime.txt
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/oh-my-vllm-text.ipc --max-tokens 64
```

Use unique sockets. Add `--num-speculative-tokens 4` to the text script for MTP.
Current regression uses ordinary, MTP4 and prefix-hit modes, with two full warmups
and at least five measured repetitions. The new 12-row throughput/TTFT baseline
is being collected; see [testing](docs/testing.md). `benchmarks/compare_vllm.py`
retains the prior nine-row historical comparison and its original configuration.

## Local OpenAI service

```bash
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --socket /tmp/oh-my-vllm-serve.ipc --num-gpu-blocks 4200 --mamba-blocks 128 --max-model-len 262144 --num-speculative-tokens 4 serve
```

Chat Completions and Responses are served at `http://127.0.0.1:8000/v1`, with
thinking, tools, JSON constraints and stored Responses. See [serving](docs/serving.md)
for supported schemas, OMP configuration and acceptance commands. CPU/client tests
pass. Real MTP4 constraints, lifecycle, long prefix reuse and both oh-my-pi
workflows have passed. Six 262144-total-token boundary checks pass without
preemption; formal 12-row throughput/TTFT acceptance remains pending.
See [handoff](docs/handoff.md) for current results.

## Verification and performance

The target is at least 95% of the refreshed vLLM EngineCore throughput and at most
110% of its TTFT on all 12 rows. The prior nine-row independent-runtime matrix
passed against its original frozen baseline; those historical results do not
satisfy the new TTFT gate.
[Acceptance evidence](docs/acceptance.md) separates current and historical results.

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
CUDA_VISIBLE_DEVICES='' scripts/with-env.sh python -m unittest discover -s tests -p 'test_*.py'
```

[Testing](docs/testing.md) covers actual-model FP64 probes, prefix/MTP/preemption,
service and performance checks. [Development](docs/development.md) covers the
pinned environment and logs; [profiling](docs/profiling.md) separates host and GPU
measurements. Every milestone requires independent review and a commit.

## Layout

- crates/kv-cache: logical hybrid pool, prefix cache and state reservations.
- crates/scheduler: requests, token budgets, chunk alignment, preemption and MTP.
- crates/zmq-worker: HTTP APIs, CLI, model process lifecycle and ZMQ client.
- python/oh_my_vllm/worker: execution, MTP, sampling, graphs and bridge.
- python/oh_my_vllm/models and kernels: concrete model and independent GPU operators.
- tests/probe_worker.py: actual-model FP64 diagnostic, never for throughput.
- benchmarks/ttft.py: independent measurements and current TTFT/throughput gates.
- benchmarks/baseline/enginecore.py: explicitly isolated reference collector.
- benchmarks/compare_vllm.py: historical comparison and process supervision.

Custom kernel development uses TileLang and the development-only TileFoundry
fork. See [the workflow](docs/tilelang-development.md) for installation and
operator semantics; final migration performance acceptance is still pending.
