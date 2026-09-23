# oh-my-vllm

[中文阅读版](README.zh.md) · [中文文档索引](docs/README.zh.md)

A Rust-first inference framework for Qwen3.8-27B-FP8 on a single B200 GPU.
Rust owns HTTP serving, scheduling and logical KV cache. Python runs the
project-owned Qwen model and GPU state/compute kernels using independent libraries.
They exchange msgpack over ZMQ DEALER. Runtime, builds and tests use conda
`oh-my-vllm`, without installing or importing vLLM.

The model at `/data0/shared/Qwen3.8-27B-FP8` has 16 full-attention and 48 GDN layers.
Block size is 784. See [architecture](docs/architecture.md),
[wire protocol](docs/design.md) and [decisions](docs/decisions/).

CUDA is now the default:147/147 operator cases, all12 throughput/TTFT rows,
174 correctness tests plus31 subtests, six full-context boundaries and updated
agentic readback pass. Set `OH_MY_VLLM_KERNEL_BACKEND=tilelang` for the frozen
comparison. Historical milestone counts are not current status; see
[handoff](docs/handoff.md) and [acceptance](docs/acceptance.md).

## Quick start

The wrappers select the framework environment, Python path, build directory and
independent kernel caches. GPU tests wait for an idle B200 and pin its UUID.

```bash
uv pip install --python /data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python -r requirements/runtime.txt
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/oh-my-vllm-text.ipc --max-tokens 64
```

Use unique sockets. Add `--num-speculative-tokens 4` to the text script for MTP.
Performance acceptance uses `benchmarks/ttft.py` against the frozen baseline (two
full warmups, five measured repetitions); see [testing](docs/testing.md).
`benchmarks/compare_vllm.py` is the historical nine-row tool and is not acceptance.

## Local OpenAI service

```bash
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --socket /tmp/oh-my-vllm-serve.ipc --num-gpu-blocks 4200 --mamba-blocks 128 --max-model-len 262144 --num-speculative-tokens 4 serve
```

Chat Completions and Responses are served at `http://127.0.0.1:8000/v1`, with
thinking, tools, JSON constraints and stored Responses. See [serving](docs/serving.md)
for supported schemas, OMP configuration and acceptance commands. CPU/client tests
pass. Real MTP4 constraints, lifecycle, long prefix reuse and both oh-my-pi
workflows have passed. Six 262144-total-token boundary checks pass without
preemption; all twelve formal throughput/TTFT comparisons pass.
See [handoff](docs/handoff.md) for current results.

## Verification and performance

The target is at least 95% of the refreshed vLLM EngineCore throughput and at most
110% of its TTFT on all 12 rows, with spread/median <=10% for both metrics.
The default CUDA implementation passes these gates; see
[acceptance evidence](docs/acceptance.md). Open code-audit findings (serving
availability, latent kernel/scheduler contracts, evidence gaps) are tracked in
[the audit](docs/audit-2026-09-23.md).

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
CUDA_VISIBLE_DEVICES='' scripts/with-env.sh python -m unittest discover -s tests -p 'test_*.py'
scripts/with-gpu.sh scripts/with-env.sh python -m pytest tests -q
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
- benchmarks/compare_vllm.py: historical nine-row comparison; shared process supervision.
- benchmarks/kernels.py: formal CUDA-versus-frozen-TileLang operator comparison.

Custom kernels are B200 CUDA C++/PTX ([CUDA workflow](docs/cuda-development.md)),
with a frozen TileLang comparison and development-only TileFoundry tools
([TileLang workflow](docs/tilelang-development.md)).
