# Development guide

## Environment

All Rust tools, Python packages, tests and model execution use
`/data0/shared/dongwu.chen/conda-envs/oh-my-vllm`. Use `scripts/with-env.sh` for
cargo/Python commands. It sets PATH, PYTHONPATH, CARGO_TARGET_DIR and the Worker
Python. The legacy adapter and old-environment default have been removed.

Install Python dependencies with uv targeting that environment. Direct dependencies are
in python/pyproject.toml; requirements/runtime.txt pins the complete stable
Linux/Python3.12 closure. Host CUDA13.1/compiler tools build kernels, while Torch2.14
uses its installed CUDA13.0 runtime libraries. The host supplies the NVIDIA driver.

```bash
uv pip install --python /data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python -r requirements/runtime.txt
uv pip check --python /data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python
```

The wrapper selects independent FlashInfer/TileLang/third-party Triton cache directories under
`/data0/shared/dongwu.chen/.cache/oh-my-vllm/tilelang-ffi012`. Override the common root
with OH_MY_VLLM_RUNTIME_CACHE. FlashInfer may compile kernels or download its own
versioned NVIDIA GEMM cubins on first use. These are independent library artifacts,
not old vLLM build outputs. Runtime startup records library versions; shutdown
checks imported modules and mapped libraries. No vLLM package may be installed.

`OH_MY_VLLM_ENFORCE_EAGER=1` disables target and MTP graphs for diagnostics only.
Only the explicitly authorized isolated collector may refresh the EngineCore
baseline. Never enable correctness probes during performance acceptance.

## Build and checks

The Rust workspace uses edition2024 and the existing conda toolchain.

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-env.sh cargo fmt --all
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh pre-commit run --all-files
```

Hooks are local/system hooks from .pre-commit-config.yaml. Stage only intentional
changed paths, including Cargo.lock when changed. Avoid unstaged hook edits that
conflict with pre-commit's temporary stash. Never disable hooks or use git add .

## GPU execution

```bash
scripts/with-gpu.sh scripts/with-env.sh python scripts/smoke-text.py --socket /tmp/oh-my-vllm-text.ipc --max-tokens 64
scripts/with-gpu.sh scripts/with-env.sh python benchmarks/compare_vllm.py --mode ordinary --batch-sizes 1 2 4 --output /tmp/ordinary.json
```

For multiple real prompts, `run --prompt-file PATH` reads one whitespace-separated
token-ID sequence per line (up to 32 requests). `--arrival-interval N` staggers
admission by scheduling steps; `--prefix-hit` seeds the prompts and requires an
initial cache hit. The output includes ordered token batches and feature counters.
See testing.md for the two-request `scripts/smoke-batch.py` checks with memory
pressure and inspection of generated text after preemption.

The GPU wrapper waits for an idle B200: no compute processes, at most64MiB used
and zero reported utilization. It selects a UUID and holds a cooperative flock.
Unrelated programs need not honor the lock. The benchmark additionally detects
external same-GPU clients while engines run, invalidating affected measurements.
Do not interrupt unrelated jobs. Use unique sockets; only remove a stale socket
after confirming its owner has exited.

The benchmark defaults to input32768/output4096,1024 capacity units, two warmups
and three measurements. Run ordinary, mtp and prefix modes. See testing.md for
accuracy probes, combinations and acceptance details. Never enable eager/probe
execution in acceptance runs. --binary supports isolated builds; the driver
executes a hash-verified private copy. Do not edit Python sources during a run.

## Logs and diagnostics

Logs go to stderr, results to stdout. Rust tracing and Python JSON lines use UTC
timestamps. OH_MY_VLLM_RUN_ID correlates both sides; step_id correlates RPC calls.
INFO records initialization, capacity and batch summaries. RUST_LOG=debug and
OH_MY_VLLM_LOG_LEVEL=DEBUG enable scheduling time/free blocks, RPC duration,
Python message decode, host execution and encode/send timings.

Host duration is not CUDA kernel duration. Use a separate profiling run for GPU
timing; keep traces outside the repository. Default INFO avoids per-step I/O.
The controller uses one Tokio thread for its serial scheduler/RPC stream. CPU
affinity experiments must apply the same inherited mask to both engines and record
it; taskset can be placed before the benchmark Python command. See handoff.md for
measured results, including failed cases and pending acceptance.

## Troubleshooting

- Init failure: inspect Python's traceback and conda interpreter in the launch log.
  Check pinned dependencies, checkpoint configuration and independent cache settings.
- Address in use: choose another socket or confirm the stale socket's owner exited.
- Unexpected worker exit: check stderr before rerunning; a dead child before
  connection is reported promptly instead of consuming the full init timeout.
- Insufficient startup memory: wait for an idle card again. An external job may
  have started after selection; do not lower limits to hide a contaminated pair.

## OpenAI serving development

See serving.md for launch/OMP commands and the protocol compatibility matrix.
Rust uses Axum/serde_json/tokio-stream; Python uses the pinned tokenizer and
XGrammar packages.

For CPU-only tokenizer/schema/HTTP tests on a GPU host, set
CUDA_VISIBLE_DEVICES='' before starting Python. This avoids device initialization
and does not validate CUDA execution. tests/fixtures/serving_worker.py is scripted
output for integration tests only; never use it to report model acceptance or speed.

Real constrained-output check against a running MTP4 server:

```bash
scripts/with-env.sh python scripts/serving-acceptance.py --output-dir /tmp/oh-my-vllm-serving-constraints
```

The script saves complete requests/responses; use matching response/request IDs
to inspect real MTP proposal/acceptance counters in server logs. Python DEBUG logs expose message decode, execution and encode/send time.

## Strict Rust formatting

```bash
scripts/with-env.sh cargo fmt --all
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh python -m unittest discover -s tests -p test_rust_style.py
scripts/with-env.sh pre-commit run cargo-fmt --all-files
scripts/with-env.sh pre-commit run rust-line-width --all-files
```

`rustfmt.toml` sets stable Rust 2024 formatting, 100-character width, Unix
newlines and multiline if/else and let/else expressions. Pre-commit checks
formatting instead of silently rewriting it. A separate hard-width hook reads
the same configuration and checks tracked/unignored new `.rs` files, including
macros, comments and string literals; tabs expand using rustfmt's tab size.
`rustfmt` alone may leave `json!` bodies beyond max_width untouched. Split their
fields manually and use `concat!` to wrap long literals without changing values.
Configuration and hook changes trigger both Rust style hooks.

## Isolated reference collection (authorized 2026-09-22)

`benchmarks/baseline/enginecore.py` is baseline-only tooling. Execute it with the
separately maintained vllm environment, never install vLLM into oh-my-vllm. It
requires official upstream SHA e9f169d16b9408bb9ae44f75072b91a5521d733c. The independent
`benchmarks/ttft.py` consumes its JSON and launches only our own worker. This explicit
exception does not change normal build/test/runtime dependency isolation.

For long-context runs, use `--max-model-len 262144 --num-gpu-blocks 4200
--mamba-blocks 128` before the CLI subcommand. These are provisional capacities
under GPU validation; they represent 1400 FA slots and 128 independent GDN slots.

## TileLang and TileFoundry

See [the custom kernel workflow](tilelang-development.md) for the pinned fork,
source installation, complete-operator HIR, production runtime twins and the
review/validation sequence. TileFoundry is not required to start an inference
worker. Runtime defaults to CUDA; frozen TileLang remains selectable for comparison.
Independent dependencies may still use Triton.

## Native CUDA development

CUDA is the default. Set OH_MY_VLLM_KERNEL_BACKEND=tilelang before launching Python
for the frozen comparison. scripts/with-env.sh isolates TVM_FFI_CACHE_DIR under the project runtime
cache. Native kernels compile for SM100a; the host CUDA13.1 compiler is available.
Use CUDA_HOME=/usr/local/cuda-13.1 when needed. No TileFoundry runtime dependency.
See cuda-development.md for the frozen comparison and accepted development workflow.
