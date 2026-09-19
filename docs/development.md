# Development guide

## Environments

| Purpose | Conda environment |
|---|---|
| Rust, framework Python, lint and hooks | /data0/shared/dongwu.chen/conda-envs/oh-my-vllm |
| GPUWorker, torch, vLLM and baseline | /data0/shared/dongwu.chen/conda-envs/vllm |

Use scripts/with-env.sh for every cargo/Python command. It sets PYTHONPATH to the
repository's python/ directory, CARGO_TARGET_DIR to target/, the framework PATH,
and OH_MY_VLLM_WORKER_PYTHON to the absolute vllm conda Python. The worker package
is not installed in that environment. No uv or conda package installation is used.
If a Python dependency is missing, use the vllm environment's python -m pip through
with-env.sh; do not install dependencies without first checking what is missing.

The vLLM environment uses an editable checkout at /data0/shared/dongwu.chen/vllm.
Installed version metadata alone is not an immutable source pin. Record the
actual source/configuration and executable identity for measurements.

The ignored .vscode/settings.json selects conda oh-my-vllm for Python analysis
and adds ${workspaceFolder}/python. Execution of GPUWorker still uses conda vllm.

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
scripts/with-gpu.sh scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python scripts/smoke-text.py --socket /tmp/oh-my-vllm-text.ipc --max-tokens 64
scripts/with-gpu.sh scripts/with-env.sh /data0/shared/dongwu.chen/conda-envs/vllm/bin/python benchmarks/compare_vllm.py --mode ordinary --batch-sizes 1 2 4 --output /tmp/ordinary.json
```

The GPU wrapper waits for an idle B200: no compute processes, at most64MiB used
and zero reported utilization. It selects a UUID and holds a cooperative flock.
Unrelated programs need not honor the lock. The benchmark additionally detects
external same-GPU clients while engines run, invalidating affected measurements.
Do not interrupt unrelated jobs. Use unique sockets; only remove a stale socket
after confirming its owner has exited.

The benchmark defaults to input32768/output4096,1024 physical blocks, one warmup
and three measurements. Run ordinary, mtp and prefix modes. See testing.md for
accuracy probes, combinations and acceptance details. Never enable eager/probe
execution in acceptance runs. --binary supports isolated builds; the driver
executes a hash-verified private copy. Do not edit Python/vLLM sources during a run.

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
  The Worker API/configuration must match the local editable checkout.
- Address in use: choose another socket or confirm the stale socket's owner exited.
- Unexpected worker exit: check stderr before rerunning; a dead child before
  connection is reported promptly instead of consuming the full init timeout.
- Insufficient startup memory: wait for an idle card again. An external job may
  have started after selection; do not lower limits to hide a contaminated pair.
