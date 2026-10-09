# Development

## Environment installation

Install `uv`, Python 3.12 or later, and a Rust toolchain for edition 2024.
The runtime closure with satisfactory tests is `requirements/runtime.txt`.
Direct Python dependencies are in `python/pyproject.toml`.
The host supplies the NVIDIA driver.
Use compatible CUDA/compiler tools from the host or project environment.

From the repository root, create and select an environment:

```bash
uv venv --python 3.12 .venv
export OH_MY_VLLM_CONDA_PREFIX="$PWD/.venv"
uv pip install --python "$OH_MY_VLLM_CONDA_PREFIX/bin/python" -r requirements/runtime.txt
uv pip install --python "$OH_MY_VLLM_CONDA_PREFIX/bin/python" -r requirements/tools.txt
uv pip check --python "$OH_MY_VLLM_CONDA_PREFIX/bin/python"
export OH_MY_VLLM_MODEL="/path/to/Qwen3.8-27B-FP8"
```

`requirements/tools.txt` pins pytest, Ruff, pre-commit, and their constrained dependencies.
Regenerate it from `requirements/tools.in` with the runtime constraint after tool changes with satisfactory compatibility tests.

An existing conda environment is also permitted.
Set `OH_MY_VLLM_CONDA_PREFIX` to its location as an alternative to `.venv`.
Keep its location in ignored `LOCAL.md`.
Install higher-level dependencies in this environment with `uv`, not direct `pip`.
Keep vLLM and its caches isolated from the project runtime.

The combination with satisfactory compatibility tests includes Torch 2.14.0, Transformers 5.17.0, FlashInfer 0.6.18.post1, TileLang 0.1.14, and apache-tvm-ffi 0.1.12.
The recorded build used CUDA 13.1 tools and Torch CUDA 13.0 runtime libraries.
These versions show the tested combination. Use the lockfile for the full closure.

## Configuration selection

`scripts/with-env.sh` selects the environment in this order:

1. `OH_MY_VLLM_CONDA_PREFIX`.
2. Activated `VIRTUAL_ENV`.
3. Activated `CONDA_PREFIX`.
4. Repository `.venv`.

A missing selected environment is an error.
The wrapper prepends its `bin` to `PATH` and repository `python` to `PYTHONPATH`.
The wrapper sets `UV_PYTHON` to its selected Python. An explicit uv `--python` option takes precedence.

Without an explicit value, `CARGO_TARGET_DIR` uses repository `target`.
Without an explicit value, `OH_MY_VLLM_WORKER_PYTHON` uses the selected environment's Python.
Without the wrapper, Rust uses this variable or `python3` from `PATH`.

Supply `WorkerConfig.model_path` explicitly for direct Rust library calls.

| Variable | Purpose |
|---|---|
| `OH_MY_VLLM_MODEL` | Checkpoint directory. Explicit `--model` takes precedence. |
| `OH_MY_VLLM_DRAFT_MODEL` | DSpark checkpoint directory. Explicit `--draft-model` takes precedence. |
| `OH_MY_VLLM_RUNTIME_CACHE` | Independent runtime cache root for all project commands. |
| `XDG_CACHE_HOME` | Default cache base. Otherwise the user's `.cache` directory. |
| `FLASHINFER_WORKSPACE_BASE` | Override FlashInfer workspace. |
| `TRITON_CACHE_DIR` | Override third-party Triton cache. |
| `TILELANG_CACHE_DIR` | Override pinned TileLang cache. |
| `TVM_FFI_CACHE_DIR` | Override native CUDA build cache. |
| `TVM_FFI_CUDA_ARCH_LIST` | CUDA architecture. Default `10.0a` for B200. |
| `OH_MY_VLLM_KERNEL_BACKEND` | Explicit `cuda` or pinned `tilelang` backend. |
| `OH_MY_VLLM_ENFORCE_EAGER` | Explicit diagnostic mode when set to `1`. |
| `OH_MY_VLLM_MULTI_STREAM` | GDN branch streams. Default `1`. Diagnostic `0` disables them. |
| `OH_MY_VLLM_CUDA_PDL` | Owned CUDA programmatic launch. Default `1`. Diagnostic `0` disables it. |
| `OH_MY_VLLM_GPU_UUID` | Optional idle B200 selection for `scripts/with-gpu.sh`. |
| `PYTORCH_ALLOC_CONF` | Explicit allocator options. This name takes precedence over the legacy name. |
| `PYTORCH_CUDA_ALLOC_CONF` | Legacy allocator options when the new name is unset. |
| `OMP_NUM_THREADS` | Host thread count. Default `1`. |

The default cache suffix is `oh-my-vllm/tilelang-ffi012` for compatibility with the library combination with satisfactory compatibility tests.
Different subdirectories hold FlashInfer, third-party Triton, TileLang, and native CUDA artifacts.
First use can compile kernels or download versioned FlashInfer GEMM cubins.
Runtime identity records library versions and rejects accidental vLLM imports or mapped libraries.
Do not use accuracy probes or eager overrides during performance acceptance.

When the two allocator variables are unset, the wrapper sets `graph_capture_record_stream_reuse:True`.
This option uses graph dependencies to reuse storage during capture with multiple streams.
Explicit settings, with an empty value, take precedence.
The wrapper removes the legacy alias after selection. Trace identity records the allocator backend.

Set stream and PDL options before worker initialization. Formal collection must have the two options enabled.

## Draft mode selection

The legacy `--num-speculative-tokens 0` and `4` select ordinary computation and native MTP4.
`--speculative-mode none`, `mtp`, or `dspark` selects the worker mode explicitly.
A mode/count combination that is not compatible fails before worker initialization.
DSpark must have a different checkpoint through `OH_MY_VLLM_DRAFT_MODEL` or `--draft-model`.

Set the draft checkpoint and use the explicit mode:

```bash
export OH_MY_VLLM_DRAFT_MODEL="/path/to/local-DSpark-checkpoint"
scripts/with-gpu.sh scripts/with-env.sh target/release/oh-my-vllm-zmq-worker --socket /tmp/dspark-text.ipc --speculative-mode dspark bench --input-len 32 --output-len 64 --batch-size 1
```

DSpark proposes at most seven tokens. Native MTP4 keeps its four-token path.

`--dspark-confidence-threshold` sets cumulative confidence. Its initial setting is `0.2`.
Use `0.0` for fixed-count proposals up to seven. Output, context, and grammar limits can decrease the proposal count.
The threshold must be finite, at least zero, and less than one.

HTTP mode stays fixed for the worker lifetime.
`spec-bench` alone can change MTP/DSpark modes when idle with shared target weights and caches.
Use the [acceptance index](acceptance.md) for source identities and measured scope.

## Build and checks

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python scripts/check_docs.py
scripts/with-env.sh pre-commit run --all-files
```

Hooks use local system tools through the same wrapper.
Configure the environment before `git commit` so hooks select it correctly.
See [contribution procedure](../CONTRIBUTING.md).

## TileFoundry development tools

Install metadata tools before the source dependency:

```bash
git submodule update --init --recursive 3rdparty/TileFoundry
scripts/with-env.sh uv pip install --python "$OH_MY_VLLM_CONDA_PREFIX/bin/python" setuptools-scm==10.2.3 vcs-versioning==2.4.1
scripts/with-env.sh uv pip install --python "$OH_MY_VLLM_CONDA_PREFIX/bin/python" --no-build-isolation -r requirements/development.txt
scripts/with-env.sh tilefoundry --help
```

The pinned fork is a development dependency, not an inference dependency.
Do not open an upstream pull request for this fork.
See [kernel development](kernels.md).

## Deployment records

Keep host paths, addresses, selected process IDs, cache locations, and maintenance commands in ignored `LOCAL.md`.
Ignored `.local/evidence/` holds full measurements from original evidence.
The tracked [evidence index](acceptance.md) identifies portable summaries and original hashes.
New servers must have their own model location, compatible environment, GPU selection, and current-source measurement.
The repository contains no host-specific default model or environment location.
