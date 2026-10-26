# Development setup

Use Python 3.12, Rust, uv, and Node 24 or later for development.
The Conda definition supplies Python and Rust.
Use the hook dependency versions from the Python and JavaScript lock files.

Use different files for development and runtime dependencies.

## Installation

Install Conda, uv, and Node first.
Do these commands:

```bash
conda env create -f environment.yml
conda activate oh-my-vllm
uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/development.txt -r requirements/runtime.txt
npm --prefix development/javascript ci --ignore-scripts --no-audit --no-fund
scripts/with-env.sh pre-commit install
```

If the environment is available, do the last four commands.
The wrapper selects `OH_MY_VLLM_CONDA_PREFIX`, then `CONDA_PREFIX`, then `VIRTUAL_ENV`, then the repository `.venv`.

Use `scripts/with-env.sh` for Python and Cargo commands.
The wrapper sets the executable paths, Python module paths, worker interpreter, and Cargo build output.

Install the [runtime packages](../requirements/runtime.txt) for service work.

## Git hooks

```bash
scripts/with-env.sh pre-commit run --all-files
```

| Hook | Contract |
|---|---|
| cuda-format | clang-format 23.1.3, native backend modules at most 800 lines |
| cargo-fmt | Rust formatter |
| rust-line-width | Rust line width at most 100 columns |
| definition-spacing | Function and module spacing through Python, Rust, and JavaScript/TypeScript syntax trees |
| cargo-clippy | All workspace members, targets, and features, with warnings as errors |
| cargo-test | Rust workspace tests in the repository and the Python service tests |
| kernel-contracts | Model evidence, HTTP accounting, and worker constraints |
| ruff-format | Python formatter |
| ruff-check | Python lint rules |
| whitespace | No trailing whitespace, one final newline |
| documentation | Translation pairs, word limits, host paths, links, and source anchors |
| glossary | Generated glossary agrees with its source |

The Cargo hooks include all owned manifests. They process each workspace once.
A hook has no work if there are no applicable files. The documentation hook always runs.

Use the [test policy](../CONTRIBUTING.md) before you add tests.
Agents do the tests in the repository for the maintainer.

## Native modules

Use CUDA 13.1. Keep host paths in ignored `LOCAL.md`.
Set `CUDA_HOME` to the CUDA installation directory.
Install the kernel development packages:

```bash
scripts/with-env.sh uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/kernel-development.txt
```

The inference bindings are in `requirements/kernels.txt`.
Development packages are not worker dependencies.
Set `CUTLASS_ROOT` to an external dependency directory.
Get the specified headers:

```bash
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass.git "$CUTLASS_ROOT"
export OH_MY_VLLM_CUTLASS="$CUTLASS_ROOT"
```

The build accepts only clean Git revision `098de2a652cf8f00fd70b2df54051c7eccbb855a`.
Set `OH_MY_VLLM_KERNEL_CACHE` to an external build directory.
Build the native modules:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.native_build
```

The inference loader does not use development imports or a compiler.
It checks source and binary identities against the build manifest.
Each binary has an immutable copy and manifest in the external cache.

For model commands, use [Qwen validation](models/qwen38/README.md).
