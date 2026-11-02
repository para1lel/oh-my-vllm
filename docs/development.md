# Development setup

Use Python 3.15.2, Rust, pip, and Node 24 or later for development.
The Conda definition supplies Python and Rust.
`requirements/runtime.txt` supplies the inference packages.
`requirements/development.txt` supplies the development packages and includes the runtime file.

## Installation

Install Conda and Node first.
Do these commands:

```bash
conda env create -f environment.yml
conda activate oh-my-vllm
export OH_MY_VLLM_CONDA_PREFIX="$CONDA_PREFIX"
PYZMQ_BACKEND=cffi python -m pip install --ignore-requires-python -r requirements/development.txt
npm --prefix development/javascript ci --ignore-scripts --no-audit --no-fund
scripts/with-env.sh pre-commit install
```

If the environment is available, do the last four commands.
The wrapper accepts only `OH_MY_VLLM_CONDA_PREFIX` and the `oh-my-vllm` Conda environment.
The `PYZMQ_BACKEND=cffi` setting builds the CPython 3.15 binding against the
selected Conda ZeroMQ library.
The `--ignore-requires-python` option is required because the current AgentX
wheel has an outdated Python upper bound.

Use `scripts/with-env.sh` for Python and Cargo commands.
The wrapper sets the executable paths, Python module paths, worker interpreter, and Cargo build output.
Do not create a repository virtual environment.
The development installation includes the runtime packages.

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

Use a CUDA 13.4 toolkit for native builds. Keep host paths in ignored `LOCAL.md`.
Set `CUDA_HOME` to the CUDA installation directory.
The development file installs the native build packages and includes `requirements/runtime.txt`.
If CUDA 13.4 is not available, stop before a native rebuild. Do not mix CUDA
13.2 headers with the CUDA 13.4 Torch runtime.

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
