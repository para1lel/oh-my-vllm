# Development setup

Use Python 3.15.2, Rust, pip, and Node 24 or later for development.
The Conda definition supplies Python and Rust.
`requirements/runtime.txt` supplies inference packages.
`requirements/development.txt` supplies development packages and includes the runtime file.
Keep host settings in ignored `LOCAL.md`.

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
scripts/with-env.sh python scripts/check.py doctor
```

If the environment is available, start with the export command.
The wrapper accepts only `OH_MY_VLLM_CONDA_PREFIX` and the `oh-my-vllm` Conda environment.
`PYZMQ_BACKEND=cffi` builds the CPython 3.15 binding against the selected Conda ZeroMQ library.
The AgentX wheel has an outdated Python upper bound. Use `--ignore-requires-python` for its installation.

Use `scripts/with-env.sh` for Python and Cargo commands.
The wrapper sets executable paths, Python module paths, the worker interpreter, and Cargo build output.
Use the Conda environment for repository commands.
The development installation includes the runtime packages.

## Development loop

Use the [check CLI](../scripts/check.py) to select a check level:

```bash
scripts/with-env.sh python scripts/check.py --help
```

| Command after the wrapper and Python | Scope |
|---|---|
| `scripts/check.py doctor` | Tools, specified parser versions, parser operation, installed Git hook |
| `scripts/check.py size` | All repository code files and function spans |
| `scripts/check.py static` | Read-only size, format, lint, spacing, whitespace, document, and glossary checks |
| `scripts/check.py cpu` | Rust workspace tests and discovered Python tests with CUDA disabled |
| `scripts/check.py hooks` | All configured pre-commit hooks, with their usual fixes |

Use `static` during edits.
Use focused CPU checks for the behavior you change:

```bash
scripts/with-env.sh python scripts/check.py cpu --list
scripts/with-env.sh python scripts/check.py cpu --python-only tests.test_constraints
```

The first command lists unique Python test IDs without running tests or Cargo.
The second runs the selected Python module.
For the full CPU scope, use `cpu` without options.
Cargo supplies the executable for the Python HTTP service cases. The Python discovery runner does not repeat those cases.

Use [the verification skill](../.claude/skills/verify-oh-my-vllm/SKILL.md) to drive visible CLI and HTTP behavior.
Its CPU fixture checks service and transport only.
It does not prove GPU inference, numerical correctness, model capacity, or performance.
Use the [test scope](testing.md) to select model acceptance checks.
Run all hooks after the final changes for delivery.
Repeat checks when a change or failure gives a new reason.

## Code size

All repository code files have a limit of 500 physical lines.
All functions have a limit of 80 physical lines.
The limits apply to production code, tests, tools, CUDA headers, and hidden skill helpers.
The [size checker](../scripts/check_code_size.py) uses syntax trees for function spans.
Blank lines and comments in a span count as physical lines.
The checker accepts LF and CRLF line endings and rejects a CR not followed by LF.

Use the inventory to see each file, function, line span, and violation:

```bash
scripts/with-env.sh python scripts/check.py size --inventory --json > "$ARTIFACT_ROOT/code-size.json"
scripts/with-env.sh python scripts/check_code_size.py --all
```

Set `ARTIFACT_ROOT` to external storage before the inventory command.
Inventory mode reports size violations without a failing exit code.
Parse errors and missing tools cause failure.
The size check fails on size violations unless inventory mode is selected.

Split code by responsibility when it exceeds a limit.
Keep public imports and CLI entry points stable where callers depend on them.
Use small functions with clear inputs and outputs.
Keep the existing numerical order, resource ownership, error behavior, and frozen acceptance values.

## Git hooks

```bash
scripts/with-env.sh python scripts/check.py hooks
```

The [pre-commit configuration](../.pre-commit-config.yaml) is the hook source.
The hooks check code size, formatting, lint, Rust behavior, Python contracts, documents, and glossary consistency.
The full run includes Clippy and tests.
The static level omits Clippy and tests.
The hooks can change formatted files and source link line numbers.
Review these changes before staging.

Keep hooks active.
Use the [contribution procedure](../CONTRIBUTING.md#delivery) for review and Git delivery.
Keep raw logs and evidence external to the repository.
Stop owned workers and temporary services after checks.

## Native modules

Use a CUDA 13.4 toolkit for native builds.
Set `CUDA_HOME` to its installation directory in local configuration.
The development file installs native build packages and includes `requirements/runtime.txt`.
If CUDA 13.4 is unavailable, stop before a native rebuild.
Do not mix CUDA 13.2 headers with the CUDA 13.4 Torch runtime.

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
Source moves can change these identities. Build changed modules before checkpoint inference.
Use [the service guide](serving.md) for startup and the selected model's guide for acceptance.
