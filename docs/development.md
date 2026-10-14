# Development environment

Use a CUDA-capable Linux host with one available B200, Python 3.12 or later, and Rust.
Use Node 24 LTS for the tutorial.
The host supplies the NVIDIA driver. Keep model weights separate from source.

## Installation

Initialize pinned native headers and development sources:

```bash
git submodule update --init --recursive
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements/runtime.txt -r requirements/tools.txt
uv pip install --python .venv/bin/python --no-deps -e python
```

[Runtime metadata](../python/pyproject.toml) declares dependencies. [Runtime lock](../requirements/runtime.txt) pins their closure.
[Tool inputs](../requirements/tools.in) and [tool lock](../requirements/tools.txt) keep maintenance tools separate.
Regenerate tool pins with the runtime constraint and compatibility checks.
Ninja is a build tool for fresh native CUDA modules. Keep its executable on `PATH` for compilation.

CUDA C++ uses the pinned [CUTLASS source](../3rdparty/cutlass/) and the selected CUDA toolkit's CCCL headers.
Python operator libraries are not runtime dependencies.
The current native migration must complete final GPU acceptance as recorded in [handoff](handoff.md).

## Configuration selection

Use `scripts/with-env.sh` for Cargo and Python commands.
Its [selection and defaults](../scripts/with-env.sh) are authoritative.
Set `OH_MY_VLLM_CONDA_PREFIX` for an existing environment, or activate a virtual/conda environment.
Otherwise the wrapper selects repository `.venv`.
A missing selected environment fails before command execution.

Set `OH_MY_VLLM_MODEL` or provide `--model` before model execution.
Set `OH_MY_VLLM_DRAFT_MODEL` or provide `--draft-model` for DSpark.
Keep host paths and maintenance records in ignored `LOCAL.md`.

## GPU ownership

```bash
scripts/with-gpu.sh scripts/with-env.sh python -m pytest tests/numerical/test_ir_fp8.py -q
```

The wrapper waits for an idle B200, locks it, and provides its UUID.
Stop task-owned GPU processes and children after each run. Verify their exit and released device memory.
Use process birth records. Do not stop other users' processes.
Only an explicit user request permits a service to stay active.

## Build and checks

```bash
scripts/with-env.sh cargo test --workspace
scripts/with-env.sh cargo fmt --all --check
scripts/with-env.sh python scripts/check_rust_line_width.py
scripts/with-env.sh cargo clippy --all-targets --all-features -- -D warnings
scripts/with-env.sh ruff format python/
scripts/with-env.sh ruff check python/
scripts/with-env.sh python scripts/format_cuda.py --check
scripts/with-env.sh python scripts/check_docs.py
scripts/with-env.sh pre-commit install
scripts/with-env.sh pre-commit run --all-files
scripts/test.sh cpu
```

Use [test layers and GPU selection](testing/suites.md) for scoped and complete runs.
Hooks enforce formatting, definition spacing, Rust width, bilingual documents, word caps, and source references.
Keep hooks enabled and configure their environment before committing.

## TileFoundry development tools

```bash
scripts/with-env.sh uv pip install setuptools-scm==10.2.3 vcs-versioning==2.4.1
scripts/with-env.sh uv pip install --no-build-isolation -r requirements/development.txt
scripts/with-env.sh tilefoundry --help
```

The [pinned fork](../3rdparty/TileFoundry/) is rebased to current upstream and keeps the project compatibility changes.
It is a development dependency. Production native headers have their own pin.
Do not push or open an upstream pull request without instruction.
Use the [kernel harness](kernels.md) for tuning and independent mathematics.

## Deployment records

Keep host configuration in `LOCAL.md`, temporary work and raw traces in external storage, and portable summaries in the [evidence index](acceptance.md).
A new environment must have its own checkpoint, GPU selection, and source-scoped measurements.
Historical results do not prove current acceptance.
