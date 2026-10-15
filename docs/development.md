# Development setup

Development uses Python 3.12, Rust, uv, and Node 24 or later.
The Conda definition supplies Python and Rust. The Python and JavaScript locks pin hook dependencies.
Keep this environment separate from future inference dependencies.

## Installation

Install Conda, uv, and Node before these commands:

```bash
conda env create -f environment.yml
conda activate oh-my-vllm
uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements/development.txt
npm --prefix development/javascript ci --ignore-scripts --no-audit --no-fund
scripts/with-env.sh pre-commit install
```

For an existing environment, activate it before the last four commands.
The wrapper selects `OH_MY_VLLM_CONDA_PREFIX`, then `CONDA_PREFIX`, then `VIRTUAL_ENV`, then repository `.venv`.
Use `scripts/with-env.sh` for Python and Cargo commands.
The wrapper sets executable lookup, Python module lookup, and Cargo build output consistently for hooks and direct commands.

## Commit hooks

```bash
scripts/with-env.sh pre-commit run --all-files
```

| Hook | Contract |
|---|---|
| cuda-format | clang-format 23.1.3, native backend modules at most 800 lines |
| cargo-fmt | Rust formatter |
| rust-line-width | Hard Rust line width of 100 columns |
| definition-spacing | Syntax-based function and module spacing in Python, Rust, and JavaScript/TypeScript |
| cargo-clippy | All targets and features, warnings are errors |
| cargo-test | Existing Rust workspace tests |
| ruff-format | Python formatter |
| ruff-check | Python lint rules |
| whitespace | No trailing whitespace, one final newline |
| documentation | Bilingual pairs, word limits, portable paths, links, and source anchors |
| glossary | Generated glossary agrees with its source when a glossary exists |

Cargo hooks cover owned manifests, with each workspace once.
Hooks with no applicable files have no work. The documentation hook always runs.
The [test policy](../CONTRIBUTING.md) governs additions. Agents run existing tests without a manual step for the maintainer.
