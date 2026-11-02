# Environment Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `oh-my-vllm` the only environment used for repository code, with one `requirements/development.txt` and the official PyTorch 2.15.0 CUDA 13.4 wheel on Python 3.15.2.

**Architecture:** Keep Conda responsible for Python and Rust. Use one pip requirements file for runtime, development, kernel, and AgentX packages. Keep `requirements/model-reference.lock` only for an explicit reference-generation environment, and keep vLLM/SGLang environments not in this repository environment.

**Tech Stack:** Conda, Python 3.15.2, PyTorch 2.15.0+cu134, CUDA 13.4 wheel index, CUDA native compiler, pip, Rust, pre-commit.

**Spec:** `docs/superpowers/specs/2026-10-30-torch-native-runtime-design.md`

## Global Constraints

- Use one install file: `requirements/development.txt`.
- Run repository code with the `oh-my-vllm` Conda environment.
- Target Python 3.15.2 and PyTorch 2.15 or newer with the CUDA 13.4 wheel.
- Do not use a second environment for repository tests, service execution, native builds, or AgentX.
- Keep explicit reference generation and vLLM/SGLang runs as documented special cases.
- Keep host paths in ignored `LOCAL.md`.
- Do not delete or reset unrelated existing worktree changes.

## Review Focus

- A command launched not in Conda must fail instead of silently using `.venv` or a different virtual environment. test `scripts/with-env.sh` with valid and invalid environment names in Task 1.
- The resolver must select `torch==2.15.0+cu134` for CPython 3.15. test the installed version, CUDA runtime, and wheel metadata in Task 2.
- AgentX must run in the same Conda interpreter. test the launcher command identity in Task 3.
- Native build identity must record CUDA toolkit and PyTorch CUDA runtime separately. test the manifest fields in Task 4.
- Documentation must not hold stale split requirements instructions. test repository references and bilingual documentation in Task 5.

### Task 1: Enforce the repository Conda environment

**Files:**

- Modify: `scripts/with-env.sh:1-20`
- Test: `tests/test_environment.py`
- Modify: `LOCAL.md` only when local host values change

**Interfaces:**

- Produces `OH_MY_VLLM_CONDA_PREFIX` as the only accepted repository interpreter selection.
- Rejects a missing prefix, a non-Conda prefix, and a Conda environment whose name is not `oh-my-vllm`.

- [ ] Write tests for valid `oh-my-vllm`, missing prefix, and wrong environment name.
- [ ] Run `scripts/with-env.sh` tests and check the two invalid cases fail before launching the child command.
- [ ] Change `scripts/with-env.sh` to use the Conda prefix and validate `conda-meta` plus `CONDA_DEFAULT_ENV` when present.
- [ ] Run the environment tests and a shell command that prints `sys.executable`. check it is with the selected Conda prefix.
- [ ] Keep existing worker and Cargo path exports unchanged.

### Task 2: Replace split requirements with one CUDA 13.4 requirements file

**Files:**

- Modify: `requirements/development.txt`
- Delete: `requirements/agentx.txt`
- Delete: `requirements/kernel-development.txt`
- Delete: `requirements/kernels.txt`
- Delete: `requirements/runtime.txt`
- Delete: `requirements/model-reference.txt`
- Keep: `requirements/model-reference.lock` for explicit reference generation only
- Modify: `environment.yml:1-6`

**Interfaces:**

- `requirements/development.txt` contains runtime, development, kernel build, AgentX, `torch==2.15.0+cu134`, and `apache-tvm-ffi` dependencies.
- The CUDA wheel source is the official `https://download.pytorch.org/whl/cu134/` index.

- [ ] Add the merged direct dependencies and explicit `torch==2.15.0+cu134` requirement first.
- [ ] Run the resolver in dry-run mode with Python 3.15 and check no split file is call fored.
- [ ] Update `environment.yml` to Python 3.15.2 while keeping the existing Rust toolchain.
- [ ] Remove the obsolete split `.txt` files without removing the reference lock file.
- [ ] Create or update the Conda environment with `conda`. Install Python packages with the environment's `python -m pip`.
- [ ] Verify `python --version`, `torch.__version__`, `torch.version.cuda`, and `importlib.metadata.version("apache-tvm-ffi")`.

### Task 3: Make AgentX use the repository interpreter

**Files:**

- Modify: `docs/models/qwen38/agentx.md`
- Modify: `docs/models/qwen38/agentx.zh.md`
- Modify: `docs/models/deepseek_v41/agentx.md`
- Modify: `docs/models/deepseek_v41/agentx.zh.md`
- Modify: `development/serving/agentx.py` and its tests if it creates or validates an external interpreter

**Interfaces:**

- AgentX receives `$OH_MY_VLLM_CONDA_PREFIX/bin/python`.
- `ARTIFACT_ROOT` stays external and stores raw traces only.

- [ ] Add a launcher identity assertion that the AgentX interpreter equals the repository interpreter.
- [ ] Remove `AGENTX_ENV` creation and installation instructions from the two language documents.
- [ ] Run the AgentX launcher unit tests and a dry-run identity check.

### Task 4: Record wheel and native toolkit identities

**Files:**

- Modify: `development/native_build.py`
- Modify: `python/oh_my_vllm/operators/runtime/native.py`
- Test: `tests/test_native_identity.py`

**Interfaces:**

- Native manifests include `python_version`, `torch_version`, `torch_cuda_version`, `cuda_home`, and `cuda_compiler`.
- Load rejects a manifest built with a different Torch CUDA runtime or compiler identity.

- [ ] Write manifest validation tests for matching and mismatching identities.
- [ ] Run the tests to observe the mismatch failures.
- [ ] Add identity collection and validation without changing kernel source fingerprints.
- [ ] Rebuild native modules with the selected compiler toolkit and the Torch 2.15 CUDA 13.4 runtime.
- [ ] Run native identity and operator smoke tests.

### Task 5: Update instructions and remove stale references

**Files:**

- Modify: `docs/development.md`
- Modify: `docs/development.zh.md`
- Modify: `docs/serving.md`
- Modify: `docs/serving.zh.md`
- Modify: `development/serving/benchmark.py`

**Interfaces:**

- All repository install commands reference only `requirements/development.txt`.
- Benchmark source identity includes `requirements/development.txt`.

- [ ] Replace split installation commands in English and Chinese documents.
- [ ] Update benchmark identity to hash the single requirements file.
- [ ] Run documentation, whitespace, and source-reference checks.
- [ ] Search the repository and check no active command references deleted `.txt` files or `AGENTX_ENV`.

## Delivery

- Run CPU and GPU unit tests in the new Conda environment.
- Run native build checks and the existing Qwen and DeepSeek numerical smoke checks.
- Get an independent review before committing.
- Commit only the environment files and documentation for this plan.
- Push `nightly` after the review.
